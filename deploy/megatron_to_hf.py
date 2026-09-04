"""Megatron GPTModel 预训练产物 → HF Qwen3 格式（供 vLLM 原生加载）。

背景:
  项目工业轨用 megatron-core GPTModel 预训练（GQA/SwiGLU/RMSNorm/RoPE/QK-Norm，
  与手写轨架构对齐），产物是 torch.save 的 megatron state_dict。
  本项目架构与 Qwen3 逐项同构，而 vLLM 原生支持 Qwen3ForCausalLM ——
  因此把 megatron 键改名/拆分为 Qwen3 HF 键，输出 config.json + model.safetensors
  + tokenizer.json，vLLM 直接 `vllm serve ./exported` 加载，无需 trust_remote_code。

用法:
  python deploy/megatron_to_hf.py \
    --input checkpoints/megatron/megatron_final.pt \
    --output ./exported \
    --tokenizer-path gleamlm/tokenizer/checkpoints/bbpe_24k \
    --vocab-size 24002

  然后:
  python -m vllm.entrypoints.openai.api_server --model ./exported/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def convert(megatron_ckpt: str, output_dir: str, tokenizer_dir: str | None, vocab_size: int) -> str:
    """megatron GPTModel state_dict → Qwen3 HF 目录。

    键名映射（参考 NVIDIA Megatron-Bridge Qwen3 bridge）:
      embedding.word_embeddings          → model.embed_tokens
      decoder.layers.N.self_attention.linear_qkv → {q,k,v}_proj（QKV 拆分）
      decoder.layers.N.self_attention.linear_proj → o_proj
      decoder.layers.N.self_attention.{q,k}_layernorm → {q,k}_norm
      decoder.layers.N.mlp.linear_fc1    → {gate,up}_proj（chunk 切半）
      decoder.layers.N.mlp.linear_fc2    → down_proj
      decoder.{input_layernorm,pre_mlp_layernorm,final_layernorm} → 同名
      output_layer                       → lm_head（tied 省略，靠 config 重建）
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt = torch.load(megatron_ckpt, map_location="cpu", weights_only=True)
    sd = ckpt["model"]
    cfg = ckpt.get("config", {})
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    num_layers = m.get("num_layers") or _infer_layers(sd)
    hidden_size = m.get("hidden_size") or _infer_hidden(sd)
    num_heads = m.get("num_attention_heads") or 1
    num_kv_heads = m.get("num_query_groups") or num_heads
    # megatron 0.16 ffn_hidden_size = gate+up 合并宽度；
    # Qwen3 intermediate_size = 单份 gate 宽度 = fc1/2
    ffn_hidden = _infer_ffn(sd) or (m.get("ffn_hidden_size", 0) // 2)

    hf_sd: dict[str, torch.Tensor] = {}

    def layer_key(n: int, sub: str) -> str:
        return f"decoder.layers.{n}.{sub}"

    # ── embedding（vocab 切片，处理 megatron padding）──
    emb = sd["embedding.word_embeddings.weight"]
    emb = emb[:vocab_size] if emb.shape[0] > vocab_size else emb
    hf_sd["model.embed_tokens.weight"] = emb

    # ── 逐层 ──
    for n in range(num_layers):
        lk = lambda s: layer_key(n, s)  # noqa: E731
        # norm（RMSNorm 无 bias）
        hf_sd[f"model.layers.{n}.input_layernorm.weight"] = sd[lk("input_layernorm.weight")]
        hf_sd[f"model.layers.{n}.post_attention_layernorm.weight"] = sd[lk("pre_mlp_layernorm.weight")]

        # attention: QKV 拆分（megatron Q→K→V 连续）
        qkv = sd[lk("self_attention.linear_qkv.weight")]  # [Q+K+V, H]
        q_out = num_heads * (hidden_size // num_heads)
        kv_out = num_kv_heads * (hidden_size // num_heads)
        q, k, v = qkv[:q_out], qkv[q_out:q_out + kv_out], qkv[q_out + kv_out:]
        hf_sd[f"model.layers.{n}.self_attn.q_proj.weight"] = q
        hf_sd[f"model.layers.{n}.self_attn.k_proj.weight"] = k
        hf_sd[f"model.layers.{n}.self_attn.v_proj.weight"] = v
        hf_sd[f"model.layers.{n}.self_attn.o_proj.weight"] = sd[lk("self_attention.linear_proj.weight")]
        # QK-Norm
        hf_sd[f"model.layers.{n}.self_attn.q_norm.weight"] = sd[lk("self_attention.q_layernorm.weight")]
        hf_sd[f"model.layers.{n}.self_attn.k_norm.weight"] = sd[lk("self_attention.k_layernorm.weight")]

        # MLP: gate/up 拆分（chunk 切半，与 megatron 原生 glu 一致）
        fc1 = sd[lk("mlp.linear_fc1.weight")]
        half = fc1.shape[0] // 2
        gate, up = fc1[:half], fc1[half:]
        hf_sd[f"model.layers.{n}.mlp.gate_proj.weight"] = gate
        hf_sd[f"model.layers.{n}.mlp.up_proj.weight"] = up
        hf_sd[f"model.layers.{n}.mlp.down_proj.weight"] = sd[lk("mlp.linear_fc2.weight")]

    # ── 最后 norm ──
    hf_sd["model.norm.weight"] = sd["decoder.final_layernorm.weight"]

    # ── config.json（Qwen3 字段）──
    head_dim = hidden_size // num_heads
    rope_theta = m.get("rope_theta", 10000.0)
    hf_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": m.get("max_position_embeddings", 4096),
        "rms_norm_eps": m.get("layernorm_epsilon", m.get("rms_norm_eps", 1e-5)),
        "rope_theta": rope_theta,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "transformers_version": "4.57.6",
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)

    # ── safetensors（tied: 不写 lm_head，靠 tie_word_embeddings 重建）──
    from safetensors.torch import save_file
    save_file(hf_sd, os.path.join(output_dir, "model.safetensors"))

    # ── tokenizer ──
    if tokenizer_dir:
        from gleamlm.tokenizer.tokenizer import BBPETokenizer
        BBPETokenizer.load(tokenizer_dir).export_to_hf_format(output_dir)

    print(f"megatron → Qwen3 HF 导出完成: {megatron_ckpt} → {output_dir}")
    print(f"  {num_layers}L × {hidden_size}d, GQA {num_heads}/{num_kv_heads}, "
          f"ffn {ffn_hidden}, head_dim {head_dim}, rope_theta {rope_theta}")
    print(f"  {len(hf_sd)} 个权重键（lm_head 已省略，tied 由 config 重建）")
    return output_dir


def _infer_layers(sd: dict) -> int:
    return max(int(k.split(".")[2]) for k in sd if k.startswith("decoder.layers.")) + 1


def _infer_hidden(sd: dict) -> int:
    return sd["embedding.word_embeddings.weight"].shape[1]


def _infer_ffn(sd: dict) -> int:
    # fc1 宽度 = gate+up 合并，intermediate_size 取单份
    k = [k for k in sd if k.startswith("decoder.layers.0.mlp.linear_fc1.weight")]
    return sd[k[0]].shape[0] // 2 if k else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Megatron GPTModel → Qwen3 HF (vLLM)")
    p.add_argument("--input", required=True, help="megatron_final.pt")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--tokenizer-path", default=None, help="BBPE tokenizer 目录")
    p.add_argument("--vocab-size", type=int, default=24002, help="真实 vocab size")
    args = p.parse_args()
    convert(args.input, args.output, args.tokenizer_path, args.vocab_size)
