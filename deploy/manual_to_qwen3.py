"""GleamLM 手工轨 checkpoint → HF Qwen3 格式（供 vLLM 原生加载）。

背景:
  手工轨模型（GleamLMModel，RMSNorm/RoPE/GQA/QK-Norm/SwiGLU/weight tying）
  与 Qwen3 架构逐项同构。把手工轨 checkpoint（sft_best.pt / dpo_best.pt）
  映射为 Qwen3 HF 键，输出 config.json + model.safetensors + tokenizer，
  vLLM 原生支持 Qwen3ForCausalLM，无需 trust_remote_code。

用法:
  python deploy/manual_to_qwen3.py \
    --input checkpoints/nano/sft/sft_best.pt \
    --output ./exported \
    --tokenizer-path gleamlm/tokenizer/checkpoints/bbpe_12k

  vllm serve ./exported --dtype bfloat16 --max-model-len 4096
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def convert(gleamlm_ckpt: str, output_dir: str, tokenizer_dir: str | None) -> str:
    """手工轨 checkpoint → Qwen3 HF 目录。

    键映射（手工轨分离键 → HF Qwen3）:
      model.token_embed.weight          → model.embed_tokens.weight
      model.layers.N.attn_norm.weight   → model.layers.N.input_layernorm.weight
      model.layers.N.ffn_norm.weight    → model.layers.N.post_attention_layernorm.weight
      model.layers.N.attn.W_q/W_k/W_v   → model.layers.N.self_attn.{q,k,v}_proj
      model.layers.N.attn.W_o           → model.layers.N.self_attn.o_proj
      model.layers.N.attn.q_norm/k_norm → model.layers.N.self_attn.{q,k}_norm
      model.layers.N.ffn.W_gate/up/down → model.layers.N.mlp.{gate,up,down}_proj
      model.final_norm.weight           → model.norm.weight
      model.lm_head.weight              → lm_head（tied 省略，靠 config 重建）
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt = torch.load(gleamlm_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    if not any(k.startswith("model.") for k in sd):
        sd = {f"model.{k}": v for k, v in sd.items()}

    from gleamlm.utils.config import extract_checkpoint_config

    cfg = extract_checkpoint_config(ckpt)

    def layer(n: int, sub: str) -> str:
        return f"model.layers.{n}.{sub}"

    num_layers = cfg["num_layers"]
    hf_sd: dict[str, torch.Tensor] = {}
    for n in range(num_layers):
        # norm
        hf_sd[f"model.layers.{n}.input_layernorm.weight"] = sd[layer(n, "attn_norm.weight")]
        hf_sd[f"model.layers.{n}.post_attention_layernorm.weight"] = sd[layer(n, "ffn_norm.weight")]
        # attention（分离键直接映射）
        hf_sd[f"model.layers.{n}.self_attn.q_proj.weight"] = sd[layer(n, "attn.W_q.weight")]
        hf_sd[f"model.layers.{n}.self_attn.k_proj.weight"] = sd[layer(n, "attn.W_k.weight")]
        hf_sd[f"model.layers.{n}.self_attn.v_proj.weight"] = sd[layer(n, "attn.W_v.weight")]
        hf_sd[f"model.layers.{n}.self_attn.o_proj.weight"] = sd[layer(n, "attn.W_o.weight")]
        hf_sd[f"model.layers.{n}.self_attn.q_norm.weight"] = sd[layer(n, "attn.q_norm.weight")]
        hf_sd[f"model.layers.{n}.self_attn.k_norm.weight"] = sd[layer(n, "attn.k_norm.weight")]
        # MLP
        hf_sd[f"model.layers.{n}.mlp.gate_proj.weight"] = sd[layer(n, "ffn.W_gate.weight")]
        hf_sd[f"model.layers.{n}.mlp.up_proj.weight"] = sd[layer(n, "ffn.W_up.weight")]
        hf_sd[f"model.layers.{n}.mlp.down_proj.weight"] = sd[layer(n, "ffn.W_down.weight")]

    hf_sd["model.embed_tokens.weight"] = sd["model.token_embed.weight"]
    hf_sd["model.norm.weight"] = sd["model.final_norm.weight"]
    # lm_head 省略（tied），靠 tie_word_embeddings 重建

    head_dim = cfg["d_model"] // cfg["num_heads"]
    hf_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": cfg["vocab_size"],
        "hidden_size": cfg["d_model"],
        "intermediate_size": cfg["d_ff"],
        "num_hidden_layers": cfg["num_layers"],
        "num_attention_heads": cfg["num_heads"],
        "num_key_value_heads": cfg["num_kv_heads"],
        "head_dim": head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": cfg["max_seq_len"],
        "rms_norm_eps": 1e-5,
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "tie_word_embeddings": cfg.get("tie_weights", True),
        "attention_bias": False,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "transformers_version": "4.57.6",
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)

    from safetensors.torch import save_file

    save_file(hf_sd, os.path.join(output_dir, "model.safetensors"))

    if tokenizer_dir:
        from gleamlm.tokenizer.tokenizer import BBPETokenizer

        BBPETokenizer.load(tokenizer_dir).export_to_hf_format(output_dir)

    print(f"手工轨 → Qwen3 HF 导出完成: {gleamlm_ckpt} → {output_dir}")
    print(f"  {cfg['num_layers']}L × {cfg['d_model']}d, "
          f"GQA {cfg['num_heads']}/{cfg['num_kv_heads']}, "
          f"ffn {cfg['d_ff']}, head_dim {head_dim}")
    print(f"  {len(hf_sd)} 个权重键（lm_head 省略，tied 由 config 重建）")
    return output_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="手工轨 checkpoint → Qwen3 HF (vLLM)")
    p.add_argument("--input", required=True, help="手工轨 checkpoint .pt (sft_best.pt / dpo_best.pt)")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--tokenizer-path", default=None, help="BBPE tokenizer 目录")
    args = p.parse_args()
    convert(args.input, args.output, args.tokenizer_path)
