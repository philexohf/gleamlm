"""Megatron → HF 转换器（按工业惯例）: 收敛分片 → key 重映射 + 反融合 QKV/GLU → HF 格式。

输入: industrial 轨 checkpoints/.../megatron_final.pt 或 iter_*.pt（单 rank 全量 state）。
输出: HF 目录（config.json + model.safetensors），可由 hf/ 的 GleamLMForCausalLM 加载。

权重映射依据（实测 megatron-core 0.16 mcore 布局）:
  - fused linear_qkv 按 [num_query_groups, (qpg+2)*head_dim] 排布:
    每组 = q(qpg 个 q head) | k(head_dim) | v(head_dim)
    → 拆回 W_q/W_k/W_v（手工轨 head-major 布局）
  - GLU linear_fc1 前半 = silu 门 (gate)、后半 = up → 拆回 W_gate/W_up
  - *._extra_state 为 TE/fsdp 占位 (None)，直接丢弃

用法:
  python tools/convert_megatron_to_hf.py \
      --ckpt checkpoints/nano_megatron/megatron_final.pt \
      --config industrial/configs/nano.yaml \
      --out hf/checkpoints/nano_megatron_hf

  # 验证模式: 与原始 Megatron 模型同输入对比 logits（需单卡 + dist 环境变量）
  RANK=0 WORLD_SIZE=1 ... python tools/convert_megatron_to_hf.py ... --verify
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hf.hf_config import gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM

# Megatron 反融合索引（见模块 docstring）
def _defuse_qkv(fused: torch.Tensor, n_q: int, n_kv: int, head_dim: int):
    """fused (n_q+n_kv+n_kv)*head_dim × d → (W_q, W_k, W_v)。"""
    qpg = n_q // n_kv
    Wq = torch.zeros(n_q * head_dim, fused.size(1), dtype=fused.dtype)
    Wk = torch.zeros(n_kv * head_dim, fused.size(1), dtype=fused.dtype)
    Wv = torch.zeros(n_kv * head_dim, fused.size(1), dtype=fused.dtype)
    for g in range(n_kv):
        base = g * (qpg + 2) * head_dim
        for l in range(qpg):  # q head 全局 id = g*qpg + l
            i = g * qpg + l
            Wq[i * head_dim : (i + 1) * head_dim] = fused[
                base + l * head_dim : base + (l + 1) * head_dim
            ]
        Wk[g * head_dim : (g + 1) * head_dim] = fused[
            base + qpg * head_dim : base + (qpg + 1) * head_dim
        ]
        Wv[g * head_dim : (g + 1) * head_dim] = fused[
            base + (qpg + 1) * head_dim : base + (qpg + 2) * head_dim
        ]
    return Wq, Wk, Wv


def _defuse_glu(fc1: torch.Tensor, d_ff: int):
    """fc1 (2*d_ff × d) → (W_gate, W_up)。前半为 silu 门。"""
    return fc1[:d_ff], fc1[d_ff:]


def _core_config_from_yaml(path: str) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    m = cfg["model"]
    return {
        "vocab_size": m["vocab_size"],
        "d_model": m["hidden_size"],
        "num_layers": m["num_layers"],
        "num_heads": m["num_attention_heads"],
        "num_kv_heads": m.get("num_query_groups", m["num_attention_heads"]),
        "d_ff": m["ffn_hidden_size"],
        "max_seq_len": m["max_position_embeddings"],
        "tie_weights": True,
        "use_flash_attn": False,
        "attn_type": "gqa",
        "ffn_type": "mlp",
        "pad_token_id": 0,
        "rope_theta": 10000.0,
        "rope_scale": 1.0,
        "rope_factor": 8.0,
    }


def convert(megatron_sd: dict, cfg: dict) -> dict:
    """Megatron state_dict → HF wrapper state_dict（key = model.*）。"""
    sd = {k: v for k, v in megatron_sd.items() if v is not None}
    hf_config = gleamlm_config_from_core(cfg)
    target = GleamLMForCausalLM(hf_config)
    target_sd = target.state_dict()

    out: dict[str, torch.Tensor] = {}
    n_layers = cfg["num_layers"]
    n_q, n_kv = cfg["num_heads"], cfg["num_kv_heads"]
    head_dim = cfg["d_model"] // n_q

    # embedding + final norm
    emb = sd["embedding.word_embeddings.weight"]
    out["model.token_embed.weight"] = emb
    out["model.lm_head.weight"] = emb.clone()  # tied
    out["model.final_norm.weight"] = sd["decoder.final_layernorm.weight"]

    for n in range(n_layers):
        p = f"decoder.layers.{n}"
        q = f"model.layers.{n}"
        # RMSNorm
        out[f"{q}.attn_norm.weight"] = sd[f"{p}.input_layernorm.weight"]
        out[f"{q}.ffn_norm.weight"] = sd[f"{p}.pre_mlp_layernorm.weight"]
        # QK-Norm
        out[f"{q}.attn.q_norm.weight"] = sd[f"{p}.self_attention.q_layernorm.weight"]
        out[f"{q}.attn.k_norm.weight"] = sd[f"{p}.self_attention.k_layernorm.weight"]
        # 反融合 QKV + o_proj
        Wq, Wk, Wv = _defuse_qkv(
            sd[f"{p}.self_attention.linear_qkv.weight"], n_q, n_kv, head_dim
        )
        out[f"{q}.attn.W_q.weight"] = Wq
        out[f"{q}.attn.W_k.weight"] = Wk
        out[f"{q}.attn.W_v.weight"] = Wv
        out[f"{q}.attn.W_o.weight"] = sd[f"{p}.self_attention.linear_proj.weight"]
        # 反融合 GLU
        Wg, Wu = _defuse_glu(sd[f"{p}.mlp.linear_fc1.weight"], cfg["d_ff"])
        out[f"{q}.ffn.W_gate.weight"] = Wg
        out[f"{q}.ffn.W_up.weight"] = Wu
        out[f"{q}.ffn.W_down.weight"] = sd[f"{p}.mlp.linear_fc2.weight"]

    # 严格校验: 与 target key 集合一致（防止漏映射/命名错）
    missing = sorted(set(target_sd) - set(out) - {k for k in target_sd if "rope_cos" in k or "rope_sin" in k})
    unexpected = sorted(set(out) - set(target_sd))
    if missing or unexpected:
        raise RuntimeError(
            f"key 不一致 — missing: {missing[:5]} | unexpected: {unexpected[:5]}"
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Megatron checkpoint .pt")
    p.add_argument("--config", required=True, help="industrial configs/*.yaml")
    p.add_argument("--out", required=True, help="HF 输出目录")
    p.add_argument("--tokenizer-path", default="gleamlm/tokenizer/checkpoints/bbpe_12k",
                   help="BBPE 原生 tokenizer 目录（导出 HF tokenizer.json）")
    p.add_argument("--verify", action="store_true", help="与 Megatron 模型对比 logits")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    megatron_sd = ck.get("model_state_dict") or ck.get("model") or ck
    cfg = _core_config_from_yaml(args.config)
    print(f"convert {args.ckpt} -> {args.out} ({cfg['num_layers']}L, vocab {cfg['vocab_size']})")

    out = convert(megatron_sd, cfg)
    hf_model = GleamLMForCausalLM(gleamlm_config_from_core(cfg))
    res = hf_model.load_state_dict(out, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"load_state_dict 异常: {res}")
    hf_model.save_pretrained(args.out)
    print(f"Saved HF model: {args.out}")

    # tokenizer.json 写入同一目录 → HFBBPETokenizer.load(args.out) 可直接用（下游 SFT/RL/推理）
    from gleamlm.tokenizer.tokenizer import BBPETokenizer

    BBPETokenizer.load(args.tokenizer_path).export_to_hf_format(args.out)

    if args.verify:
        _verify(args, cfg, out)


def _verify(args, cfg: dict, hf_sd: dict):
    """同输入对比 Megatron GPTModel 与转换后 HF 模型的 logits（max abs diff）。"""
    import os as _os

    _os.environ.setdefault("MASTER_ADDR", "localhost")
    _os.environ.setdefault("MASTER_PORT", "23456")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.transformer import TransformerConfig

    from industrial.pretrain import build_model, build_position_ids
    from gleamlm.tokenizer.tokenizer import BBPETokenizer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    model_parallel_cuda_manual_seed(seed=42)
    mcfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m = mcfg["model"]
    tcfg = TransformerConfig(
        num_layers=m["num_layers"],
        hidden_size=m["hidden_size"],
        num_attention_heads=m["num_attention_heads"],
        num_query_groups=m.get("num_query_groups", m["num_attention_heads"]),
        ffn_hidden_size=m["ffn_hidden_size"],
        kv_channels=max(m["hidden_size"] // m["num_attention_heads"], 1),
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        fp16=False,
        bf16=True,
        layernorm_epsilon=1e-6,  # 对齐手写轨 RMSNorm 默认 eps(1e-6)
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        attention_softmax_in_fp32=False,
    )
    meg_model = build_model(tcfg, m["vocab_size"], m["max_position_embeddings"]).cuda().float()
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict") or raw.get("model") or raw
    sd = {k: v for k, v in sd.items() if v is not None}
    meg_model.load_state_dict(sd)
    meg_model.eval()

    tok_path = (
        mcfg.get("data", {}).get("tokenizer_path")
        or "gleamlm/tokenizer/checkpoints/bbpe_12k"
    )
    tok = BBPETokenizer.load(tok_path)
    # 用训练数据前缀不必要；直接用 tokenizer 编一段文本
    txt = "中国的首都是北京，上海是中国最大的城市之一。"
    ids = torch.tensor([tok.encode(txt, add_bos=False, add_eos=False)])[:, :64].cuda()
    s = ids.size(1)
    causal = torch.triu(
        torch.ones(s, s, dtype=torch.bool, device=ids.device), diagonal=1
    )
    attn = causal.unsqueeze(0).expand(ids.size(0), 1, s, s)
    with torch.no_grad():
        meg_logits = meg_model(
            input_ids=ids, position_ids=build_position_ids(ids), attention_mask=attn
        ).float()

    hf_model = GleamLMForCausalLM(gleamlm_config_from_core(cfg)).cuda().float()
    hf_model.load_state_dict(hf_sd, strict=True)
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=ids).logits.float()

    diff = (meg_logits - hf_logits).abs().max().item()
    print(f"[verify] max |meg - hf| logits diff = {diff:.6f} (希望 < 1e-2)")
    if diff < 1e-2:
        print("[verify] PASS")
    else:
        print("[verify] FAIL —— 权重映射/融合顺序可能不符，需核对 QKV/GLU 布局")
        raise SystemExit(1)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
