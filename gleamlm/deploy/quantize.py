"""GleamLM FP16 quantization. Variant-agnostic — reads architecture from checkpoint metadata.

Usage:
    python gleamlm/deploy/quantize.py --input path/to/model.pt --output path/to/fp16.pt
    python -m gleamlm.deploy.quantize --input gleamlm-lite/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

from gleamlm.models.model import GleamLMModel

ARCH_KEYS = [
    "vocab_size",
    "d_model",
    "num_layers",
    "num_heads",
    "num_kv_heads",
    "d_ff",
    "max_seq_len",
    "pad_token_id",
    "use_flash_attn",
]


def extract_config(checkpoint: dict) -> dict[str, Any]:
    if "args" in checkpoint:
        args = checkpoint["args"]
        return {
            "vocab_size": getattr(args, "vocab_size", 12002),
            "d_model": getattr(args, "d_model", 768),
            "num_layers": getattr(args, "num_layers", 12),
            "num_heads": getattr(args, "num_heads", 12),
            "num_kv_heads": getattr(args, "num_kv_heads", 6),
            "d_ff": getattr(args, "d_ff", 2048),
            "dropout": 0.0,
            "max_seq_len": getattr(args, "max_seq_len", 2048),
            "pad_token_id": getattr(args, "pad_token_id", 0),
            "tie_weights": False,
            "use_flash_attn": getattr(args, "use_flash_attn", False),
        }
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        return {
            "vocab_size": cfg.get("vocab_size", 12002),
            "d_model": cfg.get("d_model", 768),
            "num_layers": cfg.get("num_layers", 12),
            "num_heads": cfg.get("num_heads", 12),
            "num_kv_heads": cfg.get("num_kv_heads", 6),
            "d_ff": cfg.get("d_ff", 2048),
            "dropout": 0.0,
            "max_seq_len": cfg.get("max_seq_len", 2048),
            "pad_token_id": cfg.get("pad_token_id", 0),
            "tie_weights": False,
            "use_flash_attn": cfg.get("use_flash_attn", False),
        }
    raise ValueError(
        "Checkpoint 缺少模型结构信息。请确保 checkpoint 包含 'args' 或 'config' 字段。"
    )


def quantize_to_fp16(input_path: str, output_path: str) -> None:
    """Convert FP32 model to FP16 and save.

    Architecture params are extracted from checkpoint metadata (args/config).
    Works with both Nano and Lite checkpoints.
    """
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    config = extract_config(checkpoint)

    model = GleamLMModel(**config)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint 缺少模型权重。"
            "请确认 checkpoint 包含 'model_state_dict' 字段，"
            "且不是在仅有 config 元数据而無權重的情況下保存的。"
        )
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys in checkpoint")

    model = model.half()

    fp32_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    fp16_size = sum(p.numel() for p in model.parameters()) * 2 / (1024**2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "dtype": "float16",
        },
        output_path,
    )

    print("\n模型量化完成:")
    print(f"  FP32 大小: {fp32_size:.1f} MB")
    print(f"  FP16 大小: {fp16_size:.1f} MB")
    print(f"  压缩比: {fp32_size / fp16_size:.1f}x")
    print(f"  保存至: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GleamLM FP16 量化导出")
    parser.add_argument(
        "--input", type=str, default="checkpoints/best_model.pt", help="输入模型路径"
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/model_fp16.pt", help="输出 FP16 模型路径"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: 输入模型不存在: {args.input}")
        sys.exit(1)

    quantize_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
