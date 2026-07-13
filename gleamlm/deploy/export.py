"""GleamLM safetensors export. Converts PyTorch checkpoints to safetensors format.

Usage:
    python -m gleamlm.deploy.export --input path/to/model.pt --output ./export/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from gleamlm.deploy.quantize import extract_config
from gleamlm.models.model import GleamLMModel


def export_safetensors(checkpoint_path: str, output_dir: str) -> None:
    try:
        import safetensors.torch
    except ImportError:
        print("Missing safetensors. Install: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = extract_config(ckpt)
    config["dropout"] = 0.0

    model = GleamLMModel(**config)
    sd = ckpt["model_state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys in checkpoint")

    if ckpt.get("dtype") == "float16":
        model = model.half()

    os.makedirs(output_dir, exist_ok=True)
    safetensors.torch.save_file(
        model.state_dict(), os.path.join(output_dir, "model.safetensors")
    )

    config["tokenizer_path"] = "tokenizer/"
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    total_params = sum(p.numel() for p in model.parameters())
    file_size = os.path.getsize(os.path.join(output_dir, "model.safetensors"))
    print(f"Exported {total_params / 1e6:.1f}M params -> {output_dir}")
    print(f"  model.safetensors: {file_size / 1024 / 1024:.1f} MB")
    print(f"  config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GleamLM checkpoint to safetensors")
    parser.add_argument("--input", type=str, required=True, help="Checkpoint .pt file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    export_safetensors(args.input, args.output)


if __name__ == "__main__":
    main()
