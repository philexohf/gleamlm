"""量化部署 — FP16 / INT8 / INT4（torchao）。

用法:
    python deploy/quantize.py --input model.pt --output model_fp16.pt --dtype fp16
    python deploy/quantize.py --input model.pt --output model_int4.pt --dtype int4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

try:
    from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
except ImportError:
    quantize_ = None
    int4_weight_only = None
    int8_weight_only = None


@torch.no_grad()
def quantize_ckpt(
    ckpt_path: str,
    output_path: str,
    dtype: str = "fp16",
    device: str = "cpu",
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt

    if dtype == "fp16":
        sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
    elif dtype in ("int8", "int4"):
        if quantize_ is None:
            raise ImportError("torchao required for int8/int4 quantization. pip install torchao")
        from gleamlm.utils.config import extract_checkpoint_config
        from hf.hf_config import gleamlm_config_from_core
        from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint

        cfg = gleamlm_config_from_core(extract_checkpoint_config(ckpt))
        model = GleamLMForCausalLM(cfg)
        load_from_checkpoint(model, ckpt, strict=True)
        model.to(device)
        q = int4_weight_only() if dtype == "int4" else int8_weight_only()
        quantize_(model, q)
        sd = model.state_dict()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # 透传输入 ckpt 的模型结构信息: 下游 extract_checkpoint_config 依赖
    # _config/args/config 重建模型，缺失会抛 ConfigValidationError
    meta = {k: ckpt[k] for k in ("_config", "args", "config") if k in ckpt}
    ckpt_out = {"model_state_dict": sd, "dtype": dtype, **meta}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(ckpt_out, output_path)
    print(f"Quantized ({dtype}) saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM quantization")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "int8", "int4"])
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quantize_ckpt(args.input, args.output, args.dtype, args.device)
