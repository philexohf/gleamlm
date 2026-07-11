"""GleamLM-Pro 126M 量化导出。FP32 → FP16，体积减半，推理精度基本无损。

用法：
    python gleamlm-pro/quantize.py
    python gleamlm-pro/quantize.py --input checkpoints/best_model.pt --output checkpoints/model_fp16.pt

流程：
    1. argparse 获取输入输出路径
    2. 委托 gleamlm.deploy.quantize_to_fp16 执行量化
"""

import argparse
import os
import sys

from gleamlm.deploy.quantize import quantize_to_fp16

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="GleamLM-Pro 126M FP16 量化导出")
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(DEFAULT_CHECKPOINT_DIR, "best_model.pt"),
        help="输入模型路径",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(DEFAULT_CHECKPOINT_DIR, "model_fp16.pt"),
        help="输出 FP16 模型路径",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: 输入模型不存在: {args.input}")
        sys.exit(1)

    quantize_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
