"""烁珑GleamLM 量化导出。FP32 → FP16 减半体积"""

import torch
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gleamlm_model import GleamLMModel


def quantize_to_fp16(input_path, output_path):
    """
    将 FP32 模型转换为 FP16 并保存
    """
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # 从 checkpoint 中获取模型参数
    if 'args' in checkpoint:
        args = checkpoint['args']
        config = {
            'vocab_size': getattr(args, 'vocab_size', 12001),
            'd_model': getattr(args, 'd_model', 512),
            'num_layers': getattr(args, 'num_layers', 12),
            'num_heads': getattr(args, 'num_heads', 8),
            'num_kv_heads': getattr(args, 'num_kv_heads', 4),
            'd_ff': getattr(args, 'd_ff', 1365),
            'dropout': 0.0,  # 推理时不需要 dropout
            'max_seq_len': getattr(args, 'max_seq_len', 1024),
            'pad_token_id': 0,
            'tie_weights': False  # 导出时不绑定，保存完整权重
        }
    else:
        config = {
            'vocab_size': 12001, 'd_model': 512, 'num_layers': 12,
            'num_heads': 8, 'num_kv_heads': 4, 'd_ff': 1365,
            'dropout': 0.0, 'max_seq_len': 1024, 'pad_token_id': 0,
            'tie_weights': False
        }

    model = GleamLMModel(**config)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model = model.half()

    fp32_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    fp16_size = sum(p.numel() for p in model.parameters()) * 2 / (1024 ** 2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dtype': 'float16'
    }, output_path)

    print(f"\n模型量化完成:")
    print(f"  FP32 大小: {fp32_size:.1f} MB")
    print(f"  FP16 大小: {fp16_size:.1f} MB")
    print(f"  压缩比: {fp32_size / fp16_size:.1f}x")
    print(f"  保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='烁珑GleamLM FP16 量化导出')
    parser.add_argument('--input', type=str, default='checkpoints/best_model.pt',
                        help='输入模型路径')
    parser.add_argument('--output', type=str, default='checkpoints/model_fp16.pt',
                        help='输出 FP16 模型路径')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: 输入模型不存在: {args.input}")
        sys.exit(1)

    quantize_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
