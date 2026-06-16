"""Xfind-Mini 推理脚本，支持交互式生成和多种采样策略"""

import torch
import argparse
import sys
import os

# Windows 终端编码修复
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import load_model_for_inference
from tokenizer.xfind_tokenizer import XfindTokenizer
from inference.streamer import TextStreamer


def load_model(model_path, device='cuda'):
    """加载模型和分词器"""
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 从 checkpoint 提取分词器路径
    if 'args' in checkpoint:
        args = checkpoint['args']
        tokenizer_path = getattr(args, 'tokenizer_path', './tokenizer/checkpoints/bpe_32k')
    elif 'config' in checkpoint:
        tokenizer_path = './tokenizer/checkpoints/bpe_32k'
    else:
        tokenizer_path = './tokenizer/checkpoints/bpe_32k'

    # 使用共享函数加载模型（传入已加载的 checkpoint 避免重复磁盘读取）
    model, config = load_model_for_inference(model_path, device, checkpoint=checkpoint)

    # 加载分词器
    tokenizer = XfindTokenizer(tokenizer_path)

    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.2f}M params, device: {device}")
    print(f"Tokenizer vocab: {len(tokenizer)}")

    return model, tokenizer, config


def generate(model, tokenizer, prompt, max_new_tokens=256,
             temperature=1.0, top_k=50, top_p=0.9, device='cuda'):
    """生成文本并实时打印"""
    streamer = TextStreamer(tokenizer)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print("Generated: ", end='', flush=True)

    full_text = prompt
    for chunk in streamer.generate_text(
        model, prompt, max_new_tokens,
        temperature, top_k, top_p
    ):
        # 增量输出
        new_text = chunk[len(full_text) - len(prompt):] if len(full_text) > len(prompt) else chunk
        try:
            print(new_text, end='', flush=True)
        except UnicodeEncodeError:
            print(new_text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'), end='', flush=True)
        full_text = prompt + chunk

    print("\n")
    return full_text


def interactive(model, tokenizer, max_new_tokens=256,
                temperature=1.0, top_k=50, top_p=0.9, device='cuda'):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("Xfind-Mini 交互式文本生成")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if not prompt:
            continue

        generate(model, tokenizer, prompt, max_new_tokens,
                 temperature, top_k, top_p, device)


def main():
    parser = argparse.ArgumentParser(description='Xfind-Mini 推理')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='模型路径')
    parser.add_argument('--prompt', type=str, default=None,
                        help='提示文本（不提供则进入交互模式）')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='温度参数')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-K 采样参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-P 采样参数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: 模型文件不存在: {args.model}")
        print("请先训练模型或指定正确的模型路径")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else 'cpu'

    model, tokenizer, config = load_model(args.model, device)

    if args.prompt:
        generate(model, tokenizer, args.prompt, args.max_new_tokens,
                 args.temperature, args.top_k, args.top_p, device)
    else:
        interactive(model, tokenizer, args.max_new_tokens,
                    args.temperature, args.top_k, args.top_p, device)


if __name__ == "__main__":
    main()
