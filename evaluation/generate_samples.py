"""使用训练好的模型生成文本样例，用于人工评估"""

import torch
import os
import sys
import argparse

# Windows 终端编码修复
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model_for_inference
from tokenizer.xfind_tokenizer import XfindTokenizer
from inference.streamer import TextStreamer


def generate_samples(model, tokenizer, prompts, max_new_tokens=128,
                     temperature=0.8, top_k=50, top_p=0.9,
                     repetition_penalty=1.0, device='cuda'):
    """对多个 prompt 生成文本样例"""
    streamer = TextStreamer(tokenizer)
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt}")
        print("-" * 40)

        generated = [prompt]
        for chunk in streamer.generate_text(
            model, prompt, max_new_tokens, temperature, top_k, top_p,
            repetition_penalty
        ):
            generated.append(chunk)

        result = ''.join(generated)
        try:
            print(f"Generated: {result[len(prompt):]}")
        except UnicodeEncodeError:
            print(f"Generated: {result[len(prompt):].encode('utf-8', errors='replace').decode('utf-8', errors='replace')}")
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='生成文本样例评估')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='模型路径')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='温度参数')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='重复惩罚（>1 惩罚已出现 token）')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # 加载模型（配置从 checkpoint 自动读取）
    model, config = load_model_for_inference(args.model, device)

    # 分词器
    tokenizer_path = './tokenizer/checkpoints/bpe_32k'
    tokenizer = XfindTokenizer(tokenizer_path)

    # 预设评估 prompts
    prompts = [
        "人工智能是",
        "深度学习通过",
        "自然语言处理",
        "计算机视觉技术",
        "大语言模型",
        "机器学习的核心是",
        "Transformer架构",
    ]

    generate_samples(
        model, tokenizer, prompts,
        args.max_new_tokens, args.temperature, args.top_k, args.top_p,
        args.repetition_penalty, device
    )


if __name__ == '__main__':
    main()
