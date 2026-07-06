"""GleamLM 推理脚本，支持交互式生成和多种采样策略"""

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

from gleamlm import load_model_for_inference
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, 'checkpoints')
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.inference.streamer import TextStreamer


def load_model(model_path, device='cuda'):
    """加载模型和分词器"""
    print(f"Loading model: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    # 从 checkpoint 提取分词器路径
    if 'args' in checkpoint:
        args = checkpoint['args']
        tokenizer_path = getattr(args, 'tokenizer_path', DEFAULT_TOKENIZER_PATH)
    elif 'config' in checkpoint:
        tokenizer_path = DEFAULT_TOKENIZER_PATH
    else:
        tokenizer_path = DEFAULT_TOKENIZER_PATH

    # 兼容旧 checkpoint 中存储的旧路径
    if not os.path.exists(tokenizer_path):
        if tokenizer_path.startswith('./'):
            alt = '../' + tokenizer_path[2:]
            if os.path.exists(alt):
                tokenizer_path = alt
        if not os.path.exists(tokenizer_path):
            tokenizer_path = DEFAULT_TOKENIZER_PATH

    # 使用共享函数加载模型（传入已加载的 checkpoint 避免重复磁盘读取）
    model, config = load_model_for_inference(model_path, device, checkpoint=checkpoint)

    tokenizer = BBPETokenizer.load(tokenizer_path)

    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.2f}M params, device: {device}")
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    return model, tokenizer, config


def generate(model, tokenizer, prompt, max_new_tokens=256,
             temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.1,
             sft_mode=False, stop_token=None):
    """生成文本并实时打印"""
    streamer = TextStreamer(tokenizer)

    # SFT 模式：ChatML 包装
    if sft_mode:
        prompt = f"<|im_start|><|user|>\n{prompt}<|im_end|>\n<|im_start|><|assistant|>\n"
        if stop_token is None:
            stop_token = "<|im_end|>"

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print("Generated: ", end='', flush=True)

    prev_len = 0
    last_chunk = prompt
    for chunk in streamer.generate_text(
        model, prompt, max_new_tokens,
        temperature, top_k, top_p, repetition_penalty
    ):
        new_text = chunk[prev_len:]
        prev_len = len(chunk)
        last_chunk = chunk
        # SFT 模式：遇到 stop_token 截断
        if sft_mode and stop_token and stop_token in new_text:
            new_text = new_text.split(stop_token)[0] + stop_token
            try:
                print(new_text, end='', flush=True)
            except UnicodeEncodeError:
                print(new_text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'), end='', flush=True)
            break
        try:
            print(new_text, end='', flush=True)
        except UnicodeEncodeError:
            print(new_text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'), end='', flush=True)

    full_text = prompt + last_chunk if last_chunk != prompt else prompt
    print("\n")
    return full_text


def interactive(model, tokenizer, max_new_tokens=256,
                temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.1,
                device='cuda', sft_mode=False):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("GleamLM 交互式文本生成")
    if sft_mode:
        print("SFT 对话模式（ChatML 格式）")
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
                 temperature, top_k, top_p, repetition_penalty,
                 sft_mode=sft_mode)


def main():
    parser = argparse.ArgumentParser(description='GleamLM 推理')
    parser.add_argument('--model', type=str, default=f'{DEFAULT_CHECKPOINT_DIR}/best_model.pt',
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
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='重复惩罚（>1.0 抑制重复，如 1.15）')
    parser.add_argument('--sft', action='store_true',
                        help='SFT 对话模式（ChatML 包装 prompt，遇 <|im_end|> 截断）')
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
                 args.temperature, args.top_k, args.top_p, args.repetition_penalty,
                 sft_mode=args.sft)
    else:
        interactive(model, tokenizer, args.max_new_tokens,
                    args.temperature, args.top_k, args.top_p, args.repetition_penalty,
                    device, sft_mode=args.sft)


if __name__ == "__main__":
    main()
