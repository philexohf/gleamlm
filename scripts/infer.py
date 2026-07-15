"""GleamLM 统一推理脚本。支持交互式生成和多种采样策略。

用法:
    python scripts/infer.py --model checkpoints/nano/best_model.pt
    python scripts/infer.py --model checkpoints/lite/sft/sft_best.pt --sft
"""

import argparse
import contextlib
import os
import sys

import torch

if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

from gleamlm import load_model_for_inference
from gleamlm.inference.chatml import format_chatml
from gleamlm.inference.streamer import TextStreamer
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def load_model(model_path, device="cuda"):
    print(f"Loading model: {model_path}")

    model, config = load_model_for_inference(model_path, device)

    tokenizer_path = config.get("tokenizer_path", DEFAULT_TOKENIZER_PATH)
    if not os.path.exists(tokenizer_path):
        if tokenizer_path.startswith("./"):
            alt = os.path.join("..", tokenizer_path[2:])
            if os.path.exists(alt):
                tokenizer_path = alt
        if not os.path.exists(tokenizer_path):
            tokenizer_path = DEFAULT_TOKENIZER_PATH

    tokenizer = BBPETokenizer.load(tokenizer_path)

    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.2f}M params, device: {device}")
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    return model, tokenizer, config


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    sft_mode=False,
    stop_token=None,
):
    streamer = TextStreamer(tokenizer)

    if sft_mode:
        prompt = format_chatml(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        if stop_token is None:
            stop_token = "<|im_end|>"

    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}")
    print("Generated: ", end="", flush=True)

    generated_text = ""
    for chunk in streamer.generate_text(
        model,
        prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        stop_on_endoftext=sft_mode,
    ):
        generated_text += chunk
        if sft_mode and stop_token and stop_token in chunk:
            chunk = chunk.split(stop_token)[0] + stop_token
            try:
                print(chunk, end="", flush=True)
            except UnicodeEncodeError:
                print(
                    chunk.encode("utf-8", errors="replace").decode("utf-8", errors="replace"),
                    end="",
                    flush=True,
                )
            break
        try:
            print(chunk, end="", flush=True)
        except UnicodeEncodeError:
            print(
                chunk.encode("utf-8", errors="replace").decode("utf-8", errors="replace"),
                end="",
                flush=True,
            )

    full_text = prompt + generated_text
    print("\n")
    return full_text


def interactive(
    model,
    tokenizer,
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    sft_mode=False,
):
    print("\n" + "=" * 60)
    print("GleamLM 交互式文本生成")
    if sft_mode:
        print("SFT 对话模式 (ChatML 格式)")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not prompt:
            continue

        generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            sft_mode=sft_mode,
        )


def main():
    parser = argparse.ArgumentParser(description="GleamLM 推理")
    parser.add_argument("--model", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--prompt", type=str, default=None, help="提示文本（不提供则进入交互模式）")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K 采样参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P 采样参数")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="重复惩罚 (>1.0 抑制重复)"
    )
    parser.add_argument("--sft", action="store_true", help="SFT 对话模式")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: 模型文件不存在: {args.model}")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_model(args.model, device)

    if args.prompt:
        generate(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            sft_mode=args.sft,
        )
    else:
        interactive(
            model,
            tokenizer,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            sft_mode=args.sft,
        )


if __name__ == "__main__":
    main()
