"""Unified GleamLM inference CLI. Merged from nano/infer.py and lite/infer.py.

Usage:
    python -m gleamlm.inference.cli --model path/to/model.pt
    python -m gleamlm.inference.cli --model checkpoints/best_model.pt --prompt "你好"
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch

if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

from gleamlm import load_model_for_inference
from gleamlm.inference.streamer import TextStreamer
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def load_model(
    model_path: str, device: str = "cuda"
) -> tuple[torch.nn.Module, BBPETokenizer, dict]:
    print(f"Loading model: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if "args" in checkpoint:
        args = checkpoint["args"]
        tokenizer_path = getattr(args, "tokenizer_path", DEFAULT_TOKENIZER_PATH)
    elif "config" in checkpoint:
        tokenizer_path = DEFAULT_TOKENIZER_PATH
    else:
        tokenizer_path = DEFAULT_TOKENIZER_PATH

    if not os.path.exists(tokenizer_path):
        if tokenizer_path.startswith("./"):
            alt = "../" + tokenizer_path[2:]
            if os.path.exists(alt):
                tokenizer_path = alt
        if not os.path.exists(tokenizer_path):
            tokenizer_path = DEFAULT_TOKENIZER_PATH

    model, config = load_model_for_inference(model_path, device, checkpoint=checkpoint)
    tokenizer = BBPETokenizer.load(tokenizer_path)

    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.2f}M params, device: {device}")
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    return model, tokenizer, config


def generate(
    model: torch.nn.Module,
    tokenizer: BBPETokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    sft_mode: bool = False,
    stop_token: str | None = None,
) -> str:
    streamer = TextStreamer(tokenizer)

    if sft_mode:
        prompt = f"<|im_start|><|user|>\n{prompt}<|im_end|>\n<|im_start|><|assistant|>\n"
        if stop_token is None:
            stop_token = "<|im_end|>"

    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}")
    print("Generated: ", end="", flush=True)

    prev_len = 0
    last_chunk = prompt
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
        new_text = chunk[prev_len:]
        prev_len = len(chunk)
        last_chunk = chunk
        if sft_mode and stop_token and stop_token in new_text:
            new_text = new_text.split(stop_token)[0] + stop_token
            _safe_print(new_text)
            break
        _safe_print(new_text)

    full_text = prompt + (last_chunk if last_chunk != prompt else "")
    print("\n")
    return full_text


def _safe_print(text: str) -> None:
    """Print with Unicode error fallback."""
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:
        print(
            text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"),
            end="",
            flush=True,
        )


def interactive(
    model: torch.nn.Module,
    tokenizer: BBPETokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cuda",
    sft_mode: bool = False,
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="GleamLM Unified Inference CLI")
    parser.add_argument("--model", type=str, required=True, help="模型 checkpoint 路径 (.pt 文件)")
    parser.add_argument("--prompt", type=str, default=None, help="提示文本（不提供则进入交互模式）")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.15, help="重复惩罚（>1.0 抑制重复）"
    )
    parser.add_argument(
        "--sft", action="store_true", help="SFT 对话模式（ChatML 包装 prompt，遇 <|im_end|> 截断）"
    )
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    model_path = args.model

    if not os.path.exists(model_path):
        print(f"Error: 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径 (--model <path>)")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_model(model_path, device)

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
            device,
            sft_mode=args.sft,
        )


if __name__ == "__main__":
    main()
