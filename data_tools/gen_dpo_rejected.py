"""生成 DPO rejected 数据。

用预训练基座模型或 SFT 模型对 SFT 数据生成"差的"回答，作为 DPO rejected。

用法:
  # 单轮：用基座模型生成 rejected
  python data_tools/gen_dpo_rejected.py --format single \
      --sft_data data/sft_data.jsonl --model_path checkpoints/best_model.pt --output data/dpo_data.jsonl

  # 多轮：用 SFT 模型生成 rejected
  python data_tools/gen_dpo_rejected.py --format multi \
      --input data/multiturn_sft.jsonl --model checkpoints/sft/sft_best.pt --output data/dpo_multiturn.jsonl
"""

import argparse
import json
import os
import sys

import torch

from gleamlm import load_model_for_inference
from gleamlm.inference.generator import generate_tokens
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def generate_rejected_single(
    model, tokenizer, instruction: str, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9
) -> str:
    """单轮：用模型生成一次回答作为 rejected。"""
    device = next(model.parameters()).device
    prompt_text = f"<|im_start|><|user|>\n{instruction}<|im_end|>\n<|im_start|><|assistant|>\n"
    prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)

    generated: list[int] = []
    for token_id in generate_tokens(
        model,
        prompt_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.15,
        penalty_window=50,
        stop_ids={tokenizer.eos_id, tokenizer.pad_id},
    ):
        generated.append(token_id)

    response = tokenizer.decode(generated, skip_special=True)
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]
    return response


def last_assistant_turn(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]] | None, str | None]:
    """从 messages 中剥离最后一轮 assistant 回复，返回 (context, last_content)。"""
    if not messages or messages[-1]["role"] != "assistant":
        return None, None
    context = messages[:-1]
    target = messages[-1]["content"]
    return context, target


def build_prompt(context: list[dict[str, str]], tokenizer) -> list[int]:
    """多轮对话上下文 → token IDs。"""
    parts: list[str] = []
    for msg in context:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|><|{role}|>\n{content}<|im_end|>\n")
    parts.append("<|im_start|><|assistant|>\n")
    return tokenizer.encode("".join(parts), add_bos=False, add_eos=False)


def generate_rejected_multi(
    model,
    tokenizer,
    context: list[dict[str, str]],
    max_new_tokens=256,
    temperature=0.95,
    top_k=50,
    top_p=0.9,
) -> str:
    """多轮：基于完整对话上下文重新生成最后一轮 assistant 回答。"""
    device = next(model.parameters()).device
    prompt_ids = build_prompt(context, tokenizer)
    im_end_id = tokenizer.special_tokens.get("<|im_end|>")
    stop_ids: set[int] = {tokenizer.eos_id, tokenizer.pad_id}
    if im_end_id is not None:
        stop_ids.add(im_end_id)

    generated: list[int] = []
    for token_id in generate_tokens(
        model,
        prompt_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.15,
        penalty_window=50,
        stop_ids=stop_ids,
    ):
        generated.append(token_id)

    return tokenizer.decode(generated, skip_special=True)


def main():
    parser = argparse.ArgumentParser(description="Generate DPO rejected data")
    parser.add_argument(
        "--format", type=str, choices=["single", "multi"], default="single", help="数据格式"
    )
    # 路径
    parser.add_argument(
        "--sft_data", type=str, default=None, help="单轮 SFT JSONL (instruction/output)"
    )
    parser.add_argument("--input", type=str, default=None, help="多轮 SFT JSONL (messages 格式)")
    parser.add_argument(
        "--model_path",
        "--model",
        dest="model_path",
        type=str,
        default=None,
        help="模型 checkpoint 路径",
    )
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=str, default="data/dpo_data.jsonl", help="输出文件")
    # 生成参数
    parser.add_argument("--limit", type=int, default=0, help="最大样本数 (0=all)")
    parser.add_argument(
        "--temperature", type=float, default=0.95, help="温度 (单轮 default=0.8, 多轮 default=0.95)"
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    fmt = args.format
    if fmt == "single":
        data_path = args.sft_data
        if not data_path:
            print("Error: --sft_data required for single format", file=sys.stderr)
            sys.exit(1)
    else:
        data_path = args.input
        if not data_path:
            print("Error: --input required for multi format", file=sys.stderr)
            sys.exit(1)

    model_path = args.model_path
    if not model_path:
        print("Error: --model_path required", file=sys.stderr)
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path}")
    model, config = load_model_for_inference(model_path, device)
    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.2f}M params")

    # 加载数据
    samples: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if (
                    fmt == "single"
                    and "instruction" in item
                    and "output" in item
                    or fmt == "multi"
                    and "messages" in item
                ):
                    samples.append(item)
            except json.JSONDecodeError:
                continue
    if args.limit:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples from {data_path}")

    temp = args.temperature if fmt == "multi" else min(args.temperature, 0.95)

    dpo_data: list[dict] = []
    for i, s in enumerate(samples):
        if fmt == "single":
            instruction = s["instruction"]
            chosen = s["output"]
            rejected = generate_rejected_single(
                model,
                tokenizer,
                instruction,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            dpo_data.append({"instruction": instruction, "chosen": chosen, "rejected": rejected})
        else:
            messages: list[dict] = s["messages"]
            context, target = last_assistant_turn(messages)
            if context is None or target is None:
                continue
            rejected = generate_rejected_multi(
                model,
                tokenizer,
                context,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            dpo_data.append(
                {
                    "messages": context + [{"role": "assistant", "content": target}],
                    "chosen": target,
                    "rejected": rejected,
                }
            )

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(samples)}]")
            with open(args.output + ".partial", "w", encoding="utf-8") as pf:
                for item in dpo_data:
                    pf.write(json.dumps(item, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done -> {args.output} ({len(dpo_data)} pairs)")


if __name__ == "__main__":
    main()
