"""Expand single-turn SFT data into multi-turn conversations.

Reads single-turn JSONL (instruction/output) and uses the DeepSeek API to generate
1-2 follow-up question-answer pairs, producing multi-turn `messages` format.

Usage:
    $env:DEEPSEEK_API_KEY = "sk-xxxx"
    python data_tools/gen_multiturn_sft.py --input data/sft_data.jsonl --output data/multiturn_sft.jsonl

The output format is compatible with the multi-turn SFTDataset:
    {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]}
"""

import argparse
import json
import os
import random
import re
import sys
import time

_sys_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

from data_tools.shared.api_client import DEFAULT_MODEL, chat_completion, get_client

SYSTEM_PROMPT = (
    "你是一个对话数据构造助手。给你一段单轮问答，请生成 1-2 个后续的追问答复，"
    "形成一个自然的多轮对话。\n\n"
    "要求：\n"
    "- 追问应与原话题深度相关（如要求举例、补充细节、追问原因、对比分析等）\n"
    "- 每个追问的问题和回答都应该是自然的中文对话语气\n"
    "- 回答长度适中（50-200字），与追问匹配\n"
    "- 输出严格 JSON 格式，不含任何额外文字"
)

OUTPUT_SCHEMA = """
{
  "follow_ups": [
    {"user": "追问问题1", "assistant": "对应回答1"},
    {"user": "追问问题2", "assistant": "对应回答2"}
  ]
}
"""


def build_prompt(instruction: str, output: str) -> str:
    return (
        f"原始对话：\n用户：{instruction}\n助手：{output}\n\n"
        f"请基于以上话题，生成 1-2 个自然的后续追问和回答。\n"
        f"输出必须是合法的 JSON，格式如下：\n{OUTPUT_SCHEMA}"
    )


def parse_response(raw: str) -> list[dict[str, str]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        data = json.loads(raw)
        fus = data.get("follow_ups", data)
        result: list[dict[str, str]] = []
        for item in fus:
            if isinstance(item, dict) and "user" in item and "assistant" in item:
                result.append({"user": item["user"], "assistant": item["assistant"]})
        return result
    except (json.JSONDecodeError, TypeError):
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Expand single-turn SFT to multi-turn via DeepSeek API"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Single-turn SFT JSONL (instruction/output)"
    )
    parser.add_argument("--output", type=str, required=True, help="Output multi-turn JSONL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max_turns", type=int, default=2, help="Max follow-up turns per sample (1-2)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Max samples to process (0 = all)")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable or pass --api_key", file=sys.stderr)
        sys.exit(1)

    samples: list[dict[str, str]] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "instruction" in item and "output" in item:
                    samples.append(item)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(samples)} single-turn samples from {args.input}")

    if args.limit and args.limit < len(samples):
        samples = random.sample(samples, args.limit)
        print(f"Sampled {len(samples)} for processing")

    client = get_client(api_key)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, s in enumerate(samples):
            instruction = s["instruction"]
            output_str = s["output"]
            print(f"[{i + 1}/{len(samples)}] {instruction[:40]}...", end=" ", flush=True)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(instruction, output_str)},
            ]
            raw = chat_completion(
                client,
                args.model,
                messages,
                temperature=0.7,
                max_tokens=1024,
                extra_kwargs={"response_format": {"type": "json_object"}},
            )
            follow_ups = parse_response(raw) if raw else []
            if not follow_ups:
                print("SKIP (no valid follow-ups)")
                continue

            max_t = min(args.max_turns, len(follow_ups))
            turns = follow_ups[:max_t]

            messages_out: list[dict[str, str]] = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output_str},
            ]
            for t in turns:
                messages_out.append({"role": "user", "content": t["user"]})
                messages_out.append({"role": "assistant", "content": t["assistant"]})

            item = {"messages": messages_out}
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            print(f"OK (+{len(turns)} turns)")

            time.sleep(0.3)

    print(f"\nDone. {written}/{len(samples)} multi-turn conversations written to {args.output}")


if __name__ == "__main__":
    main()
