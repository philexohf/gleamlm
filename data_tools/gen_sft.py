"""生成 SFT 训练数据。支持硬编码模板和 DeepSeek API 蒸馏两种模式。

用法:
    # 硬编码模板模式（生成 10000 条，无需 API）
    python data_tools/gen_sft.py --mode hardcoded --output data/sft_data.jsonl

    # API 蒸馏模式
    set DEEPSEEK_API_KEY=sk-xxxx
    python data_tools/gen_sft.py --mode api --output data/sft_api.jsonl
    python data_tools/gen_sft.py --mode api --output data/sft_api_new.jsonl --variants_per_seed 2
"""

import argparse
import json
import os
import random
import re
import sys
import time

from data_tools._api_client import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completion, get_client
from data_tools._sft_seeds import SEEDS
from data_tools._sft_templates import (
    A_DATA,
    B_DATA,
    C_DATA,
    EXTRA_PREFIXES,
    INSERT_WORDS,
    PREFIXES,
)

random.seed(42)

VARIANT_PROMPT = """请为以下问题生成 {n} 个意思相近但表达不同的变体。

要求：
- 保持原问题的核心含义不变
- 用不同的措辞、句式重新表达
- 不要过于口语化
- 每个变体一行，不要编号

原问题：{seed}

{example_text}"""

VARIANT_EXAMPLE = """示例：
原问题：介绍一下你自己。

输出：
请做一个自我介绍。
你是谁？可以介绍一下吗？
能跟我聊聊你自己吗？
简单说说你的情况吧。"""

ANSWER_SYSTEM = (
    "你是GleamLM，一个面向教育和研究的轻量级开源对话模型（约40M参数），"
    "由个人开发者基于PyTorch从零实现，参考了LLaMA3和Qwen3架构。"
    "请用中文回答问题。"
    "回答要准确、简洁、有条理，长度适中。"
    "注意：不要提及任何具体公司、产品名称或模型来源，"
    "不要说你来自哪个公司或由谁开发。像一个通用的AI助手一样回答。"
)

API_VARIANT_SYSTEM = "你是一个中文语言专家，擅长改写问题。"


# ──── Hardcoded mode ──────────────────────────────────────────────


def _gen_variants(instruction, output, cat):
    variants = []
    seen = {instruction}
    max_attempts = 120
    while len(variants) < 25 and max_attempts > 0:
        tpl = random.choice(PREFIXES[cat])
        q = random.choice([instruction, output])
        words = random.choice(INSERT_WORDS[cat])
        if words:
            words += "，"
        p = tpl.format(q=q, words=words)
        if p not in seen:
            variants.append(p)
            seen.add(p)
        max_attempts -= 1
    for tpl in EXTRA_PREFIXES:
        p = tpl.format(q=instruction)
        if len(variants) < 28 and p not in seen:
            variants.append(p)
            seen.add(p)
    return variants


def _build_category(bases, cat, target):
    results = []
    for instruction, output in bases:
        results.append({"instruction": instruction, "output": output})
        variants = _gen_variants(instruction, output, cat)
        for v in variants:
            results.append({"instruction": v, "output": output})
    if len(results) < target:
        expand_pool = results * (target // len(results) + 1)
        results = expand_pool[:target]
    random.shuffle(results)
    return results[:target]


def generate_hardcoded(output_path: str, target_count: int = 10000) -> None:
    print(f"Hardcoded mode: target={target_count}")
    a_target = int(target_count * 0.40)
    b_target = int(target_count * 0.30)
    c_target = target_count - a_target - b_target
    print(f"  A (通用问答): {a_target}, B (知识回答): {b_target}, C (创作闲聊): {c_target}")
    a = _build_category(A_DATA, "A", a_target)
    b = _build_category(B_DATA, "B", b_target)
    c = _build_category(C_DATA, "C", c_target)
    all_data = a + b + c
    random.shuffle(all_data)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done: {len(all_data)} samples -> {output_path}")


# ──── API mode ────────────────────────────────────────────────────


def _api_gen_variants(client, model, seed, n_variants=4):
    prompt = VARIANT_PROMPT.format(n=n_variants, seed=seed, example_text=VARIANT_EXAMPLE)
    messages = [
        {"role": "system", "content": API_VARIANT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    result = chat_completion(client, model, messages, temperature=0.8, max_tokens=512)
    if result is None:
        return []
    variants = [v.strip() for v in result.split("\n") if v.strip()]
    unique = []
    for v in variants:
        v = re.sub(r"^[\d\s.\-、)）*#]+\s*", "", v).strip()
        if v and v != seed and len(v) >= 5 and v not in unique:
            unique.append(v)
    return unique[:n_variants]


def _api_gen_answer(client, model, instruction):
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": instruction},
    ]
    return chat_completion(client, model, messages, temperature=0.7, max_tokens=1024, max_retries=2)


def generate_api(
    output_path,
    api_key,
    base_url,
    model,
    variants_per_seed,
    skip_variants,
    delay,
    dry_run,
):
    client = get_client(api_key, base_url)
    print(f"API: {base_url}, model: {model}")

    seeds = SEEDS[:dry_run] if dry_run > 0 else SEEDS
    print(f"种子问题数: {len(seeds)}")

    all_instructions = []
    if skip_variants:
        all_instructions = list(seeds)
        print("跳过变体生成，仅使用种子问题")
    else:
        print(f"\n=== Step 1: 生成变体（每个种子 {variants_per_seed} 个）===")
        for i, seed in enumerate(seeds):
            print(f"[{i + 1}/{len(seeds)}] {seed[:40]}...", end=" ", flush=True)
            variants = _api_gen_variants(client, model, seed, variants_per_seed)
            all_instructions.append(seed)
            all_instructions.extend(variants)
            print(f"-> {1 + len(variants)} 条")
            time.sleep(delay)

    print(f"\n总共将生成 {len(all_instructions)} 条问答")
    print("\n=== Step 2: 生成高质量回答 ===")
    results = []
    fail_count = 0

    for i, instruction in enumerate(all_instructions):
        print(f"[{i + 1}/{len(all_instructions)}] {instruction[:50]}...", end=" ", flush=True)
        answer = _api_gen_answer(client, model, instruction)
        if answer:
            results.append({"instruction": instruction, "output": answer})
            print("OK")
        else:
            fail_count += 1
            print("FAIL")
        if (i + 1) % 50 == 0 and results:
            with open(output_path + ".partial", "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  --- Partial save: {len(results)} entries ---")
        time.sleep(delay)

    if results:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        if os.path.exists(output_path):
            os.replace(output_path, output_path + ".backup")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n完成！共生成 {len(results)} 条数据, 失败: {fail_count} 条")
        print(f"输出: {os.path.abspath(output_path)}")
    else:
        print("\nError: 没有成功生成任何数据")
        sys.exit(1)


# ──── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="生成 SFT 训练数据")
    parser.add_argument("--mode", choices=["hardcoded", "api"], default="hardcoded")
    parser.add_argument("--output", default="data/sft_data.jsonl")
    parser.add_argument("--count", type=int, default=10000, help="[hardcoded] 目标条目数")
    parser.add_argument(
        "--api_key", default=None, help="[api] API key (default: $env:DEEPSEEK_API_KEY)"
    )
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="[api] API endpoint")
    parser.add_argument("--api_model", default=DEFAULT_MODEL, help="[api] 模型名称")
    parser.add_argument("--variants_per_seed", type=int, default=4, help="[api] 每个种子生成变体数")
    parser.add_argument("--skip_variants", action="store_true", help="[api] 跳过变体生成")
    parser.add_argument("--delay", type=float, default=0.5, help="[api] API 调用间隔秒数")
    parser.add_argument("--dry_run", type=int, default=0, help="[api] 只运行前 N 条种子")
    args = parser.parse_args()

    if args.mode == "hardcoded":
        generate_hardcoded(args.output, args.count)
    else:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("Error: 请设置 DEEPSEEK_API_KEY 环境变量或通过 --api_key 传入")
            sys.exit(1)
        generate_api(
            args.output,
            api_key,
            args.base_url,
            args.api_model,
            args.variants_per_seed,
            args.skip_variants,
            args.delay,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
