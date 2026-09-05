"""构建 DPO v2 chosen 池（与当前 SFT 数据 v2 / 新 policy 对齐）。

数据源（均为当前基础集，chosen = SFT 高质答案）：
  - chat_extra.jsonl         169 条闲聊（v2 实测弱点场景，全量）
  - sft_data.jsonl 单轮      2,654 条中确定性抽 1,000（知识/模板，output≤600 字防超长截断）
  - sft_data.jsonl 多轮        761 条全量（6 轮对话结构）

产出两个中间文件（喂给 generate_rejected.py 生成 rejected，跑完即可删）：
  - dpo_chosen_single.jsonl  {instruction, output}
  - dpo_chosen_multi.jsonl   {messages}（尾轮为 assistant 答案，脚本内部剥离）

用法:
  python data_tools/dpo/build_dpo_chosen.py
"""

import argparse
import json
import random

SFT_DATA = "data/nano/sft/sft_data.jsonl"
CHAT_EXTRA = "data/nano/sft/chat_extra.jsonl"
SINGLE_OUT = "data/nano/dpo/dpo_chosen_single.jsonl"
MULTI_OUT = "data/nano/dpo/dpo_chosen_multi.jsonl"

# 单轮 output 字符上限：防 encode 后超过 DPO max_seq_len（1024 token）导致训练截断损坏
MAX_OUT_CHARS = 600
# 多轮对话轮数固定为 6 条消息（3 轮 user/assistant）
MULTI_TURNS = 6


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build DPO v2 chosen pool")
    parser.add_argument("--single-n", type=int, default=1000, help="单轮抽取数")
    parser.add_argument("--seed", type=int, default=42, help="确定性抽样 seed")
    args = parser.parse_args()

    # ---- 单轮源：chat_extra 闲聊全量 + sft_data 单轮抽 n ----
    chat_extra = load_jsonl(CHAT_EXTRA)
    sft_data = load_jsonl(SFT_DATA)
    singles = [r for r in sft_data if "instruction" in r and "output" in r]
    # instruction 去重（保第一条）
    seen_ins: set[str] = set()
    seen_extra: set[str] = set()
    chat_rows, single_rows = [], []
    for r in chat_extra:
        ins = r["instruction"]
        if ins not in seen_extra:
            seen_extra.add(ins)
            chat_rows.append(r)
    for r in singles:
        ins = r["instruction"]
        if ins not in seen_ins and ins not in seen_extra:
            seen_ins.add(ins)
            single_rows.append(r)
    print(f"chat_extra: {len(chat_rows)} 条（全量）")
    print(f"sft_data 单轮去重后: {len(single_rows)} 条")

    # 优先 output <= MAX_OUT_CHARS（防训练截断），不足时放宽
    short = [r for r in single_rows if len(r["output"]) <= MAX_OUT_CHARS]
    long = [r for r in single_rows if len(r["output"]) > MAX_OUT_CHARS]
    print(f"  其中 output<=600 字: {len(short)} 条, >600 字: {len(long)} 条")
    rng = random.Random(args.seed)
    if len(short) >= args.single_n:
        picked = rng.sample(short, args.single_n)
    else:
        picked = short + rng.sample(long, args.single_n - len(short))
        print(f"  !! 短答案池不足，从长答案补抽 {args.single_n - len(short)} 条（有截断风险）")
    print(f"单轮抽取: {len(picked)} 条")

    # ---- 多轮源：messages 全量（尾轮须为 assistant 答案）----
    multis = [r for r in sft_data if "messages" in r]
    valid_multi = []
    for r in multis:
        msgs = r["messages"]
        if len(msgs) != MULTI_TURNS or not msgs or msgs[-1].get("role") != "assistant":
            continue
        valid_multi.append(r)
    print(f"多轮: {len(valid_multi)}/{len(multis)} 条（6 轮结构）")

    # ---- 写中间文件 ----
    with open(SINGLE_OUT, "w", encoding="utf-8") as f:
        for r in chat_rows + picked:
            f.write(
                json.dumps({"instruction": r["instruction"], "output": r["output"]}, ensure_ascii=False)
                + "\n"
            )
    with open(MULTI_OUT, "w", encoding="utf-8") as f:
        for r in valid_multi:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    total = len(chat_rows) + len(picked) + len(valid_multi)
    print(f"Done: single={len(chat_rows) + len(picked)} (闲聊 {len(chat_rows)} + 单轮 {len(picked)}), "
          f"multi={len(valid_multi)}, 合计 {total} 对")


if __name__ == "__main__":
    main()
