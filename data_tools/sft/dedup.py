"""SFT 数据去重与闲聊样本合并（数据维护工具）。

背景：data/nano/sft/sft_data.jsonl 由硬编码模板（data_tools/sft/_templates.py）
多次追加拼接生成，同一段 output 一字不差重复出现（实测单组最多 29 连），
把个别模板段落权重放大数十倍 —— 40M 模型对训练分布外的输入（闲聊/寒暄）
会陷入"最高先验主题复读"（如反复输出道歉/追责类文本）。

策略：
1. instruction/output 单轮样本按 output 全文去重，每组保留 1 条
   （组内挑选 instruction 最"正常"者：短、且未把答案整段嵌入问句）
2. messages 多轮样本不参与去重，原样保留
3. 可选 --chat-extra 追加闲聊样本文件（补模型没学过的对话域）

用法:
  # 预览（只统计不写文件）
  python data_tools/sft/dedup.py --input data/nano/sft/sft_data.jsonl

  # 执行：备份原件 → 去重写回 → 合并闲聊样本
  python data_tools/sft/dedup.py --input data/nano/sft/sft_data.jsonl \
      --chat-extra data/nano/sft/chat_extra.jsonl --apply
"""

import argparse
import json
import os
import sys


def instruction_quality(ins: str, out: str) -> float:
    """instruction 质量分：越低越好。惩罚把答案整段嵌进问句的畸形样本。"""
    score = float(len(ins))
    head = out[:15].strip()
    if head and head in ins:
        score += 1000.0  # 答案开头被塞进问句 → 畸形生成样本
    return score


def main():
    parser = argparse.ArgumentParser(description="SFT 数据 output 去重")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL")
    parser.add_argument("--chat-extra", type=str, default=None, help="闲聊样本 JSONL（追加到输出末尾）")
    parser.add_argument("--apply", action="store_true", help="真正写文件；缺省为预览")
    args = parser.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: invalid JSON at line {ln}: {e}", file=sys.stderr)

    n_total = len(rows)
    single = [r for r in rows if "output" in r]
    multi = [r for r in rows if "messages" in r and "output" not in r]
    print(f"Total: {n_total}  (single-turn: {len(single)}, messages: {len(multi)})")

    # 按 output 全文分组
    groups: dict[str, list] = {}
    for r in single:
        groups.setdefault(r["output"].strip(), []).append(r)

    keep, removed = [], 0
    dup_groups = 0
    for out, members in groups.items():
        if len(members) == 1:
            keep.append(members[0])
            continue
        dup_groups += 1
        removed += len(members) - 1
        best = min(members, key=lambda m: instruction_quality(m.get("instruction", ""), out))
        keep.append(best)
        if args.apply and len(members) > 2:
            # 预览模式打印大重复组样例
            pass
    print(f"Dup groups: {dup_groups}, removed: {removed}, kept: {len(keep) + len(multi)}")

    # 大重复组摘要（>=5 连）
    for out, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]:
        if len(members) >= 5:
            sample = members[0].get("instruction", "")[:40]
            print(f"  {len(members)}x | ins: {sample}")

    if not args.apply:
        print("(preview mode — add --apply to write)")
        return

    # 备份原文件
    backup = args.input.replace(".jsonl", "_bak_dup.jsonl")
    if not os.path.exists(backup):
        with open(args.input, encoding="utf-8") as f_src, open(backup, "w", encoding="utf-8") as f_dst:
            f_dst.write(f_src.read())
        print(f"Backup: {backup}")

    chat_n = 0
    if args.chat_extra:
        with open(args.chat_extra, encoding="utf-8") as f:
            chat_rows = [json.loads(l) for l in f if l.strip()]
        chat_n = len(chat_rows)
        print(f"Chat extra: {chat_n}")

    with open(args.input, "w", encoding="utf-8") as f:
        for r in keep + multi:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if args.chat_extra:
            for r in chat_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written: {args.input}  (total {len(keep) + len(multi) + chat_n})")


if __name__ == "__main__":
    main()
