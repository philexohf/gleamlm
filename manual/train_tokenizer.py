"""
BBPE tokenizer 训练/扩展脚本

用法:
  # 按 variant 配比从零训练 (从 configs/{variant}.yaml 读 data_sources)
  python manual/train_tokenizer.py --variant nano --vocab_size 12002 \
    --save_dir gleamlm/tokenizer/checkpoints/bbpe_12k --max-chars 200000000

  # 从 data_dir 目录训练 (所有 .txt 等分)
  python manual/train_tokenizer.py --vocab_size 24002 \
    --data_dir ./data/raw --save_dir checkpoints/bbpe_24k

  # 从已有 12K tokenizer 扩展到 24K（复用已有 merge 结果）
  python manual/train_tokenizer.py --vocab_size 24002 \
    --base_tokenizer checkpoints/bbpe_12k --save_dir checkpoints/bbpe_24k

  # 训练完后验证
  python manual/train_tokenizer.py --verify_only checkpoints/bbpe_24k
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import load_config


def extend_tokenizer(base_path: str, target_vocab: int, save_dir: str):
    tokenizer = BBPETokenizer.load(base_path)
    current_vocab = tokenizer.get_vocab_size()
    n_needed = target_vocab - current_vocab

    if n_needed <= 0:
        print(f"Already at {current_vocab} >= {target_vocab}, nothing to do.")
        return

    print(f"Extending {current_vocab} → {target_vocab} ({n_needed} new merges)")

    existing_ids = sorted(tokenizer.merge_pairs.keys())
    added = 0
    for i in range(0, len(existing_ids) - 1, 2):
        if added >= n_needed:
            break
        a, b = existing_ids[i], existing_ids[i + 1]
        # 防碰撞: 该 pair 已是已有 merge 时跳过，否则会覆盖原 merge 的 id 映射。
        if (a, b) in tokenizer.merges:
            continue
        new_id = tokenizer._next_id
        tokenizer.merges[(a, b)] = new_id
        tokenizer.merge_pairs[new_id] = (a, b)
        tokenizer.id_to_byte[new_id] = tokenizer.id_to_byte[a] + tokenizer.id_to_byte[b]
        tokenizer._next_id += 1
        added += 1

    if added < n_needed:
        for i in range(256):
            if added >= n_needed:
                break
            for eid in existing_ids[-100:]:
                if added >= n_needed:
                    break
                a, b = i + tokenizer._byte_offset, eid
                if (a, b) in tokenizer.merges:
                    continue
                new_id = tokenizer._next_id
                tokenizer.merges[(a, b)] = new_id
                tokenizer.merge_pairs[new_id] = (a, b)
                tokenizer.id_to_byte[new_id] = tokenizer.id_to_byte[a] + tokenizer.id_to_byte[b]
                tokenizer._next_id += 1
                added += 1

    tokenizer.save(save_dir)
    print(f"Extended to {tokenizer.get_vocab_size()} (added {n_needed} merges)")


def verify_tokenizer(path: str):
    path = os.path.join(path, "bbpe_tokenizer.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    merges = data["merges"]
    merge_pairs = data["merge_pairs"]
    next_id = data["_next_id"]
    special_count = len(data["special_tokens"])

    print(f"vocab_size: {next_id}")
    print(f"special_tokens: {special_count}")
    print(f"merges: {len(merges)}")
    print(f"merge_pairs: {len(merge_pairs)}")

    bad_refs = 0
    for mid_str, (a, b) in merge_pairs.items():
        mid = int(mid_str)
        if mid < special_count + 256:
            bad_refs += 1
            print(f"  ERROR: merge ID {mid} conflicts with byte/special range")
        if f"{a} {b}" not in merges:
            bad_refs += 1
            print(f"  ERROR: merge ({a}, {b}) → {mid} not found in merges dict")

    if bad_refs:
        print(f"FAILED: {bad_refs} errors")
    else:
        print("PASSED: structure valid")


def train_from_variant(
    variant: str, vocab_size: int, save_dir: str, max_chars: int
) -> BBPETokenizer:
    """按 configs/{variant}.yaml 的 data_sources 配比训练分词器。

    数据文件定位: data/raw/{name}_dedup.txt（与数据 pipeline 约定一致）。
    """
    cfg = load_config(os.path.join("configs", f"{variant}.yaml"))
    data_sources = cfg.data_sources
    if not data_sources:
        raise SystemExit(f"ERROR: configs/{variant}.yaml 无 data_sources 配比定义")

    files, ratios = [], []
    for s in data_sources:
        path = os.path.join("data", "raw", f"{s.name}_dedup.txt")
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: 数据文件不存在: {path}")
        files.append(path)
        ratios.append(s.ratio)

    print(f"[配比] {variant}.yaml data_sources → {','.join(f'{r:.0%}:{os.path.basename(f)}' for f, r in zip(files, ratios))}")
    return BBPETokenizer.train_from_files(
        files, vocab_size=vocab_size, save_dir=save_dir,
        max_train_chars=max_chars, ratios=ratios,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="",
                   help="模型变体 (nano/lite/pro): 从 configs/{variant}.yaml 读 data_sources 配比训练")
    p.add_argument("--vocab_size", type=int, default=12002)
    p.add_argument("--max_chars", type=int, default=200_000_000,
                   help="训练语料字符预算 (默认 200M，足够 12K 词表)")
    p.add_argument("--data_dir", type=str, default="", help="Raw text directory for training from scratch")
    p.add_argument("--base_tokenizer", type=str, default="", help="Extend from existing tokenizer dir")
    p.add_argument("--save_dir", type=str, default="gleamlm/tokenizer/checkpoints/bbpe_12k")
    p.add_argument("--verify_only", type=str, default="", help="Verify tokenizer structure")
    args = p.parse_args()

    if args.verify_only:
        verify_tokenizer(args.verify_only)
    elif args.base_tokenizer:
        extend_tokenizer(args.base_tokenizer, args.vocab_size, args.save_dir)
    elif args.variant:
        tokenizer = train_from_variant(args.variant, args.vocab_size, args.save_dir, args.max_chars)
        print(f"完成! vocab_size={tokenizer.get_vocab_size()}, saved={args.save_dir}")
    elif args.data_dir:
        files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".txt")]
        tokenizer = BBPETokenizer.train_from_files(files, vocab_size=args.vocab_size, save_dir=args.save_dir)
    else:
        print("Provide --variant / --data_dir (from scratch) or --base_tokenizer (extend) or --verify_only")
