"""数据预处理管道（与变体无关，所有变体共用）。

流程:
  step 1: 粗精确去重 (MD5)  — 先剔除完全重复，减少后续清洗/去重计算量
  step 2: 基础清洗           — 去乱码、繁转简、过滤广告/低质
  step 3: SimHash 模糊去重 / QA过滤 — 跨文本段落级去重

输出 → data/raw/{name}_dedup.txt（供 build.py 混合切分使用）

用法:
    python data_tools/pretrain/pipeline.py
    python data_tools/pretrain/pipeline.py --skip_simhash
"""

import argparse
import os

from gleamlm.preprocessing.clean_text import clean_file
from gleamlm.preprocessing.dedup_text import dedup_file
from gleamlm.preprocessing.filter_qa import filter_qa

SOURCES = [
    {"name": "edu", "type": "text"},
    {"name": "news", "type": "news"},
    {"name": "wiki", "type": "text"},
    {"name": "baike", "type": "text"},
    {"name": "qa", "type": "qa"},
]


def _raw_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw.txt")


def _raw_dedup_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw_dedup.txt")


def _clean_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_clean.txt")


def _final_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_dedup.txt")


def main():
    parser = argparse.ArgumentParser(description="数据预处理管道（去重→清洗→模糊去重）")
    parser.add_argument("--input", default="data/raw", help="原始数据目录")
    parser.add_argument("--skip_exact_dedup", action="store_true")
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_simhash", action="store_true")
    parser.add_argument("--exact_mode", default="exact", choices=["exact", "prefix"])
    parser.add_argument("--prefix_len", type=int, default=100)
    parser.add_argument("--simhash_threshold", type=int, default=3)
    parser.add_argument("--simhash_window", type=int, default=1000)
    args = parser.parse_args()

    raw_dir = args.input
    names = [s["name"] for s in SOURCES]
    print(f"Sources: {names}")
    print(f"Input: {raw_dir}")

    # ──── step 1: 粗精确去重 ────
    if args.skip_exact_dedup:
        print("\n[1/3] 跳过精确去重（--skip_exact_dedup）")
    else:
        print("\n[1/3] 粗精确去重（MD5 全文去重）")
        for s in SOURCES:
            raw = _raw_path(raw_dir, s["name"])
            deduped = _raw_dedup_path(raw_dir, s["name"])
            if not os.path.exists(raw):
                print(f"  Skip {s['name']}: {raw} not found")
                continue
            if os.path.exists(deduped) and os.path.getsize(deduped) > 0:
                print(f"  Skip {s['name']}: {deduped} exists")
                continue
            mode = "prefix" if s["type"] == "news" else args.exact_mode
            print(f"  去重: {s['name']} (mode={mode})")
            dedup_file(raw, deduped, mode=mode, prefix_len=args.prefix_len)

    # ──── step 2: 清洗 ────
    if args.skip_clean:
        print("\n[2/3] 跳过清洗（--skip_clean）")
    else:
        print("\n[2/3] 基础清洗（去乱码、繁转简、过滤低质）")
        for s in SOURCES:
            src = _raw_dedup_path(raw_dir, s["name"])
            if not os.path.exists(src):
                src = _raw_path(raw_dir, s["name"])
            clean = _clean_path(raw_dir, s["name"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: no source found")
                continue
            if os.path.exists(clean) and os.path.getsize(clean) > 0:
                print(f"  Skip {s['name']}: {clean} exists")
                continue
            print(f"  Cleaning: {s['name']}")
            clean_file(
                src,
                clean,
                min_len=30,
                max_len=3000,
                convert_zh=True,
                min_zh_ratio=0.15 if s["name"] in ("wiki", "edu") else 0.0,
                filter_ads=s["name"] == "news",
                filter_wiki_junk=s["name"] == "wiki",
            )

    # ──── step 3: SimHash 模糊去重 / QA过滤 ────
    if args.skip_simhash:
        print("\n[3/3] 跳过模糊去重（--skip_simhash）")
    else:
        print("\n[3/3] SimHash 模糊去重 / QA过滤")
        for s in SOURCES:
            src = _clean_path(raw_dir, s["name"])
            if not os.path.exists(src):
                src = _final_path(raw_dir, s["name"])
            final = _final_path(raw_dir, s["name"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: {src} not found")
                continue
            if os.path.exists(final) and os.path.getsize(final) > 0:
                print(f"  Skip {s['name']}: {final} exists")
                continue

            if s["type"] == "qa":
                print(f"  QA过滤: {s['name']}")
                filter_qa(src, final)
            else:
                print(
                    f"  SimHash: {s['name']} "
                    f"(threshold={args.simhash_threshold}, window={args.simhash_window})"
                )
                dedup_file(
                    src,
                    final,
                    mode="simhash",
                    simhash_threshold=args.simhash_threshold,
                    simhash_window=args.simhash_window,
                )

    print("  完成")


if __name__ == "__main__":
    main()
