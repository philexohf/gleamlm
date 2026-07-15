"""数据预处理一键管道：清洗 → 去重 → QA过滤 → 多源混合切分 → 清理旧缓存。

用法：
    python data_tools/prepare_data.py

    自定义配比：
    python data_tools/prepare_data.py --ratios 0.40 0.23 0.22 0.15

    跳过已完成的步骤（断点续跑）：
    python data_tools/prepare_data.py --skip_dedup
"""

import argparse
import os
import sys

from gleamlm.preprocessing.build_dataset import stream_build
from gleamlm.preprocessing.clean_text import clean_file
from gleamlm.preprocessing.dedup_text import dedup_file
from gleamlm.preprocessing.filter_qa import filter_qa

# 数据源配置（顺序即混合轮询优先级，配比影响每轮读取行数）
SOURCES = [
    {"name": "wiki", "file": "wiki_clean.txt", "type": "text", "ratio": 0.30},
    {"name": "baike", "file": "baike_clean.txt", "type": "text", "ratio": 0.12},
    {"name": "news", "file": "news_clean.txt", "type": "news", "ratio": 0.43},
    {"name": "qa", "file": "qa_clean.txt", "type": "qa", "ratio": 0.15},
]


def compute_avg_chars(filepath):
    total = 0
    lines = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            total += len(line)
            lines += 1
    return total / max(1, lines)


def main():
    parser = argparse.ArgumentParser(description="数据预处理一键管道")
    parser.add_argument("--input", default="data/raw", help="原始数据目录")
    parser.add_argument("--output", default="data/nano_data", help="输出目录")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=4,
        default=[s["ratio"] for s in SOURCES],
        help="4 源字符配比（wiki baike news qa）",
    )
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_dedup", action="store_true")
    parser.add_argument("--dedup_mode", default="exact", choices=["exact", "prefix"])
    parser.add_argument("--prefix_len", type=int, default=100)
    args = parser.parse_args()

    # 应用自定义配比
    for i, r in enumerate(args.ratios):
        SOURCES[i]["ratio"] = r

    # ──── 1. 清洗 ────
    if args.skip_clean:
        print("\n[1/4] 跳过清洗（--skip_clean）")
    else:
        print("\n[1/4] 清洗原始数据（raw → clean）")
        for s in SOURCES:
            raw = os.path.join(args.input, s["file"].replace("_clean", "_raw"))
            clean = os.path.join(args.input, s["file"])
            if not os.path.exists(raw):
                print(f"  Skip {s['name']}: {raw} not found (already clean)")
                continue
            if os.path.exists(clean) and os.path.getsize(clean) > 0:
                print(f"  Skip {s['name']}: {clean} exists (already cleaned)")
                continue
            print(f"  Cleaning: {s['name']} ({os.path.basename(raw)} → {os.path.basename(clean)})")
            clean_file(
                raw,
                clean,
                min_len=10,
                max_len=2000,
                convert_zh=True,
                min_zh_ratio=0.15 if s["name"] == "wiki" else 0.0,
                filter_ads=s["name"] == "news",
                filter_wiki_junk=s["name"] == "wiki",
            )

    # ──── 2. 去重 / QA过滤 ────
    if args.skip_dedup:
        print("\n[2/4] 跳过去重（--skip_dedup）")
    else:
        print("\n[2/4] 去重 & QA过滤")
        for s in SOURCES:
            src = os.path.join(args.input, s["file"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: {src} not found")
                continue
            deduped = os.path.join(args.input, f"{s['name']}_dedup.txt")
            if os.path.exists(deduped):
                print(f"  Skip {s['name']}: {deduped} exists (already deduped)")
                continue
            if s["type"] == "qa":
                print(f"  QA过滤: {s['name']}")
                filter_qa(src, deduped)
            else:
                mode = "prefix" if s["type"] == "news" else args.dedup_mode
                print(f"  去重: {s['name']} (mode={mode})")
                dedup_file(src, deduped, mode=mode, prefix_len=args.prefix_len)

    # ──── 3. 字符加权配比 → 行数配比 ────
    print("\n[3/4] 多源混合切分")
    input_files = []
    target_ratios = [s["ratio"] for s in SOURCES]

    print("  扫描各源行均字符...")
    avg_chars_list = []
    for s in SOURCES:
        deduped = os.path.join(args.input, f"{s['name']}_dedup.txt")
        src = os.path.join(args.input, s["file"])
        fpath = deduped if os.path.exists(deduped) else src
        if not os.path.exists(fpath):
            print(f"    WARNING: {s['name']} 源文件不存在，跳过")
            avg_chars_list.append(0)
            input_files.append(None)
            continue
        avg_c = compute_avg_chars(fpath)
        avg_chars_list.append(avg_c)
        input_files.append(fpath)
        print(f"    {s['name']}: ~{avg_c:.0f} 字/行")

    effective = []
    for i, s in enumerate(SOURCES):
        if avg_chars_list[i] > 0:
            effective.append(target_ratios[i] / avg_chars_list[i])
        else:
            effective.append(0)
    total = sum(effective)
    if total > 0:
        effective = [e / total for e in effective]

    valid_files: list[str] = []
    valid_ratios: list[float] = []
    for i, s in enumerate(SOURCES):
        if input_files[i] is not None and effective[i] > 0:
            valid_files.append(str(input_files[i]))
            valid_ratios.append(effective[i])
            print(
                f"  {s['name']}: 目标{target_ratios[i] * 100:.0f}% 字符 → "
                f"行数配比 {effective[i] * 100:.1f}%"
            )
        elif input_files[i] is not None:
            valid_files.append(str(input_files[i]))
            valid_ratios.append(0.0001)
            print(f"  {s['name']}: 目标{target_ratios[i] * 100:.0f}% 字符 → 文件为空，跳过")

    if len(valid_files) < 2:
        print("ERROR: 有效数据源不足 2 个")
        sys.exit(1)

    stream_build(
        input_paths=valid_files,
        output_dir=args.output,
        ratios=valid_ratios,
        train_ratio=0.9,
        valid_ratio=0.05,
    )

    # ──── 4. 清理旧 token 缓存 ────
    print("\n[4/4] 清理旧 token 缓存")
    for split in ("train", "valid", "test"):
        cache = os.path.join(args.output, f"{split}_ids.npy")
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  Removed: {cache}")
    print("  完成")


if __name__ == "__main__":
    main()
