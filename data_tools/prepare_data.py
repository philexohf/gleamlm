"""数据预处理一键管道：去重 → QA过滤 → 多源混合切分 → 清理旧缓存。

用法：
    python data_tools/prepare_data.py

    自定义配比：
    python data_tools/prepare_data.py --ratios 0.40 0.23 0.22 0.15

    跳过已完成的步骤（断点续跑）：
    python data_tools/prepare_data.py --skip_dedup
"""

import argparse
import os
import subprocess
import sys

# 数据源配置（顺序即混合轮询优先级，配比影响每轮读取行数）
SOURCES = [
    {"name": "wiki", "file": "wiki_clean.txt", "type": "text", "ratio": 0.30},
    {"name": "baike", "file": "baike_clean.txt", "type": "text", "ratio": 0.12},
    {"name": "news", "file": "news_clean.txt", "type": "news", "ratio": 0.43},
    {"name": "qa", "file": "qa_clean.txt", "type": "qa", "ratio": 0.15},
]


def run(cmd_list, desc):
    """运行子命令，失败则报错退出"""
    print(f"\n{'=' * 60}")
    print(f"[{desc}]")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd_list, shell=False)
    if result.returncode != 0:
        print(f"ERROR: {desc} 失败 (exit={result.returncode})")
        sys.exit(1)


def compute_avg_chars(filepath, sample_lines=50000):
    """扫描文件前 N 行，估算行均字符数（用于字符加权配比）"""
    total_chars = 0
    lines = 0
    try:
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                stripped = line.strip()
                if stripped:
                    total_chars += len(stripped)
                    lines += 1
    except Exception:
        return 0
    return total_chars / max(1, lines)


def main():
    parser = argparse.ArgumentParser(description="数据预处理一键管道")
    parser.add_argument("--input", type=str, default="data/raw", help="原始数据目录")
    parser.add_argument("--output", type=str, default="data/nano_data", help="输出目录")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help="多源配比，顺序与 SOURCES 一致，默认 0.30 0.12 0.43 0.15",
    )
    parser.add_argument("--skip_clean", action="store_true", help="跳过清洗（输入已是 clean 文件）")
    parser.add_argument("--skip_dedup", action="store_true", help="跳过去重")
    parser.add_argument(
        "--dedup_mode",
        type=str,
        default="exact",
        choices=["exact", "prefix"],
        help="去重模式 (exact/prefix)",
    )
    parser.add_argument("--prefix_len", type=int, default=100, help="prefix 模式下的字符数")
    args = parser.parse_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))

    if args.ratios:
        for i, s in enumerate(SOURCES):
            s["ratio"] = args.ratios[i]

    # 1. 清洗（raw → clean）
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
            extra = []
            if s["name"] == "wiki":
                extra += ["--min_zh_ratio", "0.15", "--filter_wiki_junk"]
            if s["name"] == "news":
                extra += ["--filter_ads"]
            run(
                [
                    "python",
                    os.path.join(tools_dir, "clean_text.py"),
                    "--input",
                    raw,
                    "--output",
                    clean,
                    "--min_len",
                    "10",
                    "--max_len",
                    "2000",
                    "--convert_zh",
                ]
                + extra,
                f"清洗: {s['name']} ({os.path.basename(raw)} → {os.path.basename(clean)})",
            )

    # 2. 去重 / QA过滤
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
                run(
                    [
                        "python",
                        os.path.join(tools_dir, "filter_qa.py"),
                        "--input",
                        src,
                        "--output",
                        deduped,
                    ],
                    f"QA过滤: {s['name']}",
                )
            else:
                # 新闻用 prefix 模式（标题去重），其他用 exact（全文去重）
                mode = "prefix" if s["type"] == "news" else args.dedup_mode
                run(
                    [
                        "python",
                        os.path.join(tools_dir, "dedup_text.py"),
                        "--input",
                        src,
                        "--output",
                        deduped,
                        "--mode",
                        mode,
                        "--prefix_len",
                        str(args.prefix_len),
                    ],
                    f"去重: {s['name']} (mode={mode})",
                )

    # 3. 字符加权配比 → 行数配比（业界标准：按字符量而非行数混合）
    print("\n[3/4] 多源混合切分")
    input_files = []
    target_ratios = [s["ratio"] for s in SOURCES]

    # 扫描各源行均字符
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

    # 按字符量加权：target_ratio / avg_chars → 行数配比
    effective = []
    for i, s in enumerate(SOURCES):
        if avg_chars_list[i] > 0:
            effective.append(target_ratios[i] / avg_chars_list[i])
        else:
            effective.append(0)
    total = sum(effective)
    if total > 0:
        effective = [e / total for e in effective]

    # 取有效文件
    valid_files = []
    valid_ratios = []
    for i, s in enumerate(SOURCES):
        if input_files[i] is not None and effective[i] > 0:
            valid_files.append(input_files[i])
            valid_ratios.append(f"{effective[i]:.4f}")
            print(
                f"  {s['name']}: 目标{target_ratios[i] * 100:.0f}% 字符 → 行数配比 {effective[i] * 100:.1f}%"
            )
        elif input_files[i] is not None:
            valid_files.append(input_files[i])
            valid_ratios.append("0.0001")
            print(f"  {s['name']}: 目标{target_ratios[i] * 100:.0f}% 字符 → 文件为空，跳过")

    if len(valid_files) < 2:
        print("ERROR: 有效数据源不足 2 个")
        sys.exit(1)

    cmd_list = (
        ["python", os.path.join(tools_dir, "build_dataset.py"), "--input"]
        + valid_files
        + ["--ratios"]
        + valid_ratios
        + ["--output_dir", args.output]
    )
    run(cmd_list, f"build_dataset ({len(input_files)} sources)")

    # 4. 清理旧 token 缓存
    print("\n[4/4] 清理旧 token 缓存")
    for split in ["train", "valid", "test"]:
        cache = os.path.join(args.output, f"{split}_ids.npy")
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  Removed: {cache}")
        else:
            print(f"  Clean: {split}_ids.npy (no old cache)")

    print("\n" + "=" * 60)
    print("数据预处理完成!")
    print(f"输出目录: {args.output}")
    for s in SOURCES:
        print(f"  {s['name']}: {s['ratio'] * 100:.0f}%")
    print(f"\n下一步: python train.py --data_dir {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
