"""繁简转换预处理 — 检测文档是否含繁体中文并自动转换。

在 pipeline 之前由用户按需运行。先抽样检测繁体特征字命中率，
超过阈值才调用 convert_zh_file 整文件转换，避免对纯简体文档
无谓消耗算力。

用法:
    python data_tools/pretrain/convert_zh.py --input data/raw/wiki_raw.txt
    python data_tools/pretrain/convert_zh.py --input data/raw/baike_raw.txt --threshold 0.02
    python data_tools/pretrain/convert_zh.py --input data/raw/news_raw.txt --force
"""

import argparse
import os
import shutil

from gleamlm.data.preprocess import convert_zh_file, detect_traditional


def main():
    parser = argparse.ArgumentParser(
        description="繁简转换预处理 — 检测繁体 → 按需转换"
    )
    parser.add_argument("--input", required=True, help="输入文本文件路径")
    parser.add_argument("--output", default=None, help="输出文件（默认覆盖输入）")
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=1000,
        help="抽样检测行数（默认 1000）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="繁体字符占比阈值（默认 0.01）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="跳过检测，强制转换",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        return

    output = args.output or args.input

    if not args.force:
        ratio = detect_traditional(args.input, args.sample_lines)
        print(
            f"繁体检测: {ratio * 100:.2f}% 命中率 (前 {args.sample_lines} 行)"
        )
        if ratio < args.threshold:
            print(f"低于阈值 {args.threshold}，跳过转换")
            return
        print(f"超过阈值 {args.threshold}，开始转换...")

    if output == args.input:
        tmp = args.input + ".tmp"
        try:
            convert_zh_file(args.input, tmp)
            shutil.move(tmp, args.input)
            print(f"已覆盖: {args.input}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        convert_zh_file(args.input, output)


if __name__ == "__main__":
    main()
