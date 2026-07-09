"""提取 Parquet 数据集的 text 列，写入纯文本文件。

支持 pyarrow 或 fastparquet 后端。流式逐文件处理，内存可控。

用法:
    python data_tools/extract_parquet.py
        --input data/chinese-fineweb-edu/IndustryCorpus \
        --output data/raw/edu.txt \
        --max_files 5
"""

import argparse
import glob
import os
import sys


def extract_with_pyarrow(files, output_path, text_col="text"):
    """使用 pyarrow 读取 parquet，流式逐 batch 写入 .txt"""
    import pyarrow.parquet as pq

    total_size = sum(os.path.getsize(f) for f in files)
    print(f"输入: {len(files)} 个 parquet 文件, {total_size / 1e9:.2f} GB")
    print(f"输出: {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        total_lines = 0
        for i, fpath in enumerate(files):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(columns=[text_col]):
                col = batch.column(text_col)
                for j in range(len(col)):
                    text = col[j].as_py()
                    if text and isinstance(text, str):
                        out.write(text.rstrip() + "\n")
                        total_lines += 1

            if (i + 1) % 5 == 0:
                print(f"  [{i + 1}/{len(files)}] {total_lines:,} 行")

    print(f"完成: {total_lines:,} 行")


def extract_with_fastparquet(files, output_path, text_col="text"):
    """使用 fastparquet 读取 parquet（更轻量，无需 pyarrow）"""
    import pandas as pd

    total_size = sum(os.path.getsize(f) for f in files)
    print(f"输入: {len(files)} 个 parquet 文件, {total_size / 1e9:.2f} GB")
    print(f"输出: {output_path}")
    print("后端: fastparquet (性能较低，建议安装 pyarrow)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        total_lines = 0
        for i, fpath in enumerate(files):
            df = pd.read_parquet(fpath, columns=[text_col])
            for text in df[text_col].dropna():
                out.write(str(text).rstrip() + "\n")
                total_lines += 1

            if (i + 1) % 5 == 0:
                print(f"  [{i + 1}/{len(files)}] {total_lines:,} 行")

    print(f"完成: {total_lines:,} 行")


def main():
    parser = argparse.ArgumentParser(description="提取 Parquet → 纯文本")
    parser.add_argument("--input", type=str, required=True, help="Parquet 文件目录路径")
    parser.add_argument("--output", type=str, required=True, help="输出 .txt 文件路径")
    parser.add_argument("--text_col", type=str, default="text", help="文本列名（默认: text）")
    parser.add_argument(
        "--max_files", type=int, default=0, help="最多提取前 N 个 parquet 文件（0=全部）"
    )
    args = parser.parse_args()

    # 收集文件列表
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not files:
        print(f"未找到 parquet 文件: {args.input}")
        sys.exit(1)

    if args.max_files > 0:
        files = files[: args.max_files]
        print(f"限制: 仅处理前 {args.max_files} 个文件")

    # 优先 pyarrow，回退 fastparquet
    try:
        import pyarrow.parquet

        extract_fn = extract_with_pyarrow
    except ImportError:
        print("pyarrow 未安装，尝试 fastparquet...")
        try:
            import pandas as pd

            pd.read_parquet
            extract_fn = extract_with_fastparquet
        except (ImportError, AttributeError):
            print("错误: 需要安装 pyarrow 或 fastparquet")
            print("  pip install pyarrow")
            print("  pip install fastparquet pandas")
            sys.exit(1)

    extract_fn(files, args.output, args.text_col)


if __name__ == "__main__":
    main()
