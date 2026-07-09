"""
构建训练数据集（流式版本）

将多个清洗后的文本文件按配比交织合并，用 <|endoftext|> 分隔文档，
然后切分为 train/valid/test。支持 GB 级大文件，无需全量读入内存。
"""

import argparse
import os
import random


def stream_build(
    input_paths,
    output_dir,
    train_ratio=0.9,
    valid_ratio=0.05,
    ratios=None,
    total_lines=None,
    buf_size=50000,
):
    """
    流式构建数据集：多文件按配比轮询读取 → 缓存区打乱 → 写输出
    """
    os.makedirs(output_dir, exist_ok=True)

    if ratios is None:
        ratios = [1.0 / len(input_paths)] * len(input_paths)
    if len(ratios) != len(input_paths):
        raise ValueError(f"ratios 数量 ({len(ratios)}) 与输入文件数 ({len(input_paths)}) 不匹配")

    print(f"Streaming build {len(input_paths)} sources → {output_dir}")
    for p, r in zip(input_paths, ratios, strict=False):
        print(f"  {os.path.basename(p)}: {r * 100:.0f}%")

    separator = "\n<|endoftext|>\n"

    # 打开输出文件
    train_f = open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8")
    valid_f = open(os.path.join(output_dir, "valid.txt"), "w", encoding="utf-8")
    test_f = open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8")

    # 打开输入文件
    readers = []
    for path in input_paths:
        readers.append(open(path, encoding="utf-8"))

    random.seed(42)
    train_lines = valid_lines = test_lines = 0
    total = 0
    first_line = {train_f: True, valid_f: True, test_f: True}

    try:
        active = len(readers)
        source_counts = [0] * len(readers)

        while active > 0:
            for idx in range(len(readers)):
                if readers[idx] is None:
                    continue

                # 按配比轮询：ratio 大的源多读几行
                burst = max(1, int(buf_size * ratios[idx] / max(ratios)))
                burst_lines = []

                for _ in range(burst):
                    line = readers[idx].readline()
                    if not line:
                        readers[idx].close()
                        readers[idx] = None
                        active -= 1
                        break
                    stripped = line.strip()
                    if stripped:
                        burst_lines.append(stripped)

                if not burst_lines:
                    continue

                source_counts[idx] += len(burst_lines)
                total += len(burst_lines)

                # 缓冲区打乱
                random.shuffle(burst_lines)

                # 写入输出（90/5/5 切分）
                for line in burst_lines:
                    r = random.random()
                    if r < train_ratio:
                        target = train_f
                        train_lines += 1
                    elif r < train_ratio + valid_ratio:
                        target = valid_f
                        valid_lines += 1
                    else:
                        target = test_f
                        test_lines += 1

                    if not first_line[target]:
                        target.write(separator)
                    else:
                        first_line[target] = False
                    target.write(line)

            # 进度
            if total % (buf_size * 5) < buf_size:
                print(f"\r  Processed {total:,} lines", end="", flush=True)

    finally:
        for r in readers:
            if r is not None:
                r.close()
        train_f.close()
        valid_f.close()
        test_f.close()

    print(f"\r  Processed {total:,} lines total")
    for i, (path, cnt) in enumerate(zip(input_paths, source_counts, strict=False)):
        print(f"  {os.path.basename(path)}: {cnt:,} lines ({ratios[i] * 100:.0f}% target)")

    print("\nDataset built:")
    print(f"  Train: {train_lines:,} lines ({train_lines / max(1, total) * 100:.0f}%)")
    print(f"  Valid: {valid_lines:,} lines ({valid_lines / max(1, total) * 100:.0f}%)")
    print(f"  Test:  {test_lines:,} lines ({test_lines / max(1, total) * 100:.0f}%)")

    for name in ["train.txt", "valid.txt", "test.txt"]:
        path = os.path.join(output_dir, name)
        size = os.path.getsize(path)
        print(f"  {name}: {size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="构建训练数据集（流式）")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="输入文本文件路径")
    parser.add_argument("--output_dir", type=str, default="./data/nano_data")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--ratios", type=float, nargs="+", default=None, help="数据源配比")
    parser.add_argument("--buf_size", type=int, default=50000, help="打乱缓冲区大小")
    args = parser.parse_args()

    stream_build(
        args.input,
        args.output_dir,
        args.train_ratio,
        args.valid_ratio,
        ratios=args.ratios,
        buf_size=args.buf_size,
    )


if __name__ == "__main__":
    main()
