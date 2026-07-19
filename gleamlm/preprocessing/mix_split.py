"""构建训练数据集（流式版本）。

将多个清洗后的文本文件按配比交织合并，用 <|im_start|> ... <|im_end|> 包裹每个文档，
然后切分为 train/valid/test。支持 GB 级大文件，无需全量读入内存。
"""

from __future__ import annotations

import argparse
import os
import random
from typing import TextIO


def stream_split(
    input_paths: list[str],
    output_dir: str,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    ratios: list[float] | None = None,
    buf_size: int = 50000,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if ratios is None:
        ratios = [1.0 / len(input_paths)] * len(input_paths)
    if len(ratios) != len(input_paths):
        raise ValueError(f"ratios 数量 ({len(ratios)}) 与输入文件数 ({len(input_paths)}) 不匹配")

    print(f"Streaming build {len(input_paths)} sources → {output_dir}")
    for p, r in zip(input_paths, ratios, strict=True):
        print(f"  {os.path.basename(p)}: {r * 100:.0f}%")

    random.seed(42)

    train_f = valid_f = test_f = None
    readers: list[TextIO | None] = []
    try:
        train_f = open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8")  # noqa: SIM115
        valid_f = open(os.path.join(output_dir, "valid.txt"), "w", encoding="utf-8")  # noqa: SIM115
        test_f = open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8")  # noqa: SIM115
        for path in input_paths:
            readers.append(open(path, encoding="utf-8"))  # noqa: SIM115

        train_lines = valid_lines = test_lines = 0
        total = 0

        active = len(readers)
        source_counts = [0] * len(readers)

        while active > 0:
            for idx in range(len(readers)):
                if readers[idx] is None:
                    continue

                burst = max(1, int(buf_size * ratios[idx] / max(ratios)))
                burst_lines: list[str] = []

                for _ in range(burst):
                    rdr = readers[idx]
                    if rdr is None:
                        break
                    line = rdr.readline()
                    if not line:
                        rdr.close()
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

                random.shuffle(burst_lines)

                for line_text in burst_lines:
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

                    target.write(f"<|im_start|>{line_text}<|im_end|>\n")

            if total % (buf_size * 5) < buf_size:
                print(f"\r  Processed {total:,} lines", end="", flush=True)

    finally:
        for reader in readers:
            if reader is not None:
                reader.close()
        if train_f is not None:
            train_f.close()
        if valid_f is not None:
            valid_f.close()
        if test_f is not None:
            test_f.close()

    print(f"\r  Processed {total:,} lines total")
    for i, (path, cnt) in enumerate(zip(input_paths, source_counts, strict=True)):
        print(f"  {os.path.basename(path)}: {cnt:,} lines ({ratios[i] * 100:.0f}% target)")

    print("\nDataset built:")
    print(f"  Train: {train_lines:,} lines ({train_lines / max(1, total) * 100:.0f}%)")
    print(f"  Valid: {valid_lines:,} lines ({valid_lines / max(1, total) * 100:.0f}%)")
    print(f"  Test:  {test_lines:,} lines ({test_lines / max(1, total) * 100:.0f}%)")

    for name in ["train.txt", "valid.txt", "test.txt"]:
        path = os.path.join(output_dir, name)
        size = os.path.getsize(path)
        print(f"  {name}: {size / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建训练数据集（流式）")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="输入文本文件路径")
    parser.add_argument("--output_dir", type=str, default="./data/nano/pretrain")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--ratios", type=float, nargs="+", default=None, help="数据源配比")
    parser.add_argument("--buf_size", type=int, default=50000, help="打乱缓冲区大小")
    args = parser.parse_args()

    stream_split(
        args.input,
        args.output_dir,
        args.train_ratio,
        args.valid_ratio,
        ratios=args.ratios,
        buf_size=args.buf_size,
    )


if __name__ == "__main__":
    main()
