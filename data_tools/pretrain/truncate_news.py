"""新闻文本截断：按句子边界将超长文档拆分为 ≤400 字的短文档。

news 原始均长 ~885 字，远超其他源 (156-186 字)，截断后缩小到 ~350 字，
改善多源混合时长度分布的平衡性。

用法:
    python data_tools/pretrain/truncate_news.py
    python data_tools/pretrain/truncate_news.py --input data/raw/news_clean.txt \
        --output data/raw/news_dedup.txt --max_len 400
"""

import argparse
import os
import re

SENT_SPLIT_RE = re.compile(r"([。！？；\n]+)")


def truncate_news(input_path: str, output_path: str, max_len: int = 400) -> None:
    total = 0
    total_out = 0
    buf: list[str] = []
    buf_len = 0

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            text = line.strip()
            if not text:
                if buf:
                    fout.write("".join(buf) + "\n")
                    total_out += 1
                    buf.clear()
                    buf_len = 0
                fout.write("\n")
                continue

            sentences = [s for s in SENT_SPLIT_RE.split(text) if s]
            for sent in sentences:
                slen = len(sent)
                if slen > max_len:
                    if buf:
                        fout.write("".join(buf) + "\n")
                        total_out += 1
                        buf.clear()
                        buf_len = 0
                    for i in range(0, slen, max_len):
                        chunk = sent[i : i + max_len]
                        fout.write(chunk + "\n")
                        total_out += 1
                    continue

                if buf_len + slen > max_len and buf:
                    fout.write("".join(buf) + "\n")
                    total_out += 1
                    buf.clear()
                    buf_len = 0

                buf.append(sent)
                buf_len += slen

            if total % 200000 == 0:
                print(f"  Processed {total:,} lines → {total_out:,} output", flush=True)

        if buf:
            fout.write("".join(buf) + "\n")
            total_out += 1

    in_size = os.path.getsize(input_path) / 1e6
    out_size = os.path.getsize(output_path) / 1e6
    print(f"\nDone: {total:,} → {total_out:,} documents")
    print(f"  {in_size:.0f} MB → {out_size:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="新闻文本按句子边界截断")
    parser.add_argument("--input", default="data/raw/news_clean.txt")
    parser.add_argument("--output", default="data/raw/news_trunc.txt")
    parser.add_argument("--max_len", type=int, default=400)
    args = parser.parse_args()
    truncate_news(args.input, args.output, args.max_len)


if __name__ == "__main__":
    main()
