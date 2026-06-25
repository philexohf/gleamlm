"""通用文本去重。支持精确去重和近邻去重（按前 N 字符哈希）。
适用于新闻、百科、维基等非 QA 数据。

用法：
    python tools/dedup_text.py --input data/raw/news_clean.txt --output data/raw/news_dedup.txt
    python tools/dedup_text.py --input data/raw/baike_clean.txt --output data/raw/baike_dedup.txt
    python tools/dedup_text.py --input data/raw/wiki_clean.txt --output data/raw/wiki_dedup.txt
"""

import hashlib
import argparse


def normalize(text, strip_whitespace=True):
    """轻量归一化：去首尾空白，合并连续空格"""
    if strip_whitespace:
        text = ' '.join(text.split())
    return text


def dedup_file(input_path, output_path, mode="exact", prefix_len=100):
    """逐行去重，流式处理不占内存。mode=exact 全文MD5 / prefix 前N字符"""
    total = 0
    kept = 0
    deduped = 0
    seen = set()

    print(f"Dedup: {input_path}")
    print(f"  mode={mode}, prefix_len={prefix_len}")

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total += 1
            text = normalize(line.strip())
            if not text:
                continue

            if mode == "exact":
                key = hashlib.md5(text.encode('utf-8')).hexdigest()
            else:  # prefix
                key = hashlib.md5(text[:prefix_len].encode('utf-8')).hexdigest()

            if key in seen:
                deduped += 1
                continue

            seen.add(key)
            fout.write(text + '\n')
            kept += 1

            if total % 500000 == 0:
                print(f"  Processed {total:,} lines, kept {kept:,}, "
                      f"dedup {deduped:,} ({100 * deduped / total:.1f}%)")

    pct = 100 * kept / max(1, total)
    dedup_pct = 100 * deduped / max(1, total)
    print(f"\nDone: {total:,} lines → {kept:,} kept ({pct:.1f}%)")
    print(f"  Deduplicated: {deduped:,} ({dedup_pct:.1f}%)")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='通用文本去重工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件')
    parser.add_argument('--output', type=str, required=True, help='输出文件')
    parser.add_argument('--mode', type=str, default='exact',
                        choices=['exact', 'prefix'],
                        help='exact=全文精确去重, prefix=前N字符去重（默认exact）')
    parser.add_argument('--prefix_len', type=int, default=100,
                        help='prefix 模式下的字符数（默认100）')
    args = parser.parse_args()

    dedup_file(args.input, args.output, args.mode, args.prefix_len)


if __name__ == '__main__':
    main()
