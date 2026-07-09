"""QA 数据专项过滤。去短答、去链接、去重复、去无效。

用法：
    python data_tools/filter_qa.py --input data/raw/qa_clean.txt --output data/raw/qa_filtered.txt
"""

import argparse
import hashlib
import re
from collections import OrderedDict


def parse_qa(line):
    """解析 Q/A 对，支持多种格式。

    格式：
        问题：{q} 回答：{a}
        Q: {q} A: {a}
        {"question": "...", "answer": "..."}
        问：{q} 答：{a}
    """
    text = line.strip()
    if not text:
        return None, None

    def _ok(q, a):
        """两个部分均非空才返回"""
        return (q.strip(), a.strip()) if q.strip() and a.strip() else (None, None)

    # 格式 1：问题：{q} 回答：{a}（原始格式）
    m = re.match(r"问题：(.+?)\s*回答：(.+)", text)
    if m:
        return _ok(m.group(1), m.group(2))

    # 格式 2：Q: {q} A: {a} / Q：{q} A：{a}
    m = re.match(r"Q\s*[:：]\s*(.+?)\s+A\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if m:
        return _ok(m.group(1), m.group(2))

    # 格式 3：问：{q} 答：{a}
    m = re.match(r"问\s*[:：]\s*(.+?)\s*答\s*[:：]\s*(.+)", text)
    if m:
        return _ok(m.group(1), m.group(2))

    # 格式 4：JSON 片段 {"question": "...", "answer": "..."}
    m = re.search(r'"question"\s*:\s*"(.+?)".*?"answer"\s*:\s*"(.+?)"', text)
    if m:
        return _ok(m.group(1), m.group(2))

    # 格式 5：Q/A 以 tab 分隔（qa 保留率很低时回退尝试）
    m = re.match(r"(.+?)\t(.+)", text)
    if m and len(m.group(1)) > 2 and len(m.group(2)) > 5:
        return _ok(m.group(1), m.group(2))

    return None, None


def filter_qa(input_path, output_path, min_answer_len=20, dedup=True):
    """
    过滤 QA 数据。min_answer_len=20 过滤短答，dedup 基于 Q-hash 去重
    """
    total = 0
    kept = 0
    skipped_short = 0
    skipped_url = 0
    skipped_dup = 0
    seen = OrderedDict()  # 保持顺序的去重

    # URL 正则
    url_re = re.compile(r"https?://\S+|www\.\S+")

    print(f"Filtering QA data: {input_path}")
    print(f"  min_answer_len={min_answer_len}, dedup={dedup}")

    with open(input_path, encoding="utf-8") as fin:
        for line in fin:
            total += 1
            q, a = parse_qa(line)
            if q is None:
                continue

            # 过滤短答
            if len(a) < min_answer_len:
                skipped_short += 1
                continue

            # 过滤含 URL 的回答
            if url_re.search(a) or url_re.search(q):
                skipped_url += 1
                continue

            # 去重（基于 Q 的哈希值）
            if dedup:
                q_hash = hashlib.md5(q.encode("utf-8")).hexdigest()
                if q_hash in seen:
                    skipped_dup += 1
                    continue
                seen[q_hash] = True

            kept += 1

            if total % 500000 == 0:
                print(
                    f"  Processed {total:,} lines, kept {kept:,} "
                    f"(short={skipped_short:,} url={skipped_url:,} dup={skipped_dup:,})"
                )

    # 第二次扫描写入（节省内存）
    with open(input_path, encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            seen.clear()
            for line in fin:
                q, a = parse_qa(line)
                if q is None:
                    continue
                if len(a) < min_answer_len:
                    continue
                if url_re.search(a) or url_re.search(q):
                    continue
                if dedup:
                    q_hash = hashlib.md5(q.encode("utf-8")).hexdigest()
                    if q_hash in seen:
                        continue
                    seen[q_hash] = True
                fout.write(line)

    pct = 100 * kept / max(1, total)
    print(f"\nDone: {total:,} lines → {kept:,} kept ({pct:.1f}%)")
    print(f"  Short answers (<{min_answer_len} chars): {skipped_short:,}")
    print(f"  URL-containing: {skipped_url:,}")
    print(f"  Duplicates: {skipped_dup:,}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="QA 数据专项过滤器")
    parser.add_argument("--input", type=str, required=True, help="输入 QA 文件")
    parser.add_argument("--output", type=str, required=True, help="输出文件")
    parser.add_argument("--min_answer_len", type=int, default=20, help="回答最小字符数（默认 20）")
    parser.add_argument(
        "--no_dedup", action="store_true", default=False, help="禁用去重（默认开启）"
    )
    args = parser.parse_args()

    filter_qa(args.input, args.output, args.min_answer_len, dedup=not args.no_dedup)


if __name__ == "__main__":
    main()
