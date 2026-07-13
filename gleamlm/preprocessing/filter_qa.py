"""QA 数据专项过滤。去短答、去链接、去重复、去无效。"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import OrderedDict


def parse_qa(line: str) -> tuple[str | None, str | None]:
    def _ok(q: str, a: str) -> tuple[str | None, str | None]:
        return (q.strip(), a.strip()) if q.strip() and a.strip() else (None, None)

    text = line.strip()
    if not text:
        return None, None

    m = re.match(r"问题：(.+?)\s*回答：(.+)", text)
    if m:
        return _ok(m.group(1), m.group(2))

    m = re.match(r"Q\s*[:：]\s*(.+?)\s+A\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if m:
        return _ok(m.group(1), m.group(2))

    m = re.match(r"问\s*[:：]\s*(.+?)\s*答\s*[:：]\s*(.+)", text)
    if m:
        return _ok(m.group(1), m.group(2))

    m = re.search(r'"question"\s*:\s*"(.+?)".*?"answer"\s*:\s*"(.+?)"', text)
    if m:
        return _ok(m.group(1), m.group(2))

    m = re.match(r"(.+?)\t(.+)", text)
    if m and len(m.group(1)) > 2 and len(m.group(2)) > 5:
        return _ok(m.group(1), m.group(2))

    return None, None


def filter_qa(
    input_path: str,
    output_path: str,
    min_answer_len: int = 20,
    dedup: bool = True,
) -> None:
    total = 0
    kept = 0
    skipped_short = 0
    skipped_url = 0
    skipped_dup = 0
    seen: OrderedDict[str, bool] = OrderedDict()

    url_re = re.compile(r"https?://\S+|www\.\S+")

    print(f"Filtering QA data: {input_path}")
    print(f"  min_answer_len={min_answer_len}, dedup={dedup}")

    with open(input_path, encoding="utf-8") as fin:
        for line in fin:
            total += 1
            q, a = parse_qa(line)
            if q is None or a is None:
                continue

            if len(a) < min_answer_len:
                skipped_short += 1
                continue

            if url_re.search(a) or url_re.search(q):
                skipped_url += 1
                continue

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

    # Second pass: write filtered output (saves memory vs storing all lines)
    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        seen.clear()
        # 第二个 pass 重新遍历文件；若外部进程在两次读取间修改文件，
        # 统计数字可能与实际输出不一致。单进程批处理场景下此不触发。
        for line in fin:
            q, a = parse_qa(line)
            if q is None or a is None:
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


def main() -> None:
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
