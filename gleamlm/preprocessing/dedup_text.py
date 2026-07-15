"""通用文本去重。支持精确去重、前缀去重和 SimHash 模糊去重。

模式:
  exact  — MD5 全文哈希，剔除完全重复文档
  prefix — 前 N 字符 MD5 哈希（去重标题相同的内容）
  simhash — SimHash 指纹 + 滑动窗口 Hamming 距离（跨文本段落级去重）
"""

from __future__ import annotations

import argparse
import hashlib


def normalize(text: str, strip_whitespace: bool = True) -> str:
    if strip_whitespace:
        text = " ".join(text.split())
    return text


def simhash(text: str, bits: int = 64) -> int:
    v = [0] * bits
    for token in text:
        h = hash(token)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedup_file(
    input_path: str,
    output_path: str,
    mode: str = "exact",
    prefix_len: int = 100,
    simhash_threshold: int = 3,
    simhash_window: int = 1000,
) -> None:
    total = 0
    kept = 0
    deduped = 0
    seen: set[str] = set()
    simhash_history: list[int] = []

    print(f"Dedup: {input_path}")
    if mode == "simhash":
        print(f"  mode=simhash, threshold={simhash_threshold}, window={simhash_window}")
    else:
        print(f"  mode={mode}, prefix_len={prefix_len}")

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            text = normalize(line.strip())
            if not text:
                continue

            if mode == "simhash":
                fp = simhash(text)
                is_dup = any(
                    hamming_distance(fp, prev) <= simhash_threshold
                    for prev in simhash_history[-simhash_window:]
                )
                if is_dup:
                    deduped += 1
                    continue
                simhash_history.append(fp)
                if len(simhash_history) > simhash_window:
                    simhash_history.pop(0)
                fout.write(text + "\n")
                kept += 1

            elif mode == "exact":
                key = hashlib.md5(text.encode("utf-8")).hexdigest()
                if key in seen:
                    deduped += 1
                    continue
                seen.add(key)
                fout.write(text + "\n")
                kept += 1

            else:  # prefix
                key = hashlib.md5(text[:prefix_len].encode("utf-8")).hexdigest()
                if key in seen:
                    deduped += 1
                    continue
                seen.add(key)
                fout.write(text + "\n")
                kept += 1

            if total % 500000 == 0:
                print(
                    f"  Processed {total:,} lines, kept {kept:,}, "
                    f"dedup {deduped:,} ({100 * deduped / total:.1f}%)"
                )

    pct = 100 * kept / max(1, total)
    dedup_pct = 100 * deduped / max(1, total)
    print(f"\nDone: {total:,} lines → {kept:,} kept ({pct:.1f}%)")
    print(f"  Deduplicated: {deduped:,} ({dedup_pct:.1f}%)")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="通用文本去重工具")
    parser.add_argument("--input", type=str, required=True, help="输入文件")
    parser.add_argument("--output", type=str, required=True, help="输出文件")
    parser.add_argument(
        "--mode",
        type=str,
        default="exact",
        choices=["exact", "prefix", "simhash"],
        help="exact=精确全文去重, prefix=前N字符, simhash=模糊去重",
    )
    parser.add_argument(
        "--prefix_len",
        type=int,
        default=100,
        help="prefix 模式下的字符数（默认100）",
    )
    parser.add_argument(
        "--simhash_threshold",
        type=int,
        default=3,
        help="SimHash Hamming 距离阈值，<=此值视为重复（默认3）",
    )
    parser.add_argument(
        "--simhash_window",
        type=int,
        default=1000,
        help="SimHash 滑动窗口大小（默认1000）",
    )
    args = parser.parse_args()

    dedup_file(
        args.input,
        args.output,
        args.mode,
        args.prefix_len,
        args.simhash_threshold,
        args.simhash_window,
    )


if __name__ == "__main__":
    main()
