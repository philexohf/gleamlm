"""通用文本去重。支持精确去重、前缀去重和 SimHash 全局模糊去重。

模式:
  exact  — MD5 全文哈希，剔除完全重复文档
  prefix — 前 N 字符 MD5 哈希（去重标题相同的内容）
  simhash — SimHash 指纹 + 全局 Hamming 距离（跨文本段落级去重）
"""

from __future__ import annotations

import argparse
import hashlib

_IO_BITS = 64


def normalize(text: str, strip_whitespace: bool = True) -> str:
    if strip_whitespace:
        text = " ".join(text.split())
    return text


def simhash(text: str, bits: int = 64, n: int = 3) -> int:
    if len(text) < n:
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)
    tokens = [text[i : i + n] for i in range(len(text) - n + 1)]
    v = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
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


class SimHashIndex:
    """SimHash LSH index. 4 band × 16 bit, each band acts as a hash table key.
    对于 Hamming 距离 ≤ 3 的指纹，至少 1 个 band 会完全匹配 (Recall 100%)."""

    def __init__(self, num_bands: int = 4, bits: int = 64):
        self.num_bands = num_bands
        self.band_bits = bits // num_bands
        self.mask = (1 << self.band_bits) - 1
        self.tables: list[dict[int, set[int]]] = [{} for _ in range(num_bands)]
        self._size = 0

    def add(self, fp: int) -> None:
        for band, table in enumerate(self.tables):
            key = (fp >> (band * self.band_bits)) & self.mask
            bucket = table.get(key)
            if bucket is None:
                table[key] = {fp}
            else:
                bucket.add(fp)
        self._size += 1

    def add_all(self, fingerprints: set[int]) -> None:
        for fp in fingerprints:
            self.add(fp)

    def find_candidates(self, fp: int) -> set[int]:
        candidates: set[int] = set()
        for band, table in enumerate(self.tables):
            key = (fp >> (band * self.band_bits)) & self.mask
            bucket = table.get(key)
            if bucket is not None:
                candidates.update(bucket)
        return candidates

    def __len__(self) -> int:
        return self._size


def dedup_file(
    input_path: str,
    output_path: str,
    mode: str = "exact",
    prefix_len: int = 100,
    simhash_threshold: int = 3,
    existing_fingerprints: set[int] | None = None,
) -> set[int]:
    total = 0
    kept = 0
    deduped = 0
    seen: set[str] = set()
    fingerprints: set[int] = set(existing_fingerprints) if existing_fingerprints else set()

    index: SimHashIndex | None = None
    if mode == "simhash":
        index = SimHashIndex()
        if fingerprints:
            index.add_all(fingerprints)

    print(f"Dedup: {input_path}")
    if mode == "simhash":
        print(
            f"  mode=simhash, threshold={simhash_threshold}, "
            f"initial fingerprints={len(fingerprints)}"
        )

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
                candidates = index.find_candidates(fp)  # type: ignore[union-attr]
                if any(hamming_distance(fp, c) <= simhash_threshold for c in candidates):
                    deduped += 1
                    continue
                fingerprints.add(fp)
                index.add(fp)  # type: ignore[union-attr]
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
    print(f"\nDone: {total:,} lines -> {kept:,} kept ({pct:.1f}%)")
    print(f"  Deduplicated: {deduped:,} ({dedup_pct:.1f}%)")
    print(f"Output: {output_path}")
    return fingerprints


def _is_similar(fp: int, fingerprints: set[int], threshold: int) -> bool:
    return any(hamming_distance(fp, prev) <= threshold for prev in fingerprints)


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
        "--prefix_len", type=int, default=100, help="prefix 模式下的字符数（默认100）"
    )
    parser.add_argument(
        "--simhash_threshold",
        type=int,
        default=3,
        help="SimHash Hamming 距离阈值，<=此值视为重复（默认3）",
    )
    args = parser.parse_args()

    dedup_file(
        args.input,
        args.output,
        args.mode,
        args.prefix_len,
        args.simhash_threshold,
    )


if __name__ == "__main__":
    main()
