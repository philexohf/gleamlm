"""通用文本预处理管线：清洗、去重、QA 过滤、多源混合切分。

clean_text   — 去 HTML/URL/空白，过滤短行/纯符号，可选繁简转换和广告过滤
dedup_file   — 精确/前缀/SimHash 模糊去重
filter_qa    — QA 数据专项过滤（去短答/链接/重复）
stream_split — 多源文本按配比流式混合，切分 train/valid/test
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from collections import OrderedDict
from typing import TextIO

try:
    import zhconv

    HAS_ZhCONV = True
except ImportError:
    HAS_ZhCONV = False

# ============================================================================
# clean_text
# ============================================================================

_HTML_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_URL_RE = re.compile(r"https?://\S+")
_SPACE_RE = re.compile(r"\s+")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_CHAR_RE = re.compile(r"[a-zA-Z]")

AD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"咨询.*[热热线电].*[：:]?\s*\d{3,}"),
    re.compile(r"(活动|加盟|招商|订[购车]).*[热热线电].*[：:]?\s*\d{3,}"),
    re.compile(r"[热线电话][：:]\s*\d{3,}"),
    re.compile(r"[Qq]{2}[：:]\s*\d{5,}"),
    re.compile(r"(微信|加微信|V信|vx)[：:：]\s*\S+"),
    re.compile(r"(扫码|扫一扫|关注公众号|添加客服)"),
    re.compile(r"(点击.*链接|立即.*下载|限时.*抢购|免费.*领取)"),
    re.compile(r"\d{3,}[-—]\d{3,}[-—]\d{3,}"),
    re.compile(r"(直营店|加盟店|连锁店|分店).*(覆盖|遍布|全国)"),
    re.compile(r"(特价|优惠|折扣|促销|限时|团购).*(活动|进行|开启)"),
    re.compile(r"(史上最低|年终大促|亏本甩卖|跳楼价)"),
    re.compile(r"(名额有限|先到先得|抢购|火爆.*中)"),
]

WIKI_JUNK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"镇区人口有"),
    re.compile(r"涵盖总面积为"),
    re.compile(r"(美国|United States).*人口普查"),
    re.compile(r"座标为"),
    re.compile(r"非建制地区"),
    re.compile(r"海拔高度为.*(米|英尺)"),
]


def clean_text(
    text: str,
    min_len: int = 10,
    max_len: int = 2000,
    convert_zh: bool = False,
    min_zh_ratio: float = 0.0,
    filter_ads: bool = False,
    filter_wiki_junk: bool = False,
) -> str | None:
    if not text or not text.strip():
        return None

    if convert_zh and HAS_ZhCONV:
        text = zhconv.convert(text, "zh-cn")

    text = _HTML_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()

    if len(text) < min_len or len(text) > max_len:
        return None

    chinese_chars = len(_ZH_CHAR_RE.findall(text))
    if min_zh_ratio > 0 and len(text) > 0 and chinese_chars / len(text) < min_zh_ratio:
        return None

    english_chars = len(_EN_CHAR_RE.findall(text))
    if chinese_chars + english_chars < len(text) * 0.3:
        return None

    if filter_ads:
        for pattern in AD_PATTERNS:
            if pattern.search(text):
                return None

    if filter_wiki_junk:
        for pattern in WIKI_JUNK_PATTERNS:
            if pattern.search(text):
                return None

    return text


def clean_file(
    input_path: str,
    output_path: str,
    min_len: int = 10,
    max_len: int = 2000,
    convert_zh: bool = False,
    min_zh_ratio: float = 0.0,
    filter_ads: bool = False,
    filter_wiki_junk: bool = False,
) -> None:
    total = 0
    kept = 0

    if convert_zh and not HAS_ZhCONV:
        print("WARNING: zhconv not installed, skip traditional→simplified conversion")

    print(f"Cleaning: {input_path}")

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            cleaned = clean_text(
                line, min_len, max_len, convert_zh, min_zh_ratio, filter_ads, filter_wiki_junk
            )
            if cleaned:
                fout.write(cleaned + "\n")
                kept += 1

            if total % 100000 == 0:
                print(
                    f"  Processed {total} lines, kept {kept} ({100 * kept / max(1, total):.1f}%)",
                    flush=True,
                )

    print(f"Done: {total} lines processed, {kept} kept ({100 * kept / max(1, total):.1f}%)")
    print(f"Output: {output_path}")


# ============================================================================
# dedup_text
# ============================================================================


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
    """SimHash LSH index with 4 bands × 16 bit. 100% recall for Hamming ≤ 3."""

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

            else:
                key = hashlib.md5(text[:prefix_len].encode("utf-8")).hexdigest()
                if key in seen:
                    deduped += 1
                    continue
                seen.add(key)
                fout.write(text + "\n")
                kept += 1

            if total % 100000 == 0:
                print(
                    f"  Processed {total:,} lines, kept {kept:,}, "
                    f"dedup {deduped:,} ({100 * deduped / total:.1f}%)",
                    flush=True,
                )

    pct = 100 * kept / max(1, total)
    dedup_pct = 100 * deduped / max(1, total)
    print(f"\nDone: {total:,} lines -> {kept:,} kept ({pct:.1f}%)")
    print(f"  Deduplicated: {deduped:,} ({dedup_pct:.1f}%)")
    print(f"Output: {output_path}")
    return fingerprints


# ============================================================================
# filter_qa
# ============================================================================


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

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
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

            fout.write(line)
            kept += 1

            if total % 100000 == 0:
                print(
                    f"  Processed {total:,} lines, kept {kept:,} "
                    f"(short={skipped_short:,} url={skipped_url:,} dup={skipped_dup:,})",
                    flush=True,
                )

    pct = 100 * kept / max(1, total)
    print(f"\nDone: {total:,} lines -> {kept:,} kept ({pct:.1f}%)")
    print(f"  Short answers (<{min_answer_len} chars): {skipped_short:,}")
    print(f"  URL-containing: {skipped_url:,}")
    print(f"  Duplicates: {skipped_dup:,}")
    print(f"Output: {output_path}")


# ============================================================================
# mix_split
# ============================================================================


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
        raise ValueError(f"ratios count ({len(ratios)}) != input files count ({len(input_paths)})")

    print(f"Streaming build {len(input_paths)} sources -> {output_dir}")
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
