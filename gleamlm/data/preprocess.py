"""数据预处理引擎 — 文件流式（工业范式），所有变体共用。

流程（由 gleamlm.data.pipeline.run_pipeline 编排）:
  step 1  粗去重     dedup_file(mode=exact / news 用 prefix)   → {name}_raw_dedup.txt
  step 2  清洗       clean_file(去HTML/URL/广告/wiki垃圾/zh比) → {name}_clean.txt
  step 3  质量过滤   score_quality_file(Gopher 5 规则)         → {name}_quality.txt
  step 4  细去重     dedup_file(mode=simhash) / minhash_dedup_file → {name}_dedup.txt + .fps
  step 5  配比切分   stream_split(字符占比 → 概率分派)          → {prefix}_{train|valid|test}.txt
  step 6  打包       gleamlm.data.pack                    → {prefix}_{split}.bin/.idx

繁简转换: convert_zh / convert_zh_file 为独立函数，由 data_tools/pretrain/convert_zh.py 按需调用，不耦合在清洗管线中。

设计要点:
  - 全部核心函数以 文件→文件 的流式形态存在，任意时刻内存只有一行
    → 规模上限由磁盘决定（TB 级可行），这是 Megatron/GPT-3 同款路线
  - 每阶段产物落盘、存在即跳过 → 断点续跑（幂等）
  - SimHash 用 LSH 分桶索引（O(1) 候选查桶），拒绝 O(n²) 全量比对
  - MinHash 签名 + band 分桶 → Jaccard 近似去重（工业标准）
  - 指纹与输出分离: 指纹基于 NFKC 归一化文本（检出全半角/大小写变体），
    输出保留原文，不做 lower/改写
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import unicodedata
from collections import OrderedDict
from typing import TextIO

try:
    import zhconv

    HAS_ZHCONV = True
except ImportError:
    HAS_ZHCONV = False

# ============================================================================

_HTML_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_URL_RE = re.compile(r"https?://\S+")
_SPACE_RE = re.compile(r"\s+")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_CHAR_RE = re.compile(r"[a-zA-Z]")
_RE_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")  # 非打印控制字符

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

# Gopher 质量评分权重（长度/语言/标点/符号/重复度）
_QUALITY_WEIGHTS = {"len": 0.25, "lang": 0.25, "punct": 0.15, "sym": 0.20, "rep": 0.15}


# ============================================================================
# 归一化
# ============================================================================


def normalize(text: str, strip_whitespace: bool = True) -> str:
    """NFKC 归一化 + 小写 + 空白压缩（用于指纹/查重键，不改写输出文本）。

    NFKC: 兼容性分解 + 规范化合成，把全角/半角、罗马数字、连字等
    兼容字符统一为最简形式（如 ① → 1、Ａ → A）。
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    if strip_whitespace:
        t = " ".join(t.split())
    return t


# ============================================================================
# 清洗
# ============================================================================


def clean_text(
    text: str,
    min_len: int = 10,
    max_len: int = 2000,
    min_zh_ratio: float = 0.0,
    min_en_ratio: float = 0.0,
    filter_ads: bool = False,
    filter_wiki_junk: bool = False,
) -> str | None:
    if not text or not text.strip():
        return None

    text = _HTML_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _RE_CONTROL.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()

    if len(text) < min_len or len(text) > max_len:
        return None

    chinese_chars = len(_ZH_CHAR_RE.findall(text))
    if min_zh_ratio > 0 and len(text) > 0 and chinese_chars / len(text) < min_zh_ratio:
        return None

    english_chars = len(_EN_CHAR_RE.findall(text))
    if min_en_ratio > 0 and len(text) > 0 and english_chars / len(text) < min_en_ratio:
        return None
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
    min_zh_ratio: float = 0.0,
    min_en_ratio: float = 0.0,
    filter_ads: bool = False,
    filter_wiki_junk: bool = False,
) -> None:
    """逐行清洗：输入文件 → 输出文件，O(1) 内存。"""
    total = 0
    kept = 0

    print(f"Cleaning: {input_path}")

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            cleaned = clean_text(
                line, min_len, max_len, min_zh_ratio, min_en_ratio, filter_ads, filter_wiki_junk
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
# 繁简转换（独立于清洗管线，由 convert_zh.py 按需调用）
# ============================================================================

# 繁体特征字集 — 抽样检测用，覆盖高频繁简异体
_TRADITIONAL_CHARS = frozenset(
    "國學書車長門開關頭體實發時會個後為對說來東過機見問張萬裡電"
    "無與當於進種還經動從現點將業處樣變結組導華話權風戰興認達"
    "傳轉選標際強準題際許師術備設報場單層義連階隊統論計產"
)


def detect_traditional(input_path: str, sample_lines: int = 1000) -> float:
    """抽样检测繁体特征字命中率。

    只扫描前 sample_lines 行，统计繁体特征字占比，
    避免对纯简体文档加载 zhconv。

    Returns:
        float: 繁体字符占比（0.0-1.0），输入为空时返回 0
    """
    total_chars = 0
    traditional_chars = 0
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            for ch in line.strip():
                total_chars += 1
                if ch in _TRADITIONAL_CHARS:
                    traditional_chars += 1
    return traditional_chars / max(total_chars, 1)


def convert_zh(text: str) -> str:
    """繁体中文 → 简体中文转换。zhconv 未安装时原样返回。"""
    if HAS_ZHCONV:
        return zhconv.convert(text, "zh-cn")
    return text


def convert_zh_file(input_path: str, output_path: str) -> None:
    """文件级流式繁→简转换，O(1) 内存。"""
    if not HAS_ZHCONV:
        raise RuntimeError("zhconv not installed, cannot convert traditional→simplified")

    total = 0
    converted = 0
    print(f"繁→简转换: {input_path}")
    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            text = line.strip()
            if not text:
                fout.write("\n")
                continue
            fout.write(zhconv.convert(text, "zh-cn") + "\n")
            converted += 1
            if total % 200000 == 0:
                print(f"  Processed {total:,} lines", flush=True)
    print(f"Done: {total:,} lines, {converted:,} written")
    print(f"Output: {output_path}")


# ============================================================================
# SimHash 去重（LSH 分桶索引）
# ============================================================================


def _feature_hash(feature: str, seed: int = 42) -> int:
    """滚动字符串 hash（比逐 token md5 快一个数量级）。"""
    h = seed
    for c in feature.encode():
        h = ((h << 5) - h) + c
        h &= 0xFFFFFFFFFFFFFFFF
    return h


def simhash(text: str, bits: int = 64, n: int = 3, hash_seed: int = 42) -> int:
    if len(text) < n:
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)
    tokens = [text[i : i + n] for i in range(len(text) - n + 1)]
    v = [0] * bits
    for token in tokens:
        h = _feature_hash(token, hash_seed)
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


# ============================================================================
# MinHash 去重（工业标准近似去重）
# ============================================================================


class MinHash:
    """MinHash 签名 — 文档集合的 Jaccard 相似度无偏估计。"""

    def __init__(self, num_perm: int = 64, shingle_size: int = 5, seed: int = 42):
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._prime = 2_147_483_647
        self._a = [(seed + 31 * i) % (self._prime - 1) + 1 for i in range(num_perm)]
        self._b = [(seed + 17 * i) % (self._prime - 1) + 1 for i in range(num_perm)]

    @staticmethod
    def _feature_hash(s: str) -> int:
        return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")

    def _shingles(self, text: str) -> set[int]:
        text = _SPACE_RE.sub(" ", text).strip()
        if len(text) < self.shingle_size:
            return {self._feature_hash(text)}
        return {
            self._feature_hash(text[i : i + self.shingle_size])
            for i in range(len(text) - self.shingle_size + 1)
        }

    def signature(self, text: str) -> list[int]:
        shingles = self._shingles(text)
        sig = [self._prime] * self.num_perm
        for s in shingles:
            for i in range(self.num_perm):
                h = (self._a[i] * s + self._b[i]) % self._prime
                if h < sig[i]:
                    sig[i] = h
        return sig

    @staticmethod
    def jaccard_from_signatures(sig_a: list[int], sig_b: list[int]) -> float:
        return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


class MinHashIndex:
    """MinHash LSH 索引 — 签名按 band 分桶，候选 O(1) 查桶。"""

    def __init__(self, num_perm: int = 64, bands: int = 16):
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self.buckets: dict[tuple, list[list[int]]] = {}

    def _keys(self, sig: list[int]) -> list[tuple]:
        rp = self.rows_per_band
        return [(b, tuple(sig[b * rp : (b + 1) * rp])) for b in range(self.bands)]

    def add(self, sig: list[int]) -> None:
        for key in self._keys(sig):
            self.buckets.setdefault(key, []).append(sig)

    def find_candidates(self, sig: list[int]) -> list[list[int]]:
        """收集所有 band 命中的候选签名（按 id 去重，避免重复比对）。"""
        seen: dict[int, list[int]] = {}
        for key in self._keys(sig):
            for s in self.buckets.get(key, ()):
                seen.setdefault(id(s), s)
        return list(seen.values())


# ============================================================================
# 去重（文件 → 文件，返回指纹集合供跨源复用）
# ============================================================================


def dedup_file(
    input_path: str,
    output_path: str,
    mode: str = "exact",
    prefix_len: int = 100,
    simhash_threshold: int = 3,
    existing_fingerprints: set[int] | None = None,
) -> set[int]:
    """文件级去重。

    mode:
      exact   — 归一化（NFKC）后 MD5 全文去重，检出全半角/大小写变体
      prefix  — 归一化后前 prefix_len 字符 MD5 去重（新闻模板头专用）
      simhash — LSH 分桶 + Hamming ≤ threshold 判定近似重复

    返回保留文档的指纹集合；existing_fingerprints 注入他源指纹实现跨源去重。
    """
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
            text = line.strip()
            if not text:
                continue
            norm = normalize(text)  # 指纹用归一化文本，输出保留原文

            if mode == "simhash":
                fp = simhash(norm)
                candidates = index.find_candidates(fp)  # type: ignore[union-attr]
                if any(hamming_distance(fp, c) <= simhash_threshold for c in candidates):
                    deduped += 1
                    continue
                fingerprints.add(fp)
                index.add(fp)  # type: ignore[union-attr]
                fout.write(text + "\n")
                kept += 1

            elif mode == "exact":
                key = hashlib.md5(norm.encode("utf-8")).hexdigest()
                if key in seen:
                    deduped += 1
                    continue
                seen.add(key)
                fout.write(text + "\n")
                kept += 1

            else:  # prefix
                key = hashlib.md5(norm[:prefix_len].encode("utf-8")).hexdigest()
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


def minhash_dedup_file(
    input_path: str,
    output_path: str,
    threshold: float = 0.8,
    num_perm: int = 64,
    bands: int = 16,
) -> None:
    """MinHash + LSH 近似去重 — Jaccard ≥ threshold 视为重复，保留首个。"""
    mh = MinHash(num_perm=num_perm)
    index = MinHashIndex(num_perm=num_perm, bands=bands)
    total = 0
    kept = 0
    deduped = 0

    print(f"Dedup: {input_path}")
    print(f"  mode=minhash, jaccard_threshold={threshold}, num_perm={num_perm}, bands={bands}")

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            text = line.strip()
            if not text:
                continue
            sig = mh.signature(normalize(text))
            if any(
                MinHash.jaccard_from_signatures(sig, cand) >= threshold
                for cand in index.find_candidates(sig)
            ):
                deduped += 1
                continue
            index.add(sig)
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


# ============================================================================
# QA 数据专项
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
    """QA 专项过滤：去短答 / 含 URL / 问题重复。"""
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
# 质量评分（Gopher / RedPajama 风格多规则）
# ============================================================================


def score_text(text: str) -> float:
    """Gopher 5 规则加权质量分（0-1）：长度 / 语言 / 标点 / 符号 / 重复度。"""
    n = max(len(text), 1)

    length_score = min(len(text) / 1000.0, 1.0)

    lang_score = max(
        len(_ZH_CHAR_RE.findall(text)) / n,
        len(_EN_CHAR_RE.findall(text)) / n,
    )

    punct_ratio = len(re.findall(r"[，。！？、；：\u201c\u201d]", text)) / n
    punct_score = min(punct_ratio * 20, 1.0)

    n_symbols = len(re.findall(r"[^\w\u4e00-\u9fff\s，。！？、；：\u201c\u201d\u2018\u2019]", text))
    symbol_ratio = n_symbols / n
    symbol_score = 1.0 if symbol_ratio < 0.1 else max(1.0 - (symbol_ratio - 0.1) * 5, 0.0)

    head = text[:2000]
    grams = [head[i : i + 5] for i in range(max(len(head) - 4, 1))]
    uniq = len(set(grams)) / max(len(grams), 1)
    repeat_score = 1.0 if uniq >= 0.5 else max(uniq * 2, 0.0)

    return (
        _QUALITY_WEIGHTS["len"] * length_score
        + _QUALITY_WEIGHTS["lang"] * lang_score
        + _QUALITY_WEIGHTS["punct"] * punct_score
        + _QUALITY_WEIGHTS["sym"] * symbol_score
        + _QUALITY_WEIGHTS["rep"] * repeat_score
    )


def score_quality_file(input_path: str, output_path: str, min_score: float = 0.30) -> None:
    """逐行打分并按阈值过滤低质文档（Gopher 式），流式 O(1) 内存。"""
    total = 0
    kept = 0

    print(f"Quality filter: {input_path}")
    print(f"  min_score={min_score}")

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            text = line.strip()
            if not text:
                continue
            if score_text(text) >= min_score:
                fout.write(text + "\n")
                kept += 1

            if total % 100000 == 0:
                print(
                    f"  Processed {total:,} lines, kept {kept:,} "
                    f"({100 * kept / max(1, total):.1f}%)",
                    flush=True,
                )

    pct = 100 * kept / max(1, total)
    print(f"\nDone: {total:,} lines -> {kept:,} kept ({pct:.1f}%)")
    print(f"Output: {output_path}")


# ============================================================================
# 多源混合切分（字符占比配比，流式）
# ============================================================================


def stream_split(
    input_paths: list[str],
    output_dir: str,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    ratios: list[float] | None = None,
    buf_size: int = 50000,
    max_chars: int | None = None,
    seed: int = 42,
    output_prefix: str | None = None,
) -> list[dict]:
    """多源按目标字符占比流式混合，切分 train/valid/test 纯文本。

    两遍扫描（均 O(1) 内存）:
      pass 1  统计每源行数与总字符数 → 计算字符预算 budget_i = max_chars * ratio_i
             （max_chars 默认取最受限源: min(total_chars_i / ratio_i)）
      pass 2  交替读取各源，每轮按预算比例读入 burst（字符数控制），
             shuffle 后按概率分派到三个输出文件；累计字符达到
             max_chars 总量上限即停止混合。

    输出: {output_dir}/train.txt|valid.txt|test.txt（或 output_prefix 指定前缀）。
    返回每源统计列表。
    """
    os.makedirs(output_dir, exist_ok=True)

    if ratios is None:
        ratios = [1.0 / len(input_paths)] * len(input_paths)
    if len(ratios) != len(input_paths):
        raise ValueError(f"ratios count ({len(ratios)}) != input files count ({len(input_paths)})")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios):.4f}")

    # ── pass 1: 统计 ────────────────────────────────────────────────────────
    stats: list[dict] = []
    for p in input_paths:
        rows = chars = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows += 1
                    chars += len(s)
        stats.append({"path": p, "rows": rows, "total_chars": chars})

    if max_chars is None:
        candidates = [
            st["total_chars"] / r
            for st, r in zip(stats, ratios, strict=True)
            if r > 0 and st["total_chars"] > 0
        ]
        max_chars = int(min(candidates)) if candidates else 0
    budgets = [max_chars * r for r in ratios]

    print(f"Streaming build {len(input_paths)} sources -> {output_dir}")
    for st, r in zip(stats, ratios, strict=True):
        print(
            f"  {os.path.basename(st['path'])}: {r * 100:.0f}%  "
            f"({st['rows']:,} rows, {st['total_chars']:,} chars)"
        )
    print(f"  char budget: max_chars={max_chars:,}")

    random.seed(seed)

    def _out(name: str) -> str:
        return f"{output_prefix}_{name}.txt" if output_prefix else os.path.join(output_dir, name + ".txt")

    train_f = open(_out("train"), "w", encoding="utf-8")  # noqa: SIM115
    valid_f = open(_out("valid"), "w", encoding="utf-8")  # noqa: SIM115
    test_f = open(_out("test"), "w", encoding="utf-8")  # noqa: SIM115
    readers: list[TextIO | None] = [open(p, encoding="utf-8") for p in input_paths]  # noqa: SIM115

    counts = {"train": 0, "valid": 0, "test": 0}
    total = 0
    total_chars = 0
    active = len(readers)
    max_budget = max(budgets) or 1.0
    source_out = [0] * len(readers)
    spent_chars = [0] * len(readers)
    # max_chars 是字符总量上限: 达到即停止混合 (配比由 burst 加权保证)；
    # 无总量约束时小源耗尽后大源独占输出，最终比例完全偏离设计值。
    char_limit = max_chars if max_chars and max_chars > 0 else float("inf")

    try:
        while active > 0 and total_chars < char_limit:
            for idx in range(len(readers)):
                rdr = readers[idx]
                if rdr is None:
                    continue

                # 全局预算封顶: burst 不超过剩余总量
                global_remaining = char_limit - total_chars
                if global_remaining <= 0:
                    break

                burst_chars = max(1, int(buf_size * budgets[idx] / max_budget))
                burst_chars = min(burst_chars, int(global_remaining))
                burst_lines: list[str] = []
                chars = 0

                while chars < burst_chars:
                    line = rdr.readline()
                    if not line:
                        rdr.close()
                        readers[idx] = None
                        active -= 1
                        break
                    s = line.strip()
                    if s:
                        burst_lines.append(s)
                        chars += len(s)

                if not burst_lines:
                    continue

                source_out[idx] += len(burst_lines)
                spent_chars[idx] += chars
                total_chars += chars
                total += len(burst_lines)
                random.shuffle(burst_lines)

                for line_text in burst_lines:
                    r = random.random()
                    if r < train_ratio:
                        train_f.write(line_text + "\n")
                        counts["train"] += 1
                    elif r < train_ratio + valid_ratio:
                        valid_f.write(line_text + "\n")
                        counts["valid"] += 1
                    else:
                        test_f.write(line_text + "\n")
                        counts["test"] += 1

            if total % (buf_size * 5) < buf_size:
                print(f"\r  Processed {total:,} lines", end="", flush=True)
    finally:
        for reader in readers:
            if reader is not None:
                reader.close()
        train_f.close()
        valid_f.close()
        test_f.close()

    print(f"\r  Processed {total:,} lines total")
    for i, (st, cnt) in enumerate(zip(stats, source_out, strict=True)):
        print(f"  {os.path.basename(st['path'])}: {cnt:,} lines ({ratios[i] * 100:.0f}% target)")

    print("\nDataset built:")
    print(f"  Train: {counts['train']:,} lines ({100 * counts['train'] / max(1, total):.1f}%)")
    print(f"  Valid: {counts['valid']:,} lines ({100 * counts['valid'] / max(1, total):.1f}%)")
    print(f"  Test:  {counts['test']:,} lines ({100 * counts['test'] / max(1, total):.1f}%)")

    return [{"source": st["path"], "rows": cnt} for st, cnt in zip(stats, source_out, strict=True)]


# ============================================================================
# 统计
# ============================================================================


def compute_stats(path: str) -> dict:
    """流式统计单文件: 行数 / 总字符 / 平均字符 / 汉字占比。"""
    rows = total_chars = cjk = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows += 1
            total_chars += len(s)
            cjk += len(_ZH_CHAR_RE.findall(s))
    return {
        "rows": rows,
        "total_chars": total_chars,
        "avg_chars": total_chars / max(rows, 1),
        "zh_ratio": cjk / max(total_chars, 1),
    }
