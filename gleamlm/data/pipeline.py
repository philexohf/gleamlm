"""预训练数据管线编排 — 6 阶段标准管线：粗去重 → 清洗 → 质量 → 细去重 → 切分 → 打包。

所有变体共用。每阶段产物落盘、存在且非空即跳过（断点续跑）。

用法:
    from gleamlm.data.pipeline import run_pipeline

    run_pipeline(
        sources=["wiki", "baike"],
        output_prefix="data/processed",
        max_chars=2_600_000_000,
        ratios="wiki:0.4,baike:0.3,edu:0.3",
    )

CLI:
    python data_tools/pretrain/run_pipeline.py --sources wiki baike
"""

from __future__ import annotations

import os
import pickle
import random
import shutil

from gleamlm.data.pack import (
    build_indexed_dataset,
    get_tokenizer,
    load_text,
)
from gleamlm.data.preprocess import (
    clean_file,
    compute_stats,
    dedup_file,
    minhash_dedup_file,
    score_quality_file,
    stream_split,
)
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

SOURCES: list[dict] = [
    {"name": "edu", "type": "text", "lang": "zh"},
    {"name": "news", "type": "news", "lang": "zh"},
    {"name": "wiki", "type": "text", "lang": "zh"},
    {"name": "baike", "type": "text", "lang": "zh"},
    # ── 英文源（需要时取消注释） ──
    # {"name": "en_wiki", "type": "text", "lang": "en"},
    # {"name": "en_books", "type": "text", "lang": "en"},
    # {"name": "en_web", "type": "text", "lang": "en"},
]

MIN_ZH_RATIO = 0.15
MIN_EN_RATIO = 0.15
MIN_LEN = 30
MAX_LEN = 3000
# SimHash 去重的行数上限：超大源（百万级）SimHash 桶内比较代价高，
# 而高质量源重复率极低 → 超过该阈值自动降级为精确去重（exact）
_SIMHASH_MAX_ROWS = 500_000


# ──── 路径约定（断点续跑的关键：固定产物路径） ──────────────────────────


def _raw_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw.txt")


def _raw_dedup_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw_dedup.txt")


def _clean_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_clean.txt")


def _quality_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_quality.txt")


def _final_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_dedup.txt")


def _fps_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_dedup.fps")


def _rows(path: str) -> int:
    """快速估算行数（多段采样平均字节率 → 按文件大小外推）。

    断点续跑时对已存在的 41-48G 产物反复全量数行会拖慢整个管线；
    行数仅用于打印与 exact/simhash 选择判断（阈值 50 万，各源量级远超，
    估算误差不影响判定）。如需精确行数用 compute_stats(path)["rows"]。
    """
    if not os.path.exists(path):
        return 0
    size = os.path.getsize(path)
    if size == 0:
        return 0
    try:
        total_lines, _ = _estimate_total(path, _estimate_avg_chars(path, n=1000))
        return total_lines
    except Exception:
        return compute_stats(path)["rows"]


def _save_fingerprints(filepath: str, fps: set[int]) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(fps, f, protocol=pickle.HIGHEST_PROTOCOL)


def _pick_first(*paths: str) -> str | None:
    """断点续跑：返回第一个存在且非空的文件路径。"""
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


# ──── Bernoulli 采样辅助（用于 step 5 字符预算控制） ────────────────────


def _estimate_avg_chars(filepath: str, n: int = 1000) -> float:
    """多段采样估算平均每行字符数（头/中/尾各 n 行）。"""
    size = os.path.getsize(filepath)
    if size == 0:
        return 0.0
    total, lines = 0, 0
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for seg in range(3):
            if seg > 0:
                f.seek(int(size * seg / 3))
                f.readline()  # 丢弃半行（errors=replace 容忍多字节截断）
            count = 0
            for line in f:
                total += len(line)
                lines += 1
                count += 1
                if count >= n:
                    break
    return total / max(1, lines)


def _estimate_total(filepath: str, avg_chars: float, n_segments: int = 3) -> tuple[int, int]:
    """推估文件总行数和总字符数（多段采样字节率，平均后按大小外推）。

    只读文件头/中/尾 n_segments 段各 ~2000 行，用平均字节率外推全量，
    O(1) 内存、毫秒级。断点续跑时避免对 40G+ 产物全量扫描。
    """
    size = os.path.getsize(filepath)
    byte_total = char_total = 0
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for seg in range(n_segments):
            if seg > 0:
                f.seek(int(size * seg / n_segments))
                f.readline()  # 丢弃半行（errors=replace 容忍多字节截断）
            count = 0
            for line in f:
                byte_total += len(line.encode("utf-8"))
                char_total += len(line)
                count += 1
                if count >= 2000:
                    break
    bytes_per_char = byte_total / max(1, char_total)
    total_chars = int(size / bytes_per_char)
    total_lines = int(total_chars / max(1.0, avg_chars))
    return total_lines, total_chars


def _bernoulli_sample(
    sources: list[dict],
    input_files: list[str],
    target_ratios: list[float],
    avg_chars_list: list[float],
    max_chars: int,
    tmp_dir: str,
    train_ratio: float = 0.9,
) -> list[str]:
    """Bernoulli 无偏采样：每源按概率逐行取舍，保证配比精准。

    max_chars 语义 = 训练集字符预算（config 的 max_train_chars）。
    切分按 train/valid/test 比例进行，因此采样总量需放大 1/train_ratio，
    使切分后 train 恰好 ≈ max_chars（valid/test 合计占剩余部分）。
    """
    random.seed(42)
    os.makedirs(tmp_dir, exist_ok=True)

    sampled_files: list[str] = []
    probs: list[float] = []

    for i, s in enumerate(sources):
        fpath = input_files[i]
        if not fpath:
            sampled_files.append("")
            probs.append(0)
            continue

        _, total_chars = _estimate_total(fpath, avg_chars_list[i])
        needed = max_chars / train_ratio * target_ratios[i]
        prob = min(needed / max(total_chars, 1), 1.0)

        if prob >= 0.95:
            sampled_files.append(fpath)
            probs.append(1.0)
        else:
            out_path = os.path.join(tmp_dir, f"{s['name']}.txt")
            sampled_files.append(out_path)
            probs.append(prob)

    from_ratio = [
        f"{probs[i] * 100:.0f}% ({s['name']})" for i, s in enumerate(sources) if probs[i] < 0.95
    ]
    if from_ratio:
        print(f"  Bernoulli 采样率: {', '.join(from_ratio)}")
    else:
        print("  所有源全量使用（采样率 ≥ 95%）")
        return sampled_files

    for i, s in enumerate(sources):
        if not input_files[i] or probs[i] >= 0.95:
            continue
        in_count, out_count = 0, 0
        with (
            open(input_files[i], encoding="utf-8") as fin,
            open(sampled_files[i], "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                in_count += 1
                if random.random() < probs[i]:
                    fout.write(line)
                    out_count += 1
                if in_count % 500000 == 0:
                    rate = 100 * out_count / max(1, in_count)
                    print(f"    {s['name']}: {in_count:,} → {out_count:,} ({rate:.0f}%)", flush=True)
        actual = 100 * out_count / max(1, in_count)
        print(f"    {s['name']}: 完成 {out_count:,} 行 ({actual:.1f}%)")

    return sampled_files


def _parse_ratios(text: str | None, names: list[str]) -> list[float]:
    """解析 "wiki:0.6,news:0.4" → 与 names 同序的占比列表；None 时等分。

    占比必须覆盖全部源且和为 1；只传部分源会被明确拒绝（不静默归一化，
    避免掩盖配比错误——用户应传全各源占比）。
    """
    if text is None:
        return [1.0 / len(names)] * len(names)
    table: dict[str, float] = {}
    for item in text.split(","):
        k, _, v = item.partition(":")
        table[k.strip()] = float(v.strip())
    total = sum(table.values())
    if abs(total - 1.0) > 1e-6:
        missing = [n for n in names if n not in table]
        hint = f"，缺失源: {missing}" if missing else ""
        raise ValueError(f"--ratios 占比必须覆盖全部源且和为 1，当前 {total}{hint}")
    unknown = set(table) - set(names)
    if unknown:
        raise ValueError(f"unknown source in --ratios: {unknown} (valid: {names})")
    return [table.get(n, 0.0) for n in names]


# ──── 主编排 — 6 阶段端到端管线 ────────────────────────────────────────────


def run_pipeline(
    input_dir: str = "data/raw",
    sources: list[str] | None = None,
    output_prefix: str = "data/processed",
    # 跳过标志
    skip_exact_dedup: bool = False,
    skip_clean: bool = False,
    skip_quality: bool = False,
    skip_dedup: bool = False,
    # 清洗参数
    min_zh_ratio: float = MIN_ZH_RATIO,
    min_en_ratio: float = MIN_EN_RATIO,
    # 质量参数
    min_quality: float = 0.30,
    # 去重参数
    use_minhash: bool = False,
    minhash_threshold: float = 0.8,
    simhash_threshold: int = 3,
    prefix_len: int = 100,
    # 切分配比
    ratios: str | None = None,
    max_chars: int | None = None,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    seed: int = 42,
    # 打包
    tokenizer: str = "bbpe",
    tokenizer_path: str | None = None,
    workers: int = 4,
    skip_verify: bool = False,
) -> None:
    """6 阶段标准预处理管线。

    每阶段产物落盘后存在即跳过（断点续跑），因此可以多次运行同一命令。

    参数说明见 CLI: python data_tools/pretrain/run_pipeline.py --help
    """
    # ── 解析 sources ──
    all_sources = SOURCES
    if sources:
        valid = {s["name"] for s in SOURCES}
        unknown = set(sources) - valid
        if unknown:
            raise ValueError(f"Unknown sources: {unknown}")
        all_sources = [s for s in SOURCES if s["name"] in sources]
    names = [s["name"] for s in all_sources]
    print(f"Sources: {names}")

    # ──── step 1: 粗精确去重 ────────────────────────────────────────────────────
    if skip_exact_dedup:
        print("\n[1/6] 跳过粗去重")
    else:
        print("\n[1/6] 粗去重（MD5 全文 / news 前缀），先剔除完全重复以减少后续计算量")
        for s in all_sources:
            raw = _raw_path(input_dir, s["name"])
            deduped = _raw_dedup_path(input_dir, s["name"])
            if not os.path.exists(raw):
                print(f"  Skip {s['name']}: {raw} not found")
                continue
            if os.path.exists(deduped) and os.path.getsize(deduped) > 0:
                print(f"  Skip {s['name']}: {deduped} exists ({_rows(deduped):,} lines)")
                continue
            mode = "prefix" if s["type"] == "news" else "exact"
            print(f"  去重: {s['name']} (mode={mode})")
            dedup_file(raw, deduped, mode=mode, prefix_len=prefix_len)

    # ──── step 2: 基础清洗（语言感知） ──────────────────────────────────────────
    if skip_clean:
        print("\n[2/6] 跳过清洗")
    else:
        zh_srcs = [s["name"] for s in all_sources if s.get("lang") == "zh"]
        en_srcs = [s["name"] for s in all_sources if s.get("lang") == "en"]
        lang_info = []
        if zh_srcs:
            lang_info.append(f"zh: min_zh_ratio={min_zh_ratio}")
        if en_srcs:
            lang_info.append(f"en: min_en_ratio={min_en_ratio}")
        print(
            f"\n[2/6] 基础清洗（min_len={MIN_LEN}, max_len={MAX_LEN}, "
            f"{', '.join(lang_info)}，news 滤广告 / wiki 滤垃圾）"
        )
        for s in all_sources:
            src = _pick_first(_raw_dedup_path(input_dir, s["name"]),
                              _raw_path(input_dir, s["name"]))
            clean = _clean_path(input_dir, s["name"])
            if src is None:
                print(f"  Skip {s['name']}: no source found")
                continue
            if os.path.exists(clean) and os.path.getsize(clean) > 0:
                print(f"  Skip {s['name']}: {clean} exists ({_rows(clean):,} lines)")
                continue

            is_zh = s.get("lang") == "zh"
            is_en = s.get("lang") == "en"
            print(f"  Cleaning: {s['name']} (lang={s.get('lang', 'unknown')})")
            clean_file(
                src,
                clean,
                min_len=MIN_LEN,
                max_len=MAX_LEN,
                min_zh_ratio=min_zh_ratio if is_zh else 0.0,
                min_en_ratio=min_en_ratio if is_en else 0.0,
                filter_ads=s["name"] == "news" and is_zh,
                filter_wiki_junk=s["name"] == "wiki" and is_zh,
            )

    # ──── step 3: 质量过滤（Gopher 5 规则） ─────────────────────────────────────
    if skip_quality:
        print("\n[3/6] 跳过质量过滤")
    else:
        print(f"\n[3/6] 质量过滤（Gopher 5 规则, min_score={min_quality}）")
        for s in all_sources:
            src = _pick_first(
                _clean_path(input_dir, s["name"]),
                _raw_dedup_path(input_dir, s["name"]),
                _raw_path(input_dir, s["name"]),
            )
            quality = _quality_path(input_dir, s["name"])
            if src is None:
                print(f"  Skip {s['name']}: no source found")
                continue
            if os.path.exists(quality) and os.path.getsize(quality) > 0:
                print(f"  Skip {s['name']}: {quality} exists ({_rows(quality):,} lines)")
                continue
            print(f"  Quality: {s['name']}")
            score_quality_file(src, quality, min_score=min_quality)

    # ──── step 4: 细去重（SimHash / MinHash） ──────────────────────────────────
    if skip_dedup:
        print("\n[4/6] 跳过细去重")
    else:
        mode = "minhash" if use_minhash else "simhash"
        extra = (
            f", jaccard>={minhash_threshold}"
            if use_minhash
            else f", hamming<={simhash_threshold}"
        )
        print(f"\n[4/6] 细去重（mode={mode}{extra}）")
        for s in all_sources:
            src = _pick_first(
                _quality_path(input_dir, s["name"]),
                _clean_path(input_dir, s["name"]),
                _raw_dedup_path(input_dir, s["name"]),
                _raw_path(input_dir, s["name"]),
            )
            final = _final_path(input_dir, s["name"])
            if src is None:
                print(f"  Skip {s['name']}: no source found")
                continue
            if os.path.exists(final) and os.path.getsize(final) > 0:
                print(f"  Skip {s['name']}: {final} exists ({_rows(final):,} lines)")
                continue
            if s["type"] == "qa":
                print(f"  QA过滤: {s['name']}")
                from gleamlm.data.preprocess import filter_qa

                filter_qa(src, final)
            else:
                src_rows = _rows(src)
                # 超大源（如 edu 百万级）SimHash 是 O(n·桶内比较)，代价高；
                # 而这类源通常是质量过滤后的高质数据（重复率极低），
                # 精确去重足够且快几个数量级 —— 大源自动降级 exact
                if src_rows > _SIMHASH_MAX_ROWS:
                    print(f"  Exact: {s['name']} ({src_rows:,} 行超大源，降级精确去重)")
                    dedup_file(src, final, mode="exact")
                elif use_minhash:
                    print(f"  MinHash: {s['name']} (threshold={minhash_threshold})")
                    minhash_dedup_file(src, final, threshold=minhash_threshold)
                else:
                    print(f"  SimHash: {s['name']} (threshold={simhash_threshold})")
                    fps = dedup_file(src, final, mode="simhash",
                                     simhash_threshold=simhash_threshold)
                    _save_fingerprints(_fps_path(input_dir, s["name"]), fps)

    # ──── step 5: 配比切分（字符占比 + Bernoulli 采样） ──────────────────────
    # 输入取去重产物；--skip-dedup 时回退到质量/清洗产物（_pick_first）
    finals = []
    for s in all_sources:
        p = _pick_first(
            _final_path(input_dir, s["name"]),
            _quality_path(input_dir, s["name"]),
            _clean_path(input_dir, s["name"]),
            _raw_dedup_path(input_dir, s["name"]),
        )
        if p:
            finals.append((s["name"], p))
    if not finals:
        print("\n[5/6] 错误: 没有可用的 {name}_dedup.txt 产物")
        return
    final_names = [n for n, _ in finals]
    final_paths = [p for _, p in finals]
    final_ratios = _parse_ratios(ratios, final_names)

    print("\n[5/6] 配比切分（字符占比 → train/valid/test）")

    # Bernoulli 采样（当 max_chars 小于总数据量时）
    sampled_paths = final_paths
    if max_chars:
        print(f"  字符预算: {max_chars / 1e9:.2f}B")
        avg_chars_list = [_estimate_avg_chars(fp) for fp in final_paths]
        tmp_dir = os.path.join(os.path.dirname(output_prefix) or ".", ".bernoulli_samples")
        combined_source_dicts = [s for s in all_sources if s["name"] in final_names]
        sampled_paths = _bernoulli_sample(
            combined_source_dicts, final_paths, final_ratios, avg_chars_list, max_chars, tmp_dir,
            train_ratio=train_ratio,
        )
        effective = [(p, r) for p, r in zip(sampled_paths, final_ratios) if p]
        if not effective:
            print("ERROR: Bernoulli 采样后无有效数据")
            return
        sampled_paths = [p for p, _ in effective]
        final_ratios = [r for _, r in effective]

    out_dir = output_prefix or "."
    stream_split(
        sampled_paths,
        out_dir,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        ratios=final_ratios,
        max_chars=None,  # Bernoulli 已控制
        seed=seed,
        output_prefix=None,  # 目录式命名: {out_dir}/{split}.txt
    )

    # 清理 Bernoulli 临时文件（与创建位置一致: 上级目录下）
    tmp_dir = os.path.join(os.path.dirname(output_prefix) or ".", ".bernoulli_samples")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ──── step 6: 打包 .bin/.idx ─────────────────────────────────────────────
    print("\n[6/6] 打包 tokenize → .bin/.idx")
    tok_path = tokenizer_path or DEFAULT_TOKENIZER_PATH
    tok = get_tokenizer(tokenizer, tok_path)
    print(f"tokenizer: {tokenizer} ({tok_path})")
    for split in ("train", "valid", "test"):
        txt = os.path.join(output_prefix, f"{split}.txt")
        if not os.path.exists(txt) or os.path.getsize(txt) == 0:
            print(f"  Skip {split}: {txt} not found or empty")
            continue
        bin_prefix = os.path.join(output_prefix, split)
        if os.path.exists(bin_prefix + ".bin") and os.path.exists(bin_prefix + ".idx"):
            print(f"  Skip {split}: {bin_prefix}.bin/.idx exists")
            continue
        print(f"  [{split}] 读取 {txt}")
        docs = load_text(txt)
        print(f"  [{split}] tokenize ({len(docs):,} docs, workers={workers})")
        build_indexed_dataset(
            bin_prefix,
            docs,
            tok,
            workers=workers,
            vocab_size=tok.vocab_size,
            skip_verify=skip_verify,
        )

    # ──── 汇总报告 ────────────────────────────────────────────────────────────
    print("\n== 汇总报告 ==")
    for name in final_names:
        chain = []
        for p_name in ("raw", "raw_dedup", "clean", "quality", "dedup"):
            path_map = {
                "raw": _raw_path(input_dir, name),
                "raw_dedup": _raw_dedup_path(input_dir, name),
                "clean": _clean_path(input_dir, name),
                "quality": _quality_path(input_dir, name),
                "dedup": _final_path(input_dir, name),
            }
            p = path_map[p_name]
            if os.path.exists(p) and os.path.getsize(p) > 0:
                chain.append((p_name, p))
        if chain:
            labels = [c[0] for c in chain]
            nums = [f"{_rows(c[1]):,}" for c in chain]
            print(f"  {name}: " + " → ".join(f"{l}={n}" for l, n in zip(labels, nums)))
    print("完成。训练读取: gleamlm/data/dataset.py → {output_prefix}/{split}.bin/.idx")
