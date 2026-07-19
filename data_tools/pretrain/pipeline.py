"""数据预处理管道（与变体无关，所有变体共用）。

流程:
  step 1: 粗精确去重 (MD5)  — 先剔除完全重复，减少后续清洗/去重计算量
  step 2: 基础清洗           — 去乱码、繁转简、过滤广告/低质（全源 min_zh_ratio=0.15）
  step 3: SimHash 逐源去重   — 各源独立全局查重
  step 4: SimHash 跨源去重   — 所有源合并指纹，跨源剔除重复

输出 → data/raw/{name}_dedup.txt（供 build.py 混合切分使用）

用法:
    python data_tools/pretrain/pipeline.py
    python data_tools/pretrain/pipeline.py --skip_simhash
    python data_tools/pretrain/pipeline.py --sources wiki baike
"""

import argparse
import os
import pickle

from gleamlm.preprocess import clean_file, dedup_file, filter_qa, normalize, simhash

SOURCES = [
    {"name": "edu", "type": "text"},
    {"name": "news", "type": "news"},
    {"name": "wiki", "type": "text"},
    {"name": "baike", "type": "text"},
    {"name": "qa", "type": "qa"},
]

MIN_Zh_RATIO = 0.15


def _raw_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw.txt")


def _raw_dedup_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw_dedup.txt")


def _clean_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_clean.txt")


def _final_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_dedup.txt")


def _fps_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_dedup.fps")


def _save_fingerprints(filepath: str, fps: set[int]) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(fps, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_fingerprints(filepath: str) -> set[int]:
    fps_file = filepath.replace("_dedup.txt", "_dedup.fps")
    if os.path.exists(fps_file) and os.path.getsize(fps_file) > 0:
        try:
            with open(fps_file, "rb") as f:
                fps = pickle.load(f)
            print(
                f"  Loaded {len(fps):,} fingerprints from {os.path.basename(fps_file)}", flush=True
            )
            return fps
        except Exception:
            print("  WARNING: corrupted fingerprint cache, regenerating...", flush=True)
            os.remove(fps_file)

    fps: set[int] = set()
    size_mb = os.path.getsize(filepath) / 1e6
    print(
        f"  Loading fingerprints from {os.path.basename(filepath)} ({size_mb:.0f} MB)...",
        flush=True,
    )
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i % 200000 == 0:
                print(f"    {i:,} lines scanned, {len(fps):,} fingerprints", flush=True)
            text = normalize(line.strip())
            if text:
                fps.add(simhash(text))
    print(f"    Done: {len(fps):,} fingerprints loaded", flush=True)
    _save_fingerprints(fps_file, fps)
    return fps


def main():
    parser = argparse.ArgumentParser(
        description="数据预处理管道（去重→清洗→SimHash 逐源+跨源去重）"
    )
    parser.add_argument("--input", default="data/raw", help="原始数据目录")
    parser.add_argument("--skip_exact_dedup", action="store_true")
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_simhash", action="store_true")
    parser.add_argument(
        "--cross_dedup", action="store_true", help="启用跨源 SimHash 全局去重（默认跳过）"
    )
    parser.add_argument("--exact_mode", default="exact", choices=["exact", "prefix"])
    parser.add_argument("--prefix_len", type=int, default=100)
    parser.add_argument("--simhash_threshold", type=int, default=3)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="只处理指定的源 (edu/news/wiki/baike/qa)，默认全部",
    )
    args = parser.parse_args()

    raw_dir = args.input
    threshold = args.simhash_threshold
    sources = SOURCES
    if args.sources:
        valid = {s["name"] for s in SOURCES}
        unknown = set(args.sources) - valid
        if unknown:
            print(f"ERROR: unknown sources: {unknown}")
            return
        sources = [s for s in SOURCES if s["name"] in args.sources]
    names = [s["name"] for s in sources]
    print(f"Sources: {names}")

    # ──── step 1: 粗精确去重 ────
    if args.skip_exact_dedup:
        print("\n[1/4] 跳过精确去重（--skip_exact_dedup）")
    else:
        print("\n[1/4] 粗精确去重（MD5 全文去重）")
        for s in sources:
            raw = _raw_path(raw_dir, s["name"])
            deduped = _raw_dedup_path(raw_dir, s["name"])
            if not os.path.exists(raw):
                print(f"  Skip {s['name']}: {raw} not found")
                continue
            if os.path.exists(deduped) and os.path.getsize(deduped) > 0:
                print(f"  Skip {s['name']}: {deduped} exists")
                continue
            mode = "prefix" if s["type"] == "news" else args.exact_mode
            print(f"  去重: {s['name']} (mode={mode})")
            dedup_file(raw, deduped, mode=mode, prefix_len=args.prefix_len)

    # ──── step 2: 清洗 ────
    if args.skip_clean:
        print("\n[2/4] 跳过清洗（--skip_clean）")
    else:
        print(f"\n[2/4] 基础清洗（min_zh_ratio={MIN_Zh_RATIO}, 去乱码、繁转简、过滤低质）")
        for s in sources:
            src = _raw_dedup_path(raw_dir, s["name"])
            if not os.path.exists(src):
                src = _raw_path(raw_dir, s["name"])
            clean = _clean_path(raw_dir, s["name"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: no source found")
                continue
            if os.path.exists(clean) and os.path.getsize(clean) > 0:
                print(f"  Skip {s['name']}: {clean} exists")
                continue
            print(f"  Cleaning: {s['name']}")
            clean_file(
                src,
                clean,
                min_len=30,
                max_len=3000,
                convert_zh=(s["name"] != "edu"),
                min_zh_ratio=MIN_Zh_RATIO,
                filter_ads=s["name"] == "news",
                filter_wiki_junk=s["name"] == "wiki",
            )

    # ──── step 3: SimHash 逐源去重 + QA过滤 ────
    source_fingerprints: dict[str, set[int]] = {}
    if args.skip_simhash:
        print("\n[3/4] 跳过 SimHash 去重（--skip_simhash）")
    else:
        print("\n[3/4] SimHash 逐源去重 / QA过滤")
        for s in sources:
            src = _clean_path(raw_dir, s["name"])
            if not os.path.exists(src):
                src = _final_path(raw_dir, s["name"])
            final = _final_path(raw_dir, s["name"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: {src} not found")
                continue
            if os.path.exists(final) and os.path.getsize(final) > 0:
                if s["type"] != "qa":
                    fps = _load_fingerprints(final)
                    source_fingerprints[s["name"]] = fps
                    print(f"  Skip {s['name']}: {final} exists ({len(fps):,} fingerprints loaded)")
                else:
                    source_fingerprints[s["name"]] = set()
                    print(f"  Skip {s['name']}: {final} exists")
                continue

            if s["type"] == "qa":
                print(f"  QA过滤: {s['name']}")
                filter_qa(src, final)
                source_fingerprints[s["name"]] = set()
            else:
                print(f"  SimHash: {s['name']} (threshold={threshold})")
                fps = dedup_file(
                    src,
                    final,
                    mode="simhash",
                    simhash_threshold=threshold,
                )
                source_fingerprints[s["name"]] = fps
                _save_fingerprints(_fps_path(raw_dir, s["name"]), fps)

        processed = sum(1 for v in source_fingerprints.values() if v)
        total_fps = sum(len(v) for v in source_fingerprints.values())
        print(f"\n  Collected fingerprints: {total_fps:,} across {processed} sources processed")

    # ──── step 4: 跨源 SimHash 全局去重 ────
    if args.skip_simhash:
        print("\n[4/4] 跳过跨源去重（--skip_simhash）")
    elif not args.cross_dedup:
        print("\n[4/4] 跳过跨源去重（默认跳过，启用: --cross_dedup）")
    else:
        print("\n[4/4] 跨源 SimHash 全局去重")
        # 冻结 Step 3 指纹快照，所有源基于同一基准去重
        snapshot: dict[str, set[int]] = {
            name: fps_set.copy() for name, fps_set in source_fingerprints.items()
        }
        for s in sources:
            final = _final_path(raw_dir, s["name"])
            if not os.path.exists(final):
                continue
            if not source_fingerprints.get(s["name"]):
                continue
            tmp = final + ".tmp"
            # 排除自身指纹，只和其他源比对（基于快照）
            other_fps: set[int] = set()
            for name, fps_set in snapshot.items():
                if name != s["name"]:
                    other_fps.update(fps_set)
            print(
                f"  Cross-dedup: {s['name']} (against {len(other_fps):,} fingerprints from other sources)"
            )
            try:
                returned = dedup_file(
                    final,
                    tmp,
                    mode="simhash",
                    simhash_threshold=threshold,
                    existing_fingerprints=other_fps,
                )
                os.replace(tmp, final)
                source_fingerprints[s["name"]] = returned - other_fps
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    print("  完成")


if __name__ == "__main__":
    main()
