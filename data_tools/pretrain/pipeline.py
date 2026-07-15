"""数据预处理一键管道（标准 LLM 训练流水线）。

流程:
  step 1: 粗精确去重 (MD5)  — 先剔除完全重复，减少后续清洗/去重计算量
  step 2: 基础清洗           — 去乱码、繁转简、过滤广告/低质
  step 3: SimHash 模糊去重 / QA过滤 — 跨文本段落级去重
  step 4: 多源混合切分       — 字符加权配比 → train/valid/test
  step 5: 清理旧 token 缓存

用法:
    python data_tools/pretrain/pipeline.py --variant nano
    python data_tools/pretrain/pipeline.py --variant lite
    python data_tools/pretrain/pipeline.py --variant pro --skip_simhash
"""

import argparse
import os
import sys

from gleamlm.preprocessing.build_dataset import stream_build
from gleamlm.preprocessing.clean_text import clean_file
from gleamlm.preprocessing.dedup_text import dedup_file
from gleamlm.preprocessing.filter_qa import filter_qa
from gleamlm.utils.config import load_config


def _raw_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw.txt")


def _raw_dedup_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_raw_dedup.txt")


def _clean_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_clean.txt")


def _final_path(input_dir, name):
    return os.path.join(input_dir, f"{name}_dedup.txt")


def compute_avg_chars(filepath):
    total = 0
    lines = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            total += len(line)
            lines += 1
    return total / max(1, lines)


def main():
    parser = argparse.ArgumentParser(description="数据预处理一键管道")
    parser.add_argument(
        "--variant",
        choices=["nano", "lite", "pro"],
        required=True,
        help="模型变体（从 configs/{variant}.yaml 读取数据源配置）",
    )
    parser.add_argument("--config_dir", default="configs", help="YAML 配置目录")
    parser.add_argument("--input", default="data/raw", help="原始数据目录")
    parser.add_argument("--output", default=None, help="输出目录（默认: data/{variant}_data）")
    # 跳过控制
    parser.add_argument("--skip_exact_dedup", action="store_true")
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_simhash", action="store_true")
    # 精确去重参数
    parser.add_argument("--exact_mode", default="exact", choices=["exact", "prefix"])
    parser.add_argument("--prefix_len", type=int, default=100)
    # SimHash 参数
    parser.add_argument("--simhash_threshold", type=int, default=3)
    parser.add_argument("--simhash_window", type=int, default=1000)
    args = parser.parse_args()

    config_path = os.path.join(args.config_dir, f"{args.variant}.yaml")
    cfg = load_config(config_path, model_name=args.variant)

    sources = (
        list(cfg.data_sources._data) if hasattr(cfg.data_sources, "_data") else cfg.data_sources
    )
    if not sources:
        print("ERROR: 未找到 data_sources 配置")
        sys.exit(1)

    data_dir = args.output or os.path.join("data", f"{args.variant}_data")
    raw_dir = args.input
    print(f"Variant: {args.variant}, sources: {[s['name'] for s in sources]}")
    print(f"Input: {raw_dir}, Output: {data_dir}")

    # ──── step 1: 粗精确去重 ────
    if args.skip_exact_dedup:
        print("\n[1/5] 跳过精确去重（--skip_exact_dedup）")
    else:
        print("\n[1/5] 粗精确去重（MD5 全文去重）")
        for s in sources:
            raw = _raw_path(raw_dir, s["name"])
            deduped = _raw_dedup_path(raw_dir, s["name"])
            if not os.path.exists(raw):
                print(f"  Skip {s['name']}: {raw} not found")
                continue
            if os.path.exists(deduped) and os.path.getsize(deduped) > 0:
                print(f"  Skip {s['name']}: {deduped} exists")
                continue
            mode = "prefix" if s.get("type") == "news" else args.exact_mode
            print(f"  去重: {s['name']} (mode={mode})")
            dedup_file(raw, deduped, mode=mode, prefix_len=args.prefix_len)

    # ──── step 2: 清洗 ────
    if args.skip_clean:
        print("\n[2/5] 跳过清洗（--skip_clean）")
    else:
        print("\n[2/5] 基础清洗（去乱码、繁转简、过滤低质）")
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
                convert_zh=True,
                min_zh_ratio=0.15 if s["name"] in ("wiki", "edu") else 0.0,
                filter_ads=s["name"] == "news",
                filter_wiki_junk=s["name"] == "wiki",
            )

    # ──── step 3: SimHash 模糊去重 / QA过滤 ────
    if args.skip_simhash:
        print("\n[3/5] 跳过模糊去重（--skip_simhash）")
    else:
        print("\n[3/5] SimHash 模糊去重 / QA过滤")
        for s in sources:
            src = _clean_path(raw_dir, s["name"])
            if not os.path.exists(src):
                src = _final_path(raw_dir, s["name"])
            final = _final_path(raw_dir, s["name"])
            if not os.path.exists(src):
                print(f"  Skip {s['name']}: {src} not found")
                continue
            if os.path.exists(final) and os.path.getsize(final) > 0:
                print(f"  Skip {s['name']}: {final} exists")
                continue

            if s.get("type") == "qa":
                print(f"  QA过滤: {s['name']}")
                filter_qa(src, final)
            else:
                print(
                    f"  SimHash: {s['name']} "
                    f"(threshold={args.simhash_threshold}, window={args.simhash_window})"
                )
                dedup_file(
                    src,
                    final,
                    mode="simhash",
                    simhash_threshold=args.simhash_threshold,
                    simhash_window=args.simhash_window,
                )

    # ──── step 4: 字符加权配比 → 多源混合切分 ────
    print("\n[4/5] 多源混合切分")
    input_files: list[str] = []
    target_ratios = [s.get("ratio", 0) for s in sources]

    print("  扫描各源行均字符...")
    avg_chars_list = []
    for s in sources:
        final = _final_path(raw_dir, s["name"])
        clean = _clean_path(raw_dir, s["name"])
        fpath = final if os.path.exists(final) else clean
        if not os.path.exists(fpath):
            print(f"    WARNING: {s['name']} 源文件不存在，跳过")
            avg_chars_list.append(0)
            input_files.append("")
            continue
        avg_c = compute_avg_chars(fpath)
        avg_chars_list.append(avg_c)
        input_files.append(fpath)
        print(f"    {s['name']}: ~{avg_c:.0f} 字/行")

    effective = []
    for i in range(len(sources)):
        if avg_chars_list[i] > 0:
            effective.append(target_ratios[i] / avg_chars_list[i])
        else:
            effective.append(0)
    total = sum(effective)
    if total > 0:
        effective = [e / total for e in effective]

    valid_files: list[str] = []
    valid_ratios: list[float] = []
    for i, s in enumerate(sources):
        if input_files[i] and effective[i] > 0:
            valid_files.append(input_files[i])
            valid_ratios.append(effective[i])
            print(
                f"  {s['name']}: 目标{target_ratios[i] * 100:.0f}% 字符 → "
                f"行数配比 {effective[i] * 100:.1f}%"
            )
        elif input_files[i]:
            valid_files.append(input_files[i])
            valid_ratios.append(0.0001)

    if len(valid_files) < 1:
        print("ERROR: 有效数据源为 0")
        sys.exit(1)

    stream_build(
        input_paths=valid_files,
        output_dir=data_dir,
        ratios=valid_ratios if valid_ratios else None,
        train_ratio=0.9,
        valid_ratio=0.05,
    )

    # ──── step 5: 清理旧 token 缓存 ────
    print("\n[5/5] 清理旧 token 缓存")
    for split in ("train", "valid", "test"):
        cache = os.path.join(data_dir, f"{split}_ids.npy")
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  Removed: {cache}")
    print("  完成")


if __name__ == "__main__":
    main()
