"""数据集混合切分（变体相关，从已处理的 *_dedup.txt 按配比构建 train/valid/test）。

用法:
    python data_tools/pretrain/build.py --variant nano
    python data_tools/pretrain/build.py --variant lite
    python data_tools/pretrain/build.py --variant pro
"""

import argparse
import os
import sys

from gleamlm.preprocessing.mix_split import stream_split
from gleamlm.utils.config import load_config


def compute_avg_chars(filepath):
    total = 0
    lines = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            total += len(line)
            lines += 1
    return total / max(1, lines)


def _resolve_path(raw_dir: str, name: str) -> str:
    for suffix in ("_dedup.txt", "_trunc.txt", "_clean.txt"):
        path = os.path.join(raw_dir, f"{name}{suffix}")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return os.path.join(raw_dir, f"{name}_dedup.txt")


def main():
    parser = argparse.ArgumentParser(description="数据集混合切分")
    parser.add_argument(
        "--variant",
        choices=["nano", "lite", "pro"],
        required=True,
        help="模型变体（从 configs/{variant}.yaml 读取配比）",
    )
    parser.add_argument("--config_dir", default="configs", help="YAML 配置目录")
    parser.add_argument("--input", default="data/raw", help="已处理数据目录（*_dedup.txt）")
    parser.add_argument("--output", default=None, help="输出目录（默认: data/{variant}/pretrain）")
    args = parser.parse_args()

    config_path = os.path.join(args.config_dir, f"{args.variant}.yaml")
    cfg = load_config(config_path, model_name=args.variant)

    sources = (
        list(cfg.data_sources._data) if hasattr(cfg.data_sources, "_data") else cfg.data_sources
    )
    if not sources:
        print("ERROR: 未找到 data_sources 配置")
        sys.exit(1)

    data_dir = args.output or os.path.join("data", args.variant, "pretrain")
    raw_dir = args.input
    print(f"Variant: {args.variant}, sources: {[s['name'] for s in sources]}")
    print(f"Input: {raw_dir}, Output: {data_dir}")

    # ──── 字符加权配比 → 多源混合切分 ────
    print("\n字符加权配比 → 混合切分")
    input_files: list[str] = []
    target_ratios = [s.get("ratio", 0) for s in sources]

    print("  扫描各源行均字符...")
    avg_chars_list = []
    for s in sources:
        fpath = _resolve_path(raw_dir, s["name"])
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

    stream_split(
        input_paths=valid_files,
        output_dir=data_dir,
        ratios=valid_ratios if valid_ratios else None,
        train_ratio=0.9,
        valid_ratio=0.05,
    )

    # ──── 清理旧 token 缓存 ────
    print("\n清理旧 token 缓存")
    for split in ("train", "valid", "test"):
        cache = os.path.join(data_dir, f"{split}_ids.npy")
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  Removed: {cache}")
    print("  完成")


if __name__ == "__main__":
    main()
