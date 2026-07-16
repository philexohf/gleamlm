"""数据集混合切分（变体相关，从已处理的源文件按配比构建 train/valid/test）。

支持 --max_chars 做无偏 Bernoulli 采样，避免小模型数据量超标。

用法:
    python data_tools/pretrain/build.py --variant nano --max_chars 2600000000
    python data_tools/pretrain/build.py --variant lite
    python data_tools/pretrain/build.py --variant pro
"""

import argparse
import os
import random
import shutil
import sys

from gleamlm.preprocessing.mix_split import stream_split
from gleamlm.utils.config import load_config

SAMPLE_LINES = 100_000


def estimate_avg_chars(filepath: str, n: int = SAMPLE_LINES) -> float:
    total, lines = 0, 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            total += len(line)
            lines += 1
            if lines >= n:
                break
    return total / max(1, lines)


def _estimate_total(filepath: str, avg_chars: float) -> tuple[int, int]:
    """推估文件总行数和总字符数（采样首 1000 行字节率）."""
    file_bytes = os.path.getsize(filepath)
    byte_total, char_total = 0, 0
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            byte_total += len(line.encode("utf-8"))
            char_total += len(line)
            if i >= 999:
                break
    bytes_per_char = byte_total / max(1, char_total)
    total_chars = int(file_bytes / bytes_per_char)
    total_lines = int(total_chars / avg_chars)
    return total_lines, total_chars


def _resolve_path(raw_dir: str, name: str) -> str:
    for suffix in ("_dedup.txt", "_trunc.txt", "_clean.txt"):
        path = os.path.join(raw_dir, f"{name}{suffix}")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return os.path.join(raw_dir, f"{name}_dedup.txt")


def _bernoulli_sample(
    sources: list[dict],
    input_files: list[str],
    target_ratios: list[float],
    avg_chars_list: list[float],
    max_chars: int,
    data_dir: str,
) -> list[str]:
    """Bernoulli 无偏采样：每源按概率逐行取舍，保证配比最精."""
    random.seed(42)
    tmp_dir = os.path.join(data_dir, ".samples")
    os.makedirs(tmp_dir, exist_ok=True)

    print("  估算每源采样率...")
    sampled_files: list[str] = []
    probs: list[float] = []
    for i, s in enumerate(sources):
        fpath = input_files[i]
        if not fpath:
            sampled_files.append("")
            probs.append(0)
            continue

        _, total_chars = _estimate_total(fpath, avg_chars_list[i])
        needed = max_chars * target_ratios[i]
        prob = needed / total_chars

        if prob >= 0.95:
            sampled_files.append(fpath)
            probs.append(1.0)
            print(f"    {s['name']}: ≥95% → 全量")
        else:
            out_path = os.path.join(tmp_dir, f"{s['name']}.txt")
            sampled_files.append(out_path)
            probs.append(prob)
            print(f"    {s['name']}: {prob*100:.0f}% sampling rate")

    print("  Bernoulli 采样中...")
    for i, s in enumerate(sources):
        if not input_files[i] or probs[i] >= 0.95:
            continue
        in_count, out_count = 0, 0
        with open(input_files[i], encoding="utf-8") as fin, open(
            sampled_files[i], "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                in_count += 1
                if random.random() < probs[i]:
                    fout.write(line)
                    out_count += 1
                if in_count % 500000 == 0:
                    rate = 100 * out_count / max(1, in_count)
                    print(f"    {s['name']}: {in_count:,} → {out_count:,} ({rate:.0f}%)", flush=True)
        actual = 100 * out_count / max(1, in_count)
        print(f"    {s['name']}: finished {out_count:,} lines ({actual:.1f}%)")

    return sampled_files


def main() -> None:
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
    parser.add_argument("--max_chars", type=int, default=None, help="混合数据总字符数上限（Bernoulli 采样后混合）")
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

    # ──── 字符加权配比 → 行数配比 ────
    print("\n字符加权配比 → 行数配比")
    input_files: list[str] = []
    target_ratios = [s.get("ratio", 0) for s in sources]

    print("  快速估算各源行均字符...")
    avg_chars_list = []
    for s in sources:
        fpath = _resolve_path(raw_dir, s["name"])
        if not os.path.exists(fpath):
            print(f"    WARNING: {s['name']} 源文件不存在，跳过")
            avg_chars_list.append(0)
            input_files.append("")
            continue
        avg_c = estimate_avg_chars(fpath)
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

    # ──── Bernoulli 采样 ────
    sampled_files: list[str]
    if args.max_chars:
        print(f"\nBernoulli 采样 (max {args.max_chars / 1e9:.2f}B chars)...")
        sampled_files = _bernoulli_sample(
            sources, input_files, target_ratios, avg_chars_list, args.max_chars, data_dir
        )
        # Filter out empty slots + rebuild valid ratios (ratios unchanged, using same order)
        final_files = []
        final_ratios = []
        valid_idx = 0
        for f in sampled_files:
            if f:
                final_files.append(f)
                final_ratios.append(valid_ratios[valid_idx] if valid_idx < len(valid_ratios) else 0)
                valid_idx += 1
        if not final_files:
            print("ERROR: 采样后无有效数据")
            sys.exit(1)
        valid_files = final_files
        valid_ratios = final_ratios
    else:
        print("\n跳过采样（--max_chars 未设）：使用全量数据")

    # ──── 清理旧输出 ────
    os.makedirs(data_dir, exist_ok=True)
    for f in os.listdir(data_dir):
        if f.endswith(".txt") or f.endswith(".npy"):
            os.remove(os.path.join(data_dir, f))

    # ──── 多源混合切分 ────
    print()
    stream_split(
        input_paths=valid_files,
        output_dir=data_dir,
        ratios=valid_ratios if valid_ratios else None,
        train_ratio=0.9,
        valid_ratio=0.05,
    )

    # ──── 清理采样临时文件 ────
    tmp_dir = os.path.join(data_dir, ".samples")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n清理采样临时文件: {tmp_dir}")

    print("  完成")


if __name__ == "__main__":
    main()
