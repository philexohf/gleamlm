"""数据预处理管道 CLI — 6 阶段标准管线入口。

核心编排逻辑在 gleamlm.data.pipeline.run_pipeline()，本文件仅负责命令行参数解析。

用法:
    python data_tools/pretrain/run_pipeline.py --sources wiki baike
    python data_tools/pretrain/run_pipeline.py --max-chars 2600000000 --ratios wiki:0.4,baike:0.3,edu:0.3
"""

import argparse
import os

from gleamlm.data.pipeline import (
    MIN_EN_RATIO,
    MIN_ZH_RATIO,
    SOURCES,
    run_pipeline,
)
from gleamlm.utils.config import load_yaml


def _variant_config(variant: str) -> dict:
    """加载 configs/{variant}.yaml（含 extends 继承）；不存在时直接报错退出。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(root, "configs", f"{variant}.yaml")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"ERROR: --variant {variant}: 找不到 {cfg_path}")
    return load_yaml(cfg_path)


def _ratios_from_variant(variant: str) -> str | None:
    """从 configs/{variant}.yaml 的 data_sources 读取目标占比。

    旧 build.py 会按变体自动读 yaml 配比，新管线曾丢失该语义、退化为等分。
    这里补回: 显式 --ratios 优先，否则按变体从 config 取，二者皆无才等分（并告警）。
    """
    cfg = _variant_config(variant)
    data_sources = cfg.get("data_sources")
    if not data_sources:
        raise SystemExit(f"ERROR: configs/{variant}.yaml 无 data_sources 配比定义")
    parts = []
    for s in data_sources:
        if "ratio" not in s:
            raise SystemExit(f"ERROR: configs/{variant}.yaml data_sources 条目缺 ratio: {s}")
        parts.append(f"{s['name']}:{s['ratio']}")
    print(f"[配比] 读取 {variant}.yaml data_sources → {','.join(parts)}")
    return ",".join(parts)


def main():
    p = argparse.ArgumentParser(description="GleamLM 数据预处理管道（6 阶段标准管线）")
    p.add_argument("--input", default="data/raw", help="原始数据目录（{name}_raw.txt）")
    p.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="只处理指定的源 (edu/news/wiki/baike/qa)，默认全部",
    )
    p.add_argument("--min-zh-ratio", type=float, default=MIN_ZH_RATIO,
                   help=f"中文源清洗: 最低汉字占比（默认 {MIN_ZH_RATIO}）")
    p.add_argument("--min-en-ratio", type=float, default=MIN_EN_RATIO,
                   help=f"英文源清洗: 最低英文字母占比（默认 {MIN_EN_RATIO}）")
    p.add_argument("--skip-exact-dedup", action="store_true", help="跳过 step1 粗去重")
    p.add_argument("--skip-clean", action="store_true", help="跳过 step2 清洗")
    p.add_argument("--skip-quality", action="store_true", help="跳过 step3 质量过滤")
    p.add_argument("--skip-dedup", action="store_true", help="跳过 step4 细去重")
    p.add_argument("--min-quality", type=float, default=0.30, help="Gopher 质量分阈值")
    p.add_argument("--minhash", action="store_true", help="细去重用 MinHash（默认 SimHash）")
    p.add_argument("--minhash-threshold", type=float, default=0.8, help="MinHash Jaccard 阈值")
    p.add_argument("--simhash-threshold", type=int, default=3, help="SimHash Hamming 阈值")
    p.add_argument("--prefix-len", type=int, default=100, help="news 前缀去重长度")
    p.add_argument(
        "--output-prefix",
        default=None,
        help="输出数据目录: 切分 txt 为 {dir}/{split}.txt，打包为 {dir}/{split}.bin/.idx；未传时从 --variant 的 data.data_dir 读取",
    )
    p.add_argument("--ratios", default=None, help="目标字符占比，如 wiki:0.6,news:0.4（默认从 --variant 的 yaml 读取）")
    p.add_argument("--variant", default=None, help="模型变体 (nano/lite/pro): 未传 --ratios 时读取 configs/{variant}.yaml 的 data_sources 配比")
    p.add_argument("--max-chars", type=int, default=None, help="总输出字符预算（天量数据时限制训练集大小）")
    p.add_argument("--tokenizer", default="bbpe", help="tokenizer (bbpe 或 gpt2)")
    p.add_argument("--tokenizer-path", default=None, help="BBPE checkpoint 目录（默认项目 12k）")
    p.add_argument("--workers", type=int, default=4, help="打包 tokenize 并行进程数")
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-verify", action="store_true", help="跳过 .bin/.idx 写后验证")
    args = p.parse_args()

    # ── 验证 sources ──
    sources = args.sources
    if sources:
        valid = {s["name"] for s in SOURCES}
        unknown = set(sources) - valid
        if unknown:
            print(f"ERROR: unknown sources: {unknown}")
            return

    # ── 解析配比来源 ──
    # 优先级: --ratios 显式 > --variant 的 configs/{variant}.yaml data_sources > 等分
    ratios = args.ratios
    if ratios is None and args.variant:
        ratios = _ratios_from_variant(args.variant)
    if ratios is None:
        print("[配比] 警告: 未传 --ratios 且未传 --variant，退化为各源等分")

    # ── 字符预算 ──
    # 优先级: --max-chars 显式 > 变体 yaml 的 training.max_train_chars > None（全量）
    max_chars = args.max_chars
    if max_chars is None and args.variant:
        max_chars = _variant_config(args.variant).get("training", {}).get("max_train_chars")
        print(f"[配比] max_chars = {max_chars}（来自 {args.variant}.yaml max_train_chars）")

    # ── 输出目录 ──
    # 优先级: --output-prefix 显式 > 变体 yaml 的 data.data_dir > data/processed
    output_prefix = args.output_prefix
    if output_prefix is None:
        if args.variant:
            output_prefix = _variant_config(args.variant).get("data", {}).get("data_dir")
            print(f"[数据] output_dir = {output_prefix}（来自 {args.variant}.yaml data.data_dir）")
        if not output_prefix:
            output_prefix = "data/processed"

    run_pipeline(
        input_dir=args.input,
        sources=sources,
        output_prefix=output_prefix,
        skip_exact_dedup=args.skip_exact_dedup,
        skip_clean=args.skip_clean,
        skip_quality=args.skip_quality,
        skip_dedup=args.skip_dedup,
        min_zh_ratio=args.min_zh_ratio,
        min_en_ratio=args.min_en_ratio,
        min_quality=args.min_quality,
        use_minhash=args.minhash,
        minhash_threshold=args.minhash_threshold,
        simhash_threshold=args.simhash_threshold,
        prefix_len=args.prefix_len,
        ratios=ratios,
        max_chars=max_chars,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        tokenizer=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
        workers=args.workers,
        skip_verify=args.skip_verify,
    )


if __name__ == "__main__":
    main()
