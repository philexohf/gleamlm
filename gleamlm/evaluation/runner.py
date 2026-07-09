"""GleamLM 统一评估入口

用法:
    # 全部评估
    python -m gleamlm.evaluation.runner --model checkpoints/best_model.pt

    # 指定评估项
    python -m gleamlm.evaluation.runner --model checkpoints/best_model.pt --benchmarks ppl,knowledge

    # CEVAL 评估
    python -m gleamlm.evaluation.runner --model checkpoints/best_model.pt --benchmarks ceval --ceval_dir data/ceval
    --cmmlu_dir data/cmmlu  # CMMLU 独立目录
"""

import argparse
import json
import os
from datetime import datetime

import torch

from gleamlm import load_model_for_inference
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="GleamLM Evaluation Runner")
    parser.add_argument("--model", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="ppl,knowledge",
        help="逗号分隔：ppl,knowledge,ceval,cmmlu",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录（如 data/lite_data）")
    parser.add_argument("--ceval_dir", type=str, default="")
    parser.add_argument("--cmmlu_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_seq_len", type=int, default=0, help="序列长度（0=从模型配置自动检测）"
    )
    parser.add_argument("--max_batches", type=int, default=None, help="PPL 最大批次数（调试用）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="", help="结果保存到 JSON 文件")
    args = parser.parse_args()

    if not args.device or args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型和分词器
    print("=" * 60)
    print("GleamLM Evaluation Runner")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Benchmarks: {args.benchmarks}")
    print("=" * 60)

    model, config = load_model_for_inference(args.model, args.device)
    tokenizer = BBPETokenizer.load(args.tokenizer)
    model.eval()

    total, _ = model.get_num_params()
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else config.get("max_seq_len", 2048)
    print(f"\nModel: {total / 1e6:.2f}M params, Vocab: {len(tokenizer)}, MaxSeqLen: {max_seq_len}")

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    all_results = {}

    # PPL
    if "ppl" in benchmarks:
        from gleamlm.evaluation import evaluate_ppl

        print(f"\n{'=' * 60}")
        print("PPL Evaluation")
        print(f"{'=' * 60}")
        result = evaluate_ppl(
            model,
            tokenizer,
            args.data_dir,
            max_seq_len=max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
            max_batches=args.max_batches,
        )
        all_results["ppl"] = result.to_dict()
        print(f"  Result: {result}")

    # Knowledge Probe
    if "knowledge" in benchmarks:
        from gleamlm.evaluation import evaluate_knowledge

        print(f"\n{'=' * 60}")
        print("Knowledge Probe")
        print(f"{'=' * 60}")
        result = evaluate_knowledge(model, tokenizer, device=args.device)
        all_results["knowledge"] = result.to_dict()

    # CEVAL
    if "ceval" in benchmarks:
        from gleamlm.evaluation import evaluate_ceval

        ceval_dir = args.ceval_dir or os.path.join(os.path.dirname(args.data_dir), "ceval")
        print(f"\n{'=' * 60}")
        print(f"CEVAL ({ceval_dir})")
        print(f"{'=' * 60}")
        try:
            result = evaluate_ceval(model, tokenizer, ceval_dir, device=args.device)
            all_results["ceval"] = result.to_dict()
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")

    # CMMLU
    if "cmmlu" in benchmarks:
        from gleamlm.evaluation import evaluate_cmmlu

        cmmlu_dir = args.cmmlu_dir or os.path.join(os.path.dirname(args.data_dir), "cmmlu")
        print(f"\n{'=' * 60}")
        print(f"CMMLU ({cmmlu_dir})")
        print(f"{'=' * 60}")
        try:
            result = evaluate_cmmlu(model, tokenizer, cmmlu_dir, device=args.device)
            all_results["cmmlu"] = result.to_dict()
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")

    # 汇总
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, data in all_results.items():
        if "ppl" in data:
            print(f"  {name:>10}: PPL={data.get('ppl', '?'):.2f}, Loss={data.get('loss', '?'):.4f}")
        elif "accuracy" in data:
            print(
                f"  {name:>10}: Acc={data['accuracy']:.2%}, {data.get('correct', '?')}/{data.get('total', '?')}"
            )

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.data_dir, f"eval_{model_name}_{ts}.json")

    all_results["_meta"] = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "model_params_m": total / 1e6,
        "vocab_size": len(tokenizer),
        "device": args.device,
    }

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
