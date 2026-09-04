"""GleamLM 统一评估入口 — PPL + lm-eval 标准 benchmark。

用法:
    python -m tools.eval_runner --model checkpoints/best_model.pt --data_dir data/lite/pretrain
    python -m tools.eval_runner --model checkpoints/best_model.pt --benchmarks ceval,cmmlu
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
        default="ppl",
        help="逗号分隔：ppl, ceval, cmmlu, mmlu, all",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录（如 data/lite/pretrain）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=0, help="0=从模型配置检测")
    parser.add_argument("--max_batches", type=int, default=None, help="PPL 最大批次数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="", help="结果 JSON 路径")
    parser.add_argument("--limit", type=int, default=None, help="lm-eval 每任务样本上限")
    args = parser.parse_args()

    if not args.device or args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # 快捷映射: "all" → ceval + cmmlu + mmlu
    BENCHMARK_MAP = {"ceval": "ceval-valid", "cmmlu": "cmmlu", "mmlu": "mmlu"}
    if "all" in benchmarks:
        benchmarks.remove("all")
        for k in BENCHMARK_MAP:
            if k not in benchmarks:
                benchmarks.append(k)
    all_results: dict[str, dict] = {}

    # ── PPL ──
    if "ppl" in benchmarks:
        from gleamlm.evaluation.ppl import evaluate_ppl

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

    # ── lm-eval benchmarks ──
    lm_eval_tasks = [b for b in benchmarks if b in ("ceval", "cmmlu", "mmlu")]
    if lm_eval_tasks:
        from eval.run_evals import evaluate_from_ckpt

        task_map = BENCHMARK_MAP
        tasks = [task_map[t] for t in lm_eval_tasks]

        print(f"\n{'=' * 60}")
        print(f"lm-eval: {tasks}")
        print(f"{'=' * 60}")

        output_dir = os.path.join(os.path.dirname(args.data_dir), "eval_results")
        results = evaluate_from_ckpt(args.model, tasks, output_dir, limit=args.limit)
        for task, res in results.get("results", {}).items():
            acc = res.get("acc,none", res.get("exact_match,none", None))
            all_results[task] = {
                "accuracy": round(acc, 4) if acc is not None else None,
                "benchmark": task,
            }
            print(f"  {task}: {acc}")

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, data in all_results.items():
        if "ppl" in data:
            print(f"  {name:>10}: PPL={data.get('ppl', '?'):.2f}, Loss={data.get('loss', '?'):.4f}")
        elif "accuracy" in data:
            print(f"  {name:>10}: Acc={data.get('accuracy', '?')}")

    # ── 保存 ──
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
