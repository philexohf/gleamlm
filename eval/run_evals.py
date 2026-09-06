"""标准评测 — lm-evaluation-harness 封装。

用法:
    python -m eval.run_evals --model checkpoints/best_model.pt --tasks ceval-valid cmmlu

国内网络：任务数据来自 HuggingFace Hub，请先设置 HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import os

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizerFast

from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from hf.hf_config import gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint


def _build_hf_tokenizer(tokenizer_dir: str, output_dir: str) -> PreTrainedTokenizerFast:
    """自研 BBPE → HF tokenizer.json → PreTrainedTokenizerFast（lm-eval HFLM 需要）。"""
    hf_tok_dir = os.path.join(output_dir, "hf_tokenizer")
    BBPETokenizer.load(tokenizer_dir).export_to_hf_format(hf_tok_dir)
    return PreTrainedTokenizerFast.from_pretrained(hf_tok_dir)


def evaluate_from_ckpt(
    ckpt_path: str, tasks: list[str], output_dir: str, limit: int | None = None,
    tokenizer_dir: str | None = None,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = extract_checkpoint_config(ckpt)
    hf_cfg = gleamlm_config_from_core(cfg)
    model = GleamLMForCausalLM(hf_cfg)
    missing, unexpected = load_from_checkpoint(model, ckpt, strict=True)
    if missing or unexpected:
        print(f"[warn] eval load — missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = _build_hf_tokenizer(tokenizer_dir or DEFAULT_TOKENIZER_PATH, output_dir)

    results = simple_evaluate(
        model=HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size="auto",
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        tasks=tasks,
        limit=limit,
    )
    out_path = os.path.join(output_dir, "results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for task, res in results["results"].items():
            acc = res.get("acc,none", res.get("exact_match,none", "N/A"))
            line = f"{task}: {acc}"
            print(line)
            f.write(line + "\n")
    print(f"Saved: {out_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM eval via lm-evaluation-harness")
    p.add_argument("--model", type=str, required=True, help="Checkpoint .pt path")
    p.add_argument("--tasks", type=str, nargs="+", default=["mmlu", "ceval-valid", "cmmlu"])
    p.add_argument("--output_dir", type=str, default="./eval_results")
    p.add_argument("--tokenizer_dir", type=str, default=None, help="BBPE tokenizer dir")
    p.add_argument("--limit", type=int, default=None, help="Limit samples per task")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_from_ckpt(args.model, args.tasks, args.output_dir, args.limit, args.tokenizer_dir)
