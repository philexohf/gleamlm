"""GleamLM 统一评估 — PPL + 标准 benchmark。

ppl.py               — 手写 PPL（基础指标，与实现无关；与 hf 参考一致，保留）
eval/run_evals.py    — lm-evaluation-harness 封装（CEVAL/CMMLU/MMLU 等标准 benchmark）
tools/eval_runner.py — 统一入口（ppl + lm-eval）
"""

from .ppl import PPLResult, evaluate_multiple, evaluate_ppl

__all__ = [
    "evaluate_ppl",
    "evaluate_multiple",
    "PPLResult",
]
