"""GleamLM 统一评估框架 — PPL / 知识探针 / CEVAL / CMMLU"""

from .benchmark import BenchmarkResult, evaluate_ceval, evaluate_cmmlu
from .knowledge import KnowledgeResult, evaluate_knowledge
from .ppl import PPLResult, evaluate_multiple, evaluate_ppl

__all__ = [
    "evaluate_ppl",
    "evaluate_multiple",
    "PPLResult",
    "evaluate_knowledge",
    "KnowledgeResult",
    "evaluate_ceval",
    "evaluate_cmmlu",
    "BenchmarkResult",
]
