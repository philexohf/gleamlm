"""基准测试评估 — CEVAL / CMMLU / 多选题"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.torch_utils import safe_autocast


@dataclass
class BenchmarkResult:
    """基准测试结果"""

    name: str  # ceval / cmmlu
    accuracy: float
    total: int
    correct: int
    subject_scores: dict[str, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.name,
            "accuracy": round(self.accuracy, 4),
            "total": self.total,
            "correct": self.correct,
            "subjects": self.subject_scores,
            **self.extra,
        }

    def __repr__(self) -> str:
        return f"BenchmarkResult({self.name}: acc={self.accuracy:.2%}, {self.correct}/{self.total})"


def _mc_generate(
    model: GleamLMModel, tokenizer: BBPETokenizer, prompt: str, choices: list[str], device: str
) -> str:
    """多选题生成：预填充 prompt KV Cache，仅对选项部分增量推理。

    将 prompt 预填充一次得到 KV Cache，然后每个选项仅输入选项 token
    并复用 Cache，避免重复计算 prompt 部分。
    """
    prompt_ids = tokenizer.encode(prompt)
    prompt_t = torch.tensor([prompt_ids], device=device)

    with torch.no_grad(), safe_autocast():
        _, prompt_kv = model(prompt_t)

    choice_scores = []

    for choice in choices:
        choice_ids = tokenizer.encode(choice)
        if not choice_ids:
            choice_scores.append(float("-inf"))
            continue

        choice_t = torch.tensor([choice_ids], device=device)

        with torch.no_grad(), safe_autocast():
            logits, _ = model(choice_t, past_kv_list=prompt_kv)

        total_log_prob = 0.0
        for i, token_id in enumerate(choice_ids):
            pred_logits = logits[0, i, :]
            log_prob = torch.nn.functional.log_softmax(pred_logits, dim=-1)[token_id].item()
            total_log_prob += log_prob

        avg_log_prob = total_log_prob / len(choice_ids)
        choice_scores.append(avg_log_prob)

    best_idx = choice_scores.index(max(choice_scores))
    return chr(ord("A") + best_idx) if best_idx < 4 else str(best_idx)


def _load_ceval(data_dir: str) -> list[dict[str, Any]]:
    """加载 CEVAL 数据集。支持单文件 ceval.json 或按学科拆分。"""
    ceval_file = os.path.join(data_dir, "ceval.json")
    if os.path.exists(ceval_file):
        with open(ceval_file, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    # 按学科拆分：ceval/high_school_math.json, etc.
    items = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json") and fname != "ceval.json":
            path = os.path.join(data_dir, fname)
            with open(path, encoding="utf-8") as f:
                subject_items = json.load(f)
                subject = os.path.splitext(fname)[0]
                for item in subject_items:
                    item.setdefault("subject", subject)
                items.extend(subject_items)
    return items


def _build_prompt(item: dict) -> str:
    """构建多选题 prompt: 问题 + A/B/C/D 选项"""
    question = item.get("question", "")
    choices = []
    for key in ["A", "B", "C", "D"]:
        if key in item:
            choices.append(item[key])

    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + i)}. {choice}\n"
    prompt += "答案："
    return prompt


def evaluate_ceval(
    model: GleamLMModel,
    tokenizer: BBPETokenizer,
    data_dir: str,
    device: str = "cuda",
    subjects: list[str] | None = None,
    max_samples: int | None = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """CEVAL 中文综合能力评估。

    数据集格式: [{question, A, B, C, D, answer, subject}, ...]
    """
    items = _load_ceval(data_dir)
    if not items:
        raise FileNotFoundError(f"No CEVAL data found in {data_dir}")

    if subjects:
        items = [it for it in items if it.get("subject", "") in subjects]
    if max_samples:
        items = items[:max_samples]

    model.eval()
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}
    total_correct = 0

    for item in items:
        subject = item.get("subject", "general")
        choices = [item.get(k, "") for k in ["A", "B", "C", "D"] if k in item]
        prompt = _build_prompt(item)
        predicted = _mc_generate(model, tokenizer, prompt, choices, device)
        expected = item.get("answer", "")

        correct = predicted == expected
        if correct:
            total_correct += 1
            subject_correct[subject] = subject_correct.get(subject, 0) + 1
        subject_total[subject] = subject_total.get(subject, 0) + 1

        if verbose and len(items) <= 20:
            mark = "✓" if correct else "✗"
            print(
                f"  [{mark}] {subject}: pred={predicted}, ans={expected} | {item.get('question', '')[:50]}..."
            )

    # 计算各学科分数
    subject_scores = {}
    for subj in subject_total:
        subject_scores[subj] = round(subject_correct.get(subj, 0) / max(1, subject_total[subj]), 4)

    accuracy = total_correct / max(1, len(items))
    result = BenchmarkResult(
        name="ceval",
        accuracy=accuracy,
        total=len(items),
        correct=total_correct,
        subject_scores=subject_scores,
    )

    if verbose:
        print(f"\nCEVAL: {result.correct}/{result.total} = {accuracy:.2%}")
        if len(subject_scores) > 1:
            for subj, acc in sorted(subject_scores.items()):
                print(f"  {subj}: {acc:.2%}")
    return result


def evaluate_cmmlu(
    model: GleamLMModel,
    tokenizer: BBPETokenizer,
    data_dir: str,
    device: str = "cuda",
    **kwargs: Any,
) -> BenchmarkResult:
    """CMMLU 评估 — 同 CEVAL 格式，可直接复用。"""
    result = evaluate_ceval(model, tokenizer, data_dir, device, **kwargs)
    result.name = "cmmlu"
    return result
