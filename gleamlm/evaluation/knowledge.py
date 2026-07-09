"""知识探针评估 — 填空测试 + 实体一致性检查"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from gleamlm.inference.sampler import sample_token
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.torch_utils import safe_autocast


@dataclass
class KnowledgeResult:
    correct: int = 0
    wrong: int = 0
    hallucination: int = 0
    detailed: list[dict[str, Any]] = field(default_factory=list)
    entity_consistency: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.hallucination

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "correct": self.correct,
            "wrong": self.wrong,
            "hallucination": self.hallucination,
            "total": self.total,
            "entity_consistency": self.entity_consistency,
            "n_consistent": sum(1 for v in self.entity_consistency.values() if v >= 2),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeResult(acc={self.accuracy:.2%}, "
            f"{self.correct}/{self.total} correct, "
            f"{self.hallucination} hallucinations)"
        )


# 默认知识探针池
DEFAULT_FACT_PROMPTS = [
    ("世界上最高的山峰是", "珠穆朗玛峰"),
    ("水的化学式是", "H2O"),
    ("中国的首都是", "北京"),
    ("爱因斯坦提出了", "相对论"),
    ("光的速度大约是每秒", "30万公里"),
    ("地球绕着什么转", "太阳"),
    ("人类登上月球是在", "1969"),
    ("DNA的全称是", "脱氧核糖核酸"),
    ("第一个进入太空的人是", "加加林"),
    ("太阳系有几大行星", "八"),
    ("诺贝尔奖的创立者是", "诺贝尔"),
    ("光合作用需要", "阳光"),
    ("地球的卫星是", "月球"),
    ("世界上最大的海洋是", "太平洋"),
    ("一年有多少天", "365"),
    ("人体最大的器官是", "皮肤"),
    ("鲨珑GleamLM是什么", "轻量级开源对话模型"),
]


DEFAULT_ENTITY_PROBES = {
    "北京": ["中国的首都是哪里", "天安门在哪个城市", "故宫位于"],
    "光速": ["光的速度是多少", "真空中最快的是什么"],
    "爱因斯坦": ["爱因斯坦提出了什么", "相对论是谁提出的", "E=mc^2是谁的公式"],
    "太阳": ["太阳是什么", "地球绕着什么转", "离我们最近的恒星是"],
    "DNA": ["DNA是什么", "遗传物质叫什么"],
    "诺贝尔": ["诺贝尔奖是谁设立的", "诺贝尔发明了什么"],
    "太平洋": ["最大的海洋", "太平洋在哪"],
    "月球": ["月球的别称", "地球的卫星是什么"],
}


def _simple_generate(
    model: GleamLMModel,
    tokenizer: BBPETokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
) -> str:
    """独立生成函数 — 使用 KV Cache 逐 token 增量推理。

    首步预填充 prompt，后续每步仅输入新 token 并复用 KV Cache，
    RoPE 通过 offset 自动递增位置编码。
    """
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], device=device)
    generated_ids = prompt_ids.copy()
    past_kv = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            with safe_autocast():
                logits, past_kv = model(input_ids, past_kv_list=past_kv)
            next_token = sample_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=0.0,
                generated_ids=generated_ids,
            )
            token_id = next_token.item()
            if token_id in {tokenizer.eos_id, tokenizer.special_tokens.get("<|im_end|>")}:
                break
            generated_ids.append(token_id)
            input_ids = torch.tensor([[token_id]], device=device)

    full = tokenizer.decode(generated_ids)
    generated = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
    return generated[:200]


def _check_answer(
    generated: str, expected: str, hallucination_keywords: list[str] | None = None
) -> str:
    """检查生成文本是否包含预期答案。返回 CORRECT / WRONG / HALLUCINATION"""
    gen_lower = generated.lower().replace(" ", "")
    answers = [a.strip().lower() for a in expected.split(",")]
    for ans in answers:
        if ans in gen_lower:
            return "CORRECT"
    if hallucination_keywords:
        for hw in hallucination_keywords:
            if hw in generated:
                return "HALLUCINATION"
    return "WRONG"


def evaluate_knowledge(
    model: GleamLMModel,
    tokenizer: BBPETokenizer,
    device: str = "cuda",
    fact_prompts: list[tuple[str, str]] | None = None,
    entity_probes: dict[str, list[str]] | None = None,
    hallucination_keywords: list[str] | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    verbose: bool = True,
) -> KnowledgeResult:
    """运行知识探针评估。"""
    if fact_prompts is None:
        fact_prompts = DEFAULT_FACT_PROMPTS
    if entity_probes is None:
        entity_probes = DEFAULT_ENTITY_PROBES

    model.eval()
    result = KnowledgeResult()

    # A1: 事实填空
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"A1: FACT FILL-IN ({len(fact_prompts)} prompts)")
        print(f"{'=' * 60}")

    for prompt, expected in fact_prompts:
        generated = _simple_generate(model, tokenizer, prompt, device, max_new_tokens, temperature)
        label = _check_answer(generated, expected, hallucination_keywords)

        if label == "CORRECT":
            result.correct += 1
        elif label == "HALLUCINATION":
            result.hallucination += 1
        else:
            result.wrong += 1

        result.detailed.append(
            {"prompt": prompt, "expected": expected, "generated": generated[:100], "result": label}
        )

        if verbose:
            symbols = {"CORRECT": "OK", "HALLUCINATION": "HALU", "WRONG": "WRONG"}
            print(f"  [{symbols[label]:>4}] {prompt} → {generated[:60]}...")

    if verbose:
        print(
            f"\n  Accuracy: {result.correct}/{result.total} ({result.accuracy:.1%}), "
            f"{result.hallucination} hallucinations"
        )

    # A2: 实体探针
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"A2: ENTITY PROBE ({len(entity_probes)} entities)")
        print(f"{'=' * 60}")

    for entity, questions in entity_probes.items():
        correct = 0
        for q in questions:
            generated = _simple_generate(model, tokenizer, q, device, max_new_tokens, temperature)
            if entity.lower() in generated.lower():
                correct += 1
        result.entity_consistency[entity] = correct

        if verbose:
            icon = "***" if correct == len(questions) else ("*" if correct > 0 else " ")
            print(f"  [{icon}] {entity}: {correct}/{len(questions)}")

    if verbose:
        consistent = sum(
            1
            for entity, v in result.entity_consistency.items()
            if entity in entity_probes and v >= len(entity_probes[entity]) * 0.67
        )
        print(f"\n  Consistent entities: {consistent}/{len(entity_probes)}")

    return result
