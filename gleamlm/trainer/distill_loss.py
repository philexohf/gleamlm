"""Knowledge Distillation loss — Teacher 教 Student。

L = α·CE(student, labels) + (1-α)·KL(σ(T/τ) || σ(S/τ))·τ²
τ 放大 logits 中的暗知识；τ² 使 soft loss 梯度与 τ 无关；Teacher 冻结。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """知识蒸馏 loss = α * CE + (1-α) * KL * τ²。

    Args:
        student_logits: Student 模型 logits [B, S, V]
        teacher_logits: Teacher 模型 logits [B, S, V] (no_grad 推理)
        labels:         目标 token ids [B, S]
        temperature:    软标签温度 τ (默认 4.0)
        alpha:          硬标签权重 (默认 0.5)
        ignore_index:   padding token 忽略

    Returns:
        蒸馏总 loss
    """
    T = temperature

    s_logits = student_logits[:, :-1, :].contiguous()
    t_logits = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    hard_loss = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    # 软标签 loss: log_softmax/τ 梯度与 1/τ 成正比，乘 τ² 后与 τ 无关
    s_log_sm = F.log_softmax(s_logits / T, dim=-1)
    t_sm = F.softmax(t_logits / T, dim=-1)
    kl = F.kl_div(
        s_log_sm.view(-1, s_log_sm.size(-1)),
        t_sm.view(-1, t_sm.size(-1)),
        reduction="batchmean",
    )
    soft_loss = kl * T * T

    return alpha * hard_loss + (1 - alpha) * soft_loss
