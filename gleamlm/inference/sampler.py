"""GleamLM 采样策略：temperature / top_k / top_p"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
    penalty_window: int = 0,
) -> torch.Tensor:

    if repetition_penalty != 1.0 and generated_ids is not None:
        if logits.requires_grad:
            logits = logits.clone()
        window_ids = generated_ids[-penalty_window:] if penalty_window > 0 else generated_ids
        for gid in set(window_ids):
            logits[..., gid] = logits[..., gid] / repetition_penalty

    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    if probs.dim() == 1:
        sampled = torch.multinomial(probs, num_samples=1)
    else:
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled
