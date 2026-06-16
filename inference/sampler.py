"""Xfind-Mini 采样策略：temperature / top_k / top_p"""

import torch
import torch.nn.functional as F


def sample_token(logits, temperature=1.0, top_k=0, top_p=0.0,
                 repetition_penalty=1.0, generated_ids=None):

    # 重复惩罚
    if repetition_penalty != 1.0 and generated_ids is not None:
        logits = logits.clone()
        for gid in set(generated_ids):
            val = logits[..., gid]
            if val > 0:
                logits[..., gid] = val / repetition_penalty
            else:
                logits[..., gid] = val * repetition_penalty

    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    if probs.dim() == 1:
        sampled = torch.multinomial(probs, num_samples=1)
    else:
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled
