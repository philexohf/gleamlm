"""
Xfind-Mini 采样策略

提供多种文本生成采样方法：
    - temperature: 温度调节采样
    - top_k: 仅保留概率最高的 K 个 token
    - top_p (nucleus): 累积概率达到 p 的最小 token 集合
"""

import torch
import torch.nn.functional as F


def sample_token(logits, temperature=1.0, top_k=0, top_p=0.0,
                 repetition_penalty=1.0, generated_ids=None):
    """
    从 logits 中采样下一个 token

    Args:
        logits: [batch, vocab_size] 或 [vocab_size]
        temperature: 温度参数，越高越随机（默认 1.0）
        top_k: Top-K 采样，仅保留概率最高的 K 个 token（0=禁用）
        top_p: Top-P 采样，累积概率达到 p 时截断（0.0=禁用）
        repetition_penalty: 重复惩罚（>1.0 惩罚已出现 token，1.0=禁用）
        generated_ids: 已生成的 token ID 列表，用于重复惩罚

    Returns:
        sampled_ids: [batch] 或标量
    """

    # 重复惩罚（clone 避免修改调用方的原始 logits）
    if repetition_penalty != 1.0 and generated_ids is not None:
        logits = logits.clone()
        for gid in set(generated_ids):
            val = logits[..., gid]
            if val > 0:
                logits[..., gid] = val / repetition_penalty
            else:
                logits[..., gid] = val * repetition_penalty

    # 温度缩放
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Top-K 过滤
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # 获取第 K 大的值作为阈值
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # Top-P 过滤
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过 p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过 p 的 token（确保至少保留 1 个）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  # 始终保留最高概率的 token

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # 采样
    probs = F.softmax(logits, dim=-1)
    # 使用多项式分布采样
    if probs.dim() == 1:
        sampled = torch.multinomial(probs, num_samples=1)
    else:
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled
