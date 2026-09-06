"""RLHF 训练工具 — PPO、GRPO 与共享奖励函数。

PPO (Proximal Policy Optimization): value network + clip + GAE + entropy
GRPO (Group Relative Policy Optimization): group 内归一化优势，无 value network
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


def compute_reward(response: str, ground_truth: str | None = None) -> float:
    """规则奖励 — 有 ground_truth 时做答案匹配，否则启发式打分。

    评分维度（启发式，无 ground_truth 时）:
      - 结构化 (+0.2): 包含换行 → 有分段/列表
      - 长度合理 (+0.1): 20-512 字符内
      - 完整性   (+0.1): 以句号/问号/感叹号结尾

    有 ground_truth（工业规则 reward，对齐工业轨 default_reward）:
      - 答案命中 +1.0
      - 空回答 -1.0
      - 未命中 0.0
    """
    if ground_truth is not None:
        if not response:
            return -1.0
        return 1.0 if str(ground_truth).strip() in response else 0.0

    r = 0.0
    if "\n" in response:
        r += 0.2
    if 20 <= len(response) <= 512:
        r += 0.1
    if response and response[-1] in "。！？.!?":
        r += 0.1
    return r


# GRPO: 无 value network，用 group 内归一化奖励做优势估计 (MC baseline)；
# loss = -E[log π_θ(y|x)·A] + β·KL(π_θ || π_ref)，β=0.01-0.1。


def grpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    response_tokens: torch.Tensor,
    prompt_len: int,
    adv: torch.Tensor,
    beta: float = 0.04,
) -> torch.Tensor:
    """GRPO 策略梯度 + KL penalty。

    三行公式:
      L = L_policy + β · L_KL
      L_policy = -E[ log π_θ(y|x) · A ]
      A = (r_i - mean(r_group)) / std(r_group)

    Args:
        policy_logits:  当前 policy 的完整序列 logits [B, S, V]
        ref_logits:     冻结的 reference model logits [B, S, V]
        response_tokens: rollout 实际采样出的 token ids [B, S]
        prompt_len:     prompt 部分的 token 数
        adv:            group 内归一化后的优势 [B]
        beta:           KL penalty 系数 (默认 0.04)

    Returns:
        policy_loss + beta * kl
    """
    response_logits = policy_logits[:, prompt_len - 1 : -1, :]
    response_ref = ref_logits[:, prompt_len - 1 : -1, :]
    response_len = max(response_logits.size(1), 1)
    response_tokens = response_tokens[:, prompt_len:]

    # 按 response 长度归一化的平均对数概率
    log_probs = F.log_softmax(response_logits, dim=-1)
    tok_lp = log_probs.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)
    avg_lp = tok_lp.sum(dim=-1) / response_len

    policy_loss = -(avg_lp * adv).mean()

    ref_lp = F.log_softmax(response_ref, dim=-1)
    ref_tok_lp = ref_lp.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)
    avg_ref_lp = ref_tok_lp.sum(dim=-1) / response_len
    # KL 惩罚用 exp(Δ)-Δ-1 (恒 ≥0): 直接 mean(log_p - log_ref) 可为负，
    # 早期 policy≈ref 时会把“惩罚”变成奖励，方向反了。
    diff = avg_lp - avg_ref_lp
    kl = (diff.exp() - diff - 1.0).mean()

    return policy_loss + beta * kl


# PPO: clip loss (限制 π_θ/π_old 在 [1-ε, 1+ε]) + value loss + entropy bonus；
# 优势用 GAE，比 GRPO 的 group baseline 更精确但多一个 value network。


class ValueHead(nn.Module):
    """PPO 价值网络 — 从最后一层 hidden states 预测标量 V(s)。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = cast(torch.Tensor, self.proj(hidden_states))
        return out.squeeze(-1)


def ppo_loss(
    policy_logits: torch.Tensor,
    old_logits: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    response_tokens: torch.Tensor,
    prompt_len: int,
    epsilon: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """PPO clipped surrogate objective + value loss + entropy bonus。

    Args:
        policy_logits: 当前 policy 的完整序列 logits [B, S, V]
        old_logits:    旧 policy (π_old) 的完整序列 logits [B, S, V]
        values:        ValueHead 预测的 V(s) [B, S]
        rewards:       每条序列的 terminal reward [B]（标量，只在最后位置给出）
        response_tokens: rollout 实际采样的完整序列 token [B, S]（重要性比必须
                        用采样时的 token 计算，用当前策略 argmax 会算错比率）
        prompt_len:    prompt 部分的 token 数（loss 只算 response 部分）
        epsilon:       PPO clip 范围 (默认 0.2)
        value_coeff:   value loss 权重
        entropy_coeff: entropy bonus 权重
        gamma:         GAE 折扣因子
        lam:           GAE λ 参数

    Returns:
        总 loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
    """
    resp_logits = policy_logits[:, prompt_len - 1 : -1, :]
    old_resp = old_logits[:, prompt_len - 1 : -1, :]
    R = max(resp_logits.size(1), 1)
    B = resp_logits.size(0)
    resp_tokens = response_tokens[:, prompt_len:]

    log_probs = F.log_softmax(resp_logits, dim=-1)
    old_lp = F.log_softmax(old_resp, dim=-1)
    tok_lp = log_probs.gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1)
    old_tok_lp = old_lp.gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1)
    ratio = (tok_lp - old_tok_lp).exp()

    resp_vals = values[:, prompt_len - 1 : -1].float()
    next_vals = torch.cat([resp_vals[:, 1:], torch.zeros(B, 1, device=resp_vals.device)], dim=1)
    resp_rewards = torch.zeros_like(resp_vals)
    resp_rewards[:, -1] = rewards.float()
    deltas = resp_rewards + gamma * next_vals - resp_vals

    adv = torch.zeros_like(deltas)
    gae: torch.Tensor = torch.zeros_like(deltas[:, 0])
    for t in reversed(range(R)):
        gae = deltas[:, t] + gamma * lam * gae
        adv[:, t] = gae

    # 优势项对 policy 梯度是常数: 不 detach 会让值函数梯度倒灌进 policy/backbone
    adv = adv.detach()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    returns = adv + resp_vals
    value_loss = F.mse_loss(resp_vals, returns.detach())

    probs = F.softmax(resp_logits, dim=-1)
    entropy = -(probs * log_probs).sum(-1).mean()

    return policy_loss + value_coeff * value_loss - entropy_coeff * entropy
