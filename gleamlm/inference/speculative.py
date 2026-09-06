"""
Speculative Decoding — 小模型草稿 + 大模型验证。

核心流程:
  1. Draft model 自回归生成 K 个候选 token (γ=K-1)
  2. Target model 一次 forward 验证全部候选
  3. Rejection sampling: 保留 p_target / p_draft 比例接受的 token
  4. 被拒绝的位置用 target model 重新采样

用法:"""

# 自回归每步只生成 1 个 token，而 GPU 前向是 memory-bound (读权重 ≫ 计算)；
# draft 猜 K 个 → target 一次前向验证 K 个，rejection sampling 保证输出
# 分布严格等于 target (无损加速)；γ 越大加速越明显但浪费越多。

import torch
import torch.nn.functional as F
from torch import nn


def _greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1, keepdim=True)


def _categorical_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        return _greedy(logits)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(*logits.shape[:2], 1)


class SpeculativeGenerator:
    def __init__(self, target_model: nn.Module, draft_model: nn.Module, device: torch.device):
        self.target = target_model.eval().to(device)
        self.draft = draft_model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        gamma: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        seq = input_ids.to(self.device)
        # 仅支持 batch=1: 拒绝/接受分支按整序列处理
        assert seq.dim() == 2 and seq.size(0) == 1, "SpeculativeGenerator 仅支持 batch=1"
        generated = 0

        while generated < max_new_tokens:
            remain = max_new_tokens - generated
            K = min(gamma, remain)

            n = seq.size(1)
            draft_seq = seq.clone()
            draft_logits_list = []

            # Stage 1: draft 自回归 K 步，logits 存下来供 rejection 用 (q(x))
            for _ in range(K):
                logits, _, _, _ = self.draft(draft_seq)
                # _categorical_sample 返回 [B, 1, 1]，squeeze 成 [B, 1] 才能与序列 cat
                nxt = _categorical_sample(logits[:, -1:, :], temperature).squeeze(1)
                draft_seq = torch.cat([draft_seq, nxt], dim=1)
                draft_logits_list.append(logits[:, -1, :])

            # Stage 2: target 一次 forward 验证整个 draft_seq，拿到 K 个位置的 logits
            tgt_logits, _, _, _ = self.target(draft_seq)
            tgt_logits_list = [tgt_logits[:, n + i - 1, :] for i in range(K)]

            # Stage 3: 以 min(1, p_T(x)/q_T(x)) 接受 draft 采样的 token；
            # 拒绝则从 (p_T-q_T)⁺ 残差重采样——两项之和恒等于 p_T，输出分布严格
            # 等于 target 的 temperature 缩放分布（p/q 都必须用同一 T，否则无偏性失效）
            accepted = 0
            for i in range(K):
                x = draft_seq[:, n + i].unsqueeze(-1)
                if temperature <= 0:
                    # greedy: 双方 argmax 一致才接受，拒绝位直接取 target argmax
                    tgt_choice = tgt_logits_list[i].argmax(dim=-1, keepdim=True)
                    if torch.equal(tgt_choice, x):
                        seq = torch.cat([seq, x], dim=1)
                        accepted += 1
                        generated += 1
                    else:
                        seq = torch.cat([seq, tgt_choice], dim=1)
                        generated += 1
                        break
                    continue
                p = F.softmax(tgt_logits_list[i] / temperature, dim=-1)
                q = F.softmax(draft_logits_list[i] / temperature, dim=-1)

                ratio = (p.gather(-1, x) / (q.gather(-1, x) + 1e-8)).squeeze(-1)
                accept = torch.rand(1, device=self.device) < ratio
                if accept:
                    seq = torch.cat([seq, x], dim=1)
                    accepted += 1
                    generated += 1
                else:
                    # 拒绝: 从 (p-q)⁺ 残差分布采样（无偏性的来源）
                    residual = (p - q).clamp(min=0)
                    residual = residual / (residual.sum(dim=-1, keepdim=True) + 1e-8)
                    replacement = torch.multinomial(residual.view(-1, residual.size(-1)), 1)
                    seq = torch.cat([seq, replacement], dim=1)
                    generated += 1
                    break

            # 全部接受 → 额外从 target 采样一个 token（免费的最后一 token）；
            # 已到 max_new_tokens 时跳过，避免越界多生成一个
            if accepted == K and generated < max_new_tokens:
                final_logits = tgt_logits[:, -1:, :]
                nxt = _categorical_sample(final_logits, temperature).squeeze(1)
                seq = torch.cat([seq, nxt], dim=1)
                generated += 1

        return seq
