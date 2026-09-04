import math

import torch
import torch.nn.functional as F

from gleamlm.models.model import GQA, RMSNorm, repeat_kv


# NoPE: 去掉 RoPE，靠 QK-Norm 从上下文语义隐含位置关系；d_model 不够宽时不如 RoPE。
# ALiBi: softmax 前加 head-wise 等比斜率 × 距离惩罚，无学习参数
def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """ALiBi head-wise slope 生成，几何级数: 2^(-8/num_heads)。"""
    start = 2 ** (-8 / num_heads)
    return start ** torch.arange(1, num_heads + 1)


def build_alibi_bias(seq_len: int, num_heads: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    """构建 [1, n_head, seq_len, total_len] ALiBi bias 矩阵。"""
    total = offset + seq_len
    slopes = get_alibi_slopes(num_heads).to(device)          # [n_head]
    pos_i = torch.arange(total, device=device).unsqueeze(1)  # [total, 1]
    pos_j = torch.arange(total, device=device).unsqueeze(0)  # [1, total]
    dist = (pos_i - pos_j).abs()                              # [total, total]
    bias = -slopes.view(-1, 1, 1) * dist.unsqueeze(0)        # [n_head, total, total]
    return bias.unsqueeze(0)                                   # [1, n_head, total, total]


class NoPEGQA(GQA):
    """NoPE — GQA 去掉 RoPE，只走 QK-Norm。"""

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm:
            Q = self.q_norm(Q)
        if self.k_norm:
            K = self.k_norm(K)

        # 跳过 apply_rope — NoPE 的核心区别

        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        current_kv = (K, V)

        if self.use_flash_attn and past_kv is None:
            K_fa = repeat_kv(K, self.num_groups).contiguous()
            V_fa = repeat_kv(V, self.num_groups).contiguous()
            output = F.scaled_dot_product_attention(
                Q, K_fa, V_fa, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = self.W_o(output)
            return output, None, current_kv
        else:
            K = repeat_kv(K, self.num_groups)
            V = repeat_kv(V, self.num_groups)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
                query_keep = torch.isfinite(mask).any(dim=-1, keepdim=True)
                scores = scores.masked_fill(~query_keep, 0.0)
            attn_weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~query_keep, 0.0)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = self.W_o(output)
            return output, attn_weights, current_kv


class AliBiGQA(GQA):
    """ALiBi — GQA 去掉 RoPE，改加线性位置偏置。"""

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        offset = past_kv[0].size(2) if past_kv is not None else 0

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm:
            Q = self.q_norm(Q)
        if self.k_norm:
            K = self.k_norm(K)

        # 跳过 RoPE

        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        current_kv = (K, V)

        K = repeat_kv(K, self.num_groups)
        V = repeat_kv(V, self.num_groups)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # ALiBi bias = -slope * |i-j|，截取当前查询范围
        total = offset + seq_len
        alibi_bias = build_alibi_bias(seq_len, self.num_heads, x.device, offset)
        scores = scores + alibi_bias[:, :, offset:, :total]

        if mask is not None:
            scores = scores + mask
            query_keep = torch.isfinite(mask).any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~query_keep, 0.0)
        attn_weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~query_keep, 0.0)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)
        return output, attn_weights, current_kv


# Sliding window: 单层只关注前 window_size 个 token；堆叠 L 层后感受野 = L × W。
class SlidingWindowGQA(GQA):
    """Sliding Window GQA — Mistral / Mixtral 的核心 attention 变体。

    每个 token 只关注前 window_size 个 token（叠加在 causal mask 之上）。
    window_size 内是标准 GQA，window_size 外权重强制为 0。
    """

    def __init__(
        self, d_model: int, num_heads: int, num_kv_heads: int,
        dropout: float, use_flash_attn: bool = False,
        window_size: int = 4096, **kwargs,
    ) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
                         dropout=dropout, use_flash_attn=use_flash_attn, **kwargs)
        self.window_size = window_size

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
        if mask is not None:
            _, seq_len, _ = x.shape
            offset = past_kv[0].size(2) if past_kv is not None else 0
            total = offset + seq_len
            q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
            k_idx = torch.arange(total, device=x.device).unsqueeze(0)
            window_mask = torch.where(
                q_idx + offset - k_idx > self.window_size,
                float("-inf"), 0.0,
            )
            mask = mask + window_mask.unsqueeze(0).unsqueeze(0)

        return super().forward(x, rope_cos, rope_sin, mask, past_kv)
