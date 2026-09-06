from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from gleamlm.models.model import RMSNorm


# Mamba: 用 SSM 替代 attention，推理每 token O(1)（只维护状态 h_t）。
# 选择性扫描让 A/B/C/Δ 依赖输入，相当于给 SSM 加可学习的门控。
def selective_scan(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """选择性扫描 — SSM 的核心计算。

    离散化 (zero-order hold):
      A_bar = exp(Δ * A)
      B_bar = Δ * B   (简化)
      h_t = A_bar * h_{t-1} + B_bar * x_t
      y_t = C_t * h_t

    形状约定:
      x:     [B, S, d_inner]    每个通道一个 SSM
      A:     [d_inner, d_state] 每个通道独立的 state 转移矩阵
      B/C:   [B, S, d_state]    输入相关的选择向量
      delta: [B, S, d_inner]    每个通道独立的步长
    返回:    [B, S, d_inner]

    本实现用显式循环（无并行 scan）。
    """
    batch_size, seq_len, d_inner = x.shape
    d_state = A.size(-1)
    dtype = x.dtype

    delta = F.softplus(delta)  # [B, S, d_inner]
    A_bar = torch.exp(delta.unsqueeze(-1) * A)  # [B, S, d_inner, d_state]
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)  # [B, S, d_inner, d_state]
    u = B_bar * x.unsqueeze(-1)  # [B, S, d_inner, d_state]

    # 显式循环 scan（非并行版本）
    h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=dtype)
    ys = []
    for t in range(seq_len):
        h = A_bar[:, t] * h + u[:, t]  # [B, d_inner, d_state]
        y = (C[:, t].unsqueeze(-2) * h).sum(dim=-1)  # [B, d_inner]
        ys.append(y)

    return torch.stack(ys, dim=1)  # [B, S, d_inner]


class MambaBlock(nn.Module):
    """
    Mamba 层 — conv1d + SiLU gate + selective SSM。

    架构:
      x → RMSNorm → conv1d → SiLU → SSM → gate(out) → output

    SSM 参数:
      d_state = d_model 的 "state size" — 通常 16-64
      d_conv = 卷积核大小 — 通常 4
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        use_gate: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        d_inner = int(d_model * expand_factor)
        self.use_gate = use_gate

        # 输入投影: x → [x', z]，z 为控制门
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # 因果 padding: 只看过去
            groups=d_inner,  # depthwise conv
            bias=False,
        )

        # B/C/Δ 由 x' 投影得到（selective）；A 固定，log 参数化保证 exp 后为正
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_inner * d_state + 1, dtype=torch.float)).view(
                d_inner, d_state
            )
        )
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        """
        MambaBlock 前向。兼容 DecoderLayer 内嵌的 attn 槽位契约（与 GQA 同为 3 元组返回）。
        返回 (output, None, None) — 后两位替代 attn_weights / current_kv 占位。
        """
        batch_size, seq_len, _ = x.shape

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # [B, S, d_inner] each

        x_conv = x_inner.transpose(1, 2)  # [B, d_inner, S]
        x_conv = self.conv1d(x_conv)[..., :seq_len]  # 因果裁切
        x_conv = F.silu(x_conv)  # [B, d_inner, S]
        x_conv = x_conv.transpose(1, 2)  # [B, S, d_inner]

        # SSM: A 取负保证衰减，B/C/Δ 由投影得到
        A = -torch.exp(self.A_log)  # [d_inner, d_state] (负 → 衰减)
        bc_delta = self.x_proj(x_conv)  # [B, S, d_state*2 + d_inner]
        B, C, delta = bc_delta.split([self.d_state, self.d_state, self.d_inner], dim=-1)

        y_ssm = selective_scan(x_conv, A, B, C, delta)  # [B, S, d_inner]

        y = y_ssm + self.D * x_conv

        if self.use_gate:
            y = y * F.silu(z)

        y = self.out_proj(y)
        output = residual + y

        return output, None, None
