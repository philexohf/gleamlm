"""GleamLM Decoder-only 模型实现"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from gleamlm.types import PastKeyValue, PastKeyValueList


class RMSNorm(nn.Module):
    """RMS 归一化：x / sqrt(mean(x²) + ε) * γ"""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x.float() / rms).to(dtype)
        return x * self.weight


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 频率基 (cos/sin)"""
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 施加旋转位置编码。cos/sin 长度必须 >= offset+seq_len，由调用方负责缓存扩展。"""
    seq_len = xq.size(2)
    cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    xq_out = xq * cos + _rotate_half(xq) * sin
    xk_out = xk * cos + _rotate_half(xk) * sin
    return xq_out, xk_out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将前一半和后一半维度配对旋转：dim k 与 dim k+d/2 互换并取负"""
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:]
    return torch.cat([-x2, x1], dim=-1)


class GroupedQueryAttention(nn.Module):
    """GQA + QK-Norm + Flash Attention（可选）"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.use_flash_attn = use_flash_attn
        self.attn_dropout = dropout

        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()  # type: ignore[assignment]
            self.k_norm = nn.Identity()  # type: ignore[assignment]

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _repeat_kv(self, kv: torch.Tensor, num_groups: int) -> torch.Tensor:
        batch, kv_heads, seq_len, head_dim = kv.shape
        kv = kv.unsqueeze(2).expand(batch, kv_heads, num_groups, seq_len, head_dim)
        return kv.reshape(batch, kv_heads * num_groups, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: PastKeyValue | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PastKeyValue]:
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        offset = past_kv[0].size(2) if past_kv is not None else 0
        Q, K = apply_rotary_emb(Q, K, rope_cos, rope_sin, offset)

        # TODO: replace torch.cat with pre-allocated KV buffer for inference
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        current_kv = (K, V)

        if self.use_flash_attn and past_kv is None:
            K_fa = K.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            K_fa = K_fa.reshape(
                batch_size, self.num_kv_heads * self.num_groups, -1, self.head_dim
            ).contiguous()
            V_fa = V.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            V_fa = V_fa.reshape(
                batch_size, self.num_kv_heads * self.num_groups, -1, self.head_dim
            ).contiguous()

            output = F.scaled_dot_product_attention(
                Q,
                K_fa,
                V_fa,
                is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = self.W_o(output)
            return output, None, current_kv
        else:
            K = self._repeat_kv(K, self.num_groups)
            V = self._repeat_kv(V, self.num_groups)

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = self.W_o(output)
            return output, attn_weights, current_kv


class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.dropout(self.W_down(gate * up))  # type: ignore[no-any-return]


class DecoderBlock(nn.Module):
    """Decoder 层：RMSNorm → GQA → +残差 → RMSNorm → SwiGLU → +残差"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, dropout, use_flash_attn, use_qk_norm
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: PastKeyValue | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue]:
        residual = x
        x = self.attn_norm(x)
        attn_out, _, current_kv = self.attn(x, rope_cos, rope_sin, mask, past_kv)
        x = residual + attn_out
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x, current_kv


class GleamLMModel(nn.Module):
    """V4 Deep-Narrow 架构。use_flash_attn=True 启用 PyTorch Flash Attention"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        tie_weights: bool = True,
        use_flash_attn: bool = False,
        use_gradient_checkpointing: bool = False,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.head_dim = d_model // num_heads
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.rope_max_len = max_seq_len * 4
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._use_flash_attn = use_flash_attn

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, num_heads, num_kv_heads, d_ff, dropout, use_flash_attn, use_qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len * 4)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.d_model**-0.5)

        for _name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module is self.lm_head:
                    continue
                fan_in = module.weight.size(1)
                nn.init.normal_(module.weight, mean=0.0, std=fan_in**-0.5)

        if self.lm_head.weight is not self.token_embed.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _create_causal_mask(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> torch.Tensor:
        """创建因果注意力掩码。offset > 0 时处理 KV cache 场景下的前文偏移。"""
        total = offset + seq_len
        mask = torch.triu(
            torch.full((seq_len, total), float("-inf"), device=device), diagonal=offset + 1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self, input_ids: torch.Tensor, past_kv_list: PastKeyValueList | None = None
    ) -> tuple[torch.Tensor, PastKeyValueList]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)
        x = self.emb_dropout(x)

        if past_kv_list is not None:
            if not isinstance(past_kv_list, list):
                raise ValueError(f"past_kv_list must be a list, got {type(past_kv_list)}")
            if not past_kv_list:
                offset = 0
            else:
                if len(past_kv_list) != self.num_layers:
                    raise ValueError(
                        f"past_kv_list length ({len(past_kv_list)}) != num_layers ({self.num_layers})"
                    )
                offset = past_kv_list[0][0].size(2)
        else:
            offset = 0

        total_len = offset + seq_len
        if total_len > self.rope_cos.size(0):  # type: ignore[operator]
            raise ValueError(
                f"Sequence length {total_len} exceeds pre-allocated RoPE cache "
                f"({self.rope_cos.size(0)}). Increase max_seq_len in config or "  # type: ignore[operator]
                f"set a larger multiplier in GleamLMModel.__init__."
            )

        attn_mask = self._create_causal_mask(
            seq_len,
            device,
            offset=offset,
        )

        new_kv_list: PastKeyValueList = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            if self.training and self.use_gradient_checkpointing:
                x, current_kv = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    self.rope_cos,
                    self.rope_sin,
                    attn_mask,
                    past_kv,
                    use_reentrant=False,
                )
            else:
                x, current_kv = layer(x, self.rope_cos, self.rope_sin, attn_mask, past_kv)
            new_kv_list.append(current_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_list

    def get_num_params(self) -> tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        total_buffers = sum(b.numel() for b in self.buffers())
        return total_params, total_params + total_buffers
