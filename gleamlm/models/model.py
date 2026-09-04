"""GleamLM Decoder-only Transformer model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from gleamlm.types import PastKeyValue, PastKeyValueList


# RMSNorm 去掉均值归零和 bias：Pre-Norm 残差结构会自然吸收均值偏移，去均值无害。
class RMSNorm(nn.Module):
    """RMS layer normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        # RMS 分母是 √E[x²]，不是标准差（不减去均值）
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        # weight 默认 fp32，最后回 cast 到输入 dtype 保证输出类型一致

        return (x * self.weight).to(x_dtype)


# RoPE 通过旋转注入位置，使 attention score 只依赖相对位置 m-n。
# 用 real ops (x*cos + rotate_half(x)*sin) 而非复数：PyTorch complex 精度损失大。
# YaRN 外推: 高频维度保持原旋转速度（外推），低频维度线性插值，中间按 ramp 过渡。
def _find_correction_dim(
    original_max_seq_len: int, dim: int, base: float, num_rotations: float
) -> float:
    """YaRN: compute dimension index from a rotation count threshold."""
    return (dim * math.log(original_max_seq_len / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _compute_yarn_freqs(
    head_dim: int,
    base: float = 10000.0,
    factor: float = 1.0,
    original_max_seq_len: int = 2048,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """YaRN frequency blend: interpolate low-freq dims, extrapolate high-freq dims."""
    dim = head_dim // 2
    # 两套频率: 外推保持原始频率，插值除以 factor 拉长
    pos_freqs = base ** (torch.arange(0, dim, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    # 按波长阈值反推维度边界，确定外推/插值的分界
    low = max(0.0, _find_correction_dim(original_max_seq_len, dim, base, beta_fast))
    high = min(float(dim - 1), _find_correction_dim(original_max_seq_len, dim, base, beta_slow))

    # ramp: 低频=0（纯插值），高频=1（纯外推），中间线性过渡
    linear_ramp = (torch.arange(dim, dtype=torch.float) - low) / (high - low + 1e-8)
    ramp = torch.clamp(linear_ramp, 0.0, 1.0)

    inv_freq = inv_freq_extrapolation * ramp + inv_freq_interpolation * (1.0 - ramp)
    return inv_freq


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    rope_scale: float = 1.0,
    rope_factor: float = 8.0,
    original_max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin cache.

    rope_scale = 1.0  → standard RoPE
    rope_scale > 1.0  → Linear PI (t /= scale)
    rope_scale > 1.0 + original_max_seq_len < max_seq_len → YaRN blend
    """
    if rope_scale > 1.0 and original_max_seq_len is not None and original_max_seq_len < max_seq_len:
        # YaRN 模式：按维度 blend 频率，不做全局 position 缩放
        freq = _compute_yarn_freqs(
            head_dim, base=base, factor=rope_scale,
            original_max_seq_len=original_max_seq_len,
        )
        t = torch.arange(max_seq_len, dtype=torch.float)
    else:
        # 标准 RoPE / Linear PI：频率按维度递减，位置除以 rope_scale
        freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float) / rope_scale

    freqs = torch.outer(t, freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # 每对维度共享同一频率
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.size(2)
    # offset: KV cache 已有长度，RoPE 位置从 offset 继续

    cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


# GQA: num_kv_heads 组 Q head 共享一组 K/V；num_kv=1 退化为 MQA。
def repeat_kv(kv: torch.Tensor, num_groups: int) -> torch.Tensor:
    batch, kv_heads, seq_len, head_dim = kv.shape
    kv = kv.unsqueeze(2).expand(batch, kv_heads, num_groups, seq_len, head_dim)
    return kv.reshape(batch, kv_heads * num_groups, seq_len, head_dim)


class GQA(nn.Module):
    """Grouped Query Attention with QK-Norm and Flash Attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.use_flash_attn = use_flash_attn

        # Q 全量投影，K/V 缩减（每 group 共享一组）
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        # QK-Norm: 限制 Q/K 范数，防止 logits 爆炸导致 softmax 坍缩、梯度消失
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # mask(默认 fp32) 需对齐 Q 的 dtype：AMP/FSDP bf16 参数下 Q 为 bf16，
        # SDPA/math 路径对 additive mask 有与 query 同 dtype 的要求
        if mask is not None:
            mask = mask.to(dtype=Q.dtype)

        offset = past_kv[0].size(2) if past_kv is not None else 0
        Q, K = apply_rope(Q, K, rope_cos, rope_sin, offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        current_kv = (K, V)

        # Flash Attention 只在无 KV cache 时用：is_causal 要求严格因果，
        # 有前缀时退化为原生实现手动加 mask
        if self.use_flash_attn and past_kv is None:
            # Flash attn 不支持 GQA，需先 expand K/V

            K_fa = repeat_kv(K, self.num_groups).contiguous()
            V_fa = repeat_kv(V, self.num_groups).contiguous()

            # mask 恒由上层创建 (因果 + padding 合并)；必须显式传入，
            # 否则 sliding window 与 padding 约束被静默丢弃。
            if mask is None:
                output = F.scaled_dot_product_attention(
                    Q, K_fa, V_fa,
                    is_causal=True,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            else:
                output = F.scaled_dot_product_attention(
                    Q, K_fa, V_fa,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = self.W_o(output)
            return output, None, current_kv
        else:
            K = repeat_kv(K, self.num_groups)
            V = repeat_kv(V, self.num_groups)

            # 1/√d_k 将点积方差规约到 1，防止 softmax 饱和
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
                # pad query 行整行 -inf → softmax 得 nan（反向梯度也 nan）；
                # 先把 pad 行 scores 置 0（保持有限），softmax 后乘 0 丢弃该行
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


# SwiGLU: 三权重 gated FFN；d_ff = 8/3 * d_model 使其参数量与 4× ReLU FFN 相当
class MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, **kwargs) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.dropout(self.W_down(gate * up)), None


# MoE: router 按 top-k 激活部分 expert，FLOPs 不变但参数量增加。
# aux_loss = E * Σ(f_i * P_i) 防止 router 把所有 token 路由到同一个 expert。
# MLP/MoE 统一返回 (output, aux_loss) 元组，DecoderLayer 解包后经 model 汇总。
class MoE(nn.Module):
    """Mixture of Experts — sparse FFN with top-k routing and load-balance aux loss."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        num_experts: int = 8,
        top_k: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLP(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len, d_model]

        batch_size, seq_len, d_model = x.shape

        router_logits = self.router(x)
        top_k_vals, top_k_idx = router_logits.topk(self.top_k, dim=-1)
        probs = F.softmax(top_k_vals.float(), dim=-1).to(x.dtype)

        output = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            mask = (top_k_idx == e_id).any(dim=-1)
            if not mask.any():
                continue
            expert_in = x[mask]
            expert_out, _ = expert(expert_in)
            weight = probs[(top_k_idx == e_id)]
            output[mask] += expert_out * weight.unsqueeze(-1)

        tokens_per_expert = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )
        for e_id in range(self.num_experts):
            tokens_per_expert[e_id] = (top_k_idx == e_id).any(dim=-1).sum()
        f_i = tokens_per_expert / max(batch_size * seq_len, 1)
        P_i = router_logits.softmax(dim=-1).mean(dim=(0, 1))
        aux_loss = self.num_experts * (f_i * P_i).sum()

        return output, aux_loss


# Pre-Norm: 梯度直接经残差通路回传，不穿过子层，深层不衰减；
# 本实现为标准串行 (x + Attn(Norm(x))，x + FFN(Norm(x)))。
class DecoderLayer(nn.Module):
    """Pre-LN decoder layer with pluggable attention, FFN, and normalization variants."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        # 可注入子类，方便实验不同 attention / FFN 变体

        attn_variant: type = GQA,
        ffn_variant: type = MLP,
        norm_variant: type = RMSNorm,
        # 兼容参数 — MoE / attn 变体各取所需，其余通过 **kwargs 静默 absorb

        num_experts: int = 8,
        top_k: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_norm = norm_variant(d_model)
        self.attn = attn_variant(
            d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
            dropout=dropout, use_flash_attn=use_flash_attn,
            **kwargs,
        )
        self.ffn_norm = norm_variant(d_model)
        self.ffn = ffn_variant(
            d_model=d_model, d_ff=d_ff, dropout=dropout,
            num_experts=num_experts, top_k=top_k, **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x = self.attn_norm(x)
        attn_out, _, current_kv = self.attn(x, rope_cos, rope_sin, mask, past_kv)
        x = residual + attn_out

        residual = x
        x = self.ffn_norm(x)
        ffn_out, self.aux_loss = self.ffn(x)
        x = residual + ffn_out

        return x, current_kv


class GleamLMModel(nn.Module):
    """Deep-Narrow Decoder-only Transformer."""

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
        attn_variant: type = GQA,
        ffn_variant: type = MLP,
        norm_variant: type = RMSNorm,
        num_experts: int = 8,
        top_k: int = 2,
        # YaRN 长度外推参数

        rope_scale: float = 1.0,
        rope_factor: float = 8.0,
        rope_theta: float = 10000.0,
        # 按层混合变体配置

        layer_configs: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.head_dim = d_model // num_heads
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.rope_scale = rope_scale
        self.rope_factor = rope_factor
        self.rope_theta = rope_theta
        # YaRN 原始上下文长度: 不做除以 rope_scale — 这就是模型训练时的基础长度

        self.rope_original_max_seq_len = max_seq_len
        # RoPE 缓存 = 基础长度 × 扩展因子 × 缓冲乘数
        # 例: max_seq_len=4096, rope_scale=8.0, rope_factor=1.0 → 32768

        self.rope_max_len = int(max_seq_len * max(rope_scale, 1.0) * rope_factor)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._use_flash_attn = use_flash_attn

        # Token Embedding: padding_idx 让 pad token 在反向传播中梯度为 0

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_dropout = nn.Dropout(dropout)

        # 逐层配置: 全局默认 → layer_configs 覆写（attn/ffn/norm 变体与超参）
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            l_attn = attn_variant
            l_ffn = ffn_variant
            l_norm = norm_variant
            l_extra: dict = {"num_experts": num_experts, "top_k": top_k}

            if layer_configs is not None and layer_idx < len(layer_configs):
                cfg = layer_configs[layer_idx]
                l_attn = cfg.get("attn_variant", l_attn)
                l_ffn = cfg.get("ffn_variant", l_ffn)
                l_norm = cfg.get("norm_variant", l_norm)
                # l_extra 逐键覆写，避免用 cfg 整体替换丢失其他默认键

                for k in ("num_experts", "top_k", "window_size"):
                    if k in cfg:
                        l_extra[k] = cfg[k]

            self.layers.append(DecoderLayer(
                d_model, num_heads, num_kv_heads, d_ff, dropout,
                use_flash_attn,
                attn_variant=l_attn, ffn_variant=l_ffn, norm_variant=l_norm,
                **l_extra,
            ))

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: 输入映射与输出映射共享权重，节省 vocab_size*d_model 参数
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        # RoPE / YaRN 预计算 (persistent=False: 不写入 state_dict，from_pretrained 时重算)

        cos, sin = precompute_freqs_cis(
            self.head_dim, self.rope_max_len,
            base=self.rope_theta, rope_scale=self.rope_scale,
            rope_factor=self.rope_factor,
            original_max_seq_len=self.rope_original_max_seq_len,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _recompute_rope_cache(self) -> None:
        """Recompute non-persistent RoPE buffers from stored parameters.

        Needed after from_pretrained because the model is created on meta device
        and non-persistent buffers are lost during materialization.
        """
        # 缓冲必须建在参数所在设备上，否则 GPU 部署时 forward 报 device mismatch
        device = next(self.parameters()).device
        cos, sin = precompute_freqs_cis(
            head_dim=self.head_dim,
            max_seq_len=self.rope_max_len,
            base=self.rope_theta,
            rope_scale=self.rope_scale,
            rope_factor=self.rope_factor,
            original_max_seq_len=self.rope_original_max_seq_len,
        )
        self.register_buffer("rope_cos", cos.to(device), persistent=False)
        self.register_buffer("rope_sin", sin.to(device), persistent=False)

    # 初始化使各层方差保持 O(1): Embedding/Linear 用 1/√d，LM Head 用小标准差
    # 防止初始 logits 过大导致 softmax 饱和
    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.d_model**-0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.normal_(module.weight, mean=0.0, std=module.weight.size(1)**-0.5)
        if self.lm_head.weight is not self.token_embed.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # 因果掩码 [1, 1, S, total_len]: 上三角 -inf；offset>0 时已有前缀不 mask
    def _create_causal_mask(
        self, seq_len: int, offset: int = 0, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        total = offset + seq_len
        # diagonal=offset+1: 让前 offset 列全 0 (已有 KV)，当前序列内上三角 -inf

        mask = torch.triu(
            torch.full((seq_len, total), float("-inf"), device=device), diagonal=offset + 1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv_list: PastKeyValueList | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValueList, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        offset = past_kv_list[0][0].size(2) if past_kv_list else 0

        total_len = offset + seq_len
        if total_len > self.rope_cos.size(0):  # type: ignore[operator]
            raise ValueError(
                f"Sequence length {total_len} exceeds pre-allocated RoPE cache "
                f"({self.rope_cos.size(0)}). Increase max_seq_len in config or "  # type: ignore[operator]
                f"set a larger multiplier in GleamLMModel.__init__."
            )

        attn_mask = self._create_causal_mask(seq_len, offset=offset, device=device)

        # HF 的 attention_mask (1=keep 0=pad)，生成阶段可能覆盖未来 token，截断到 total_len
        if attention_mask is not None:
            pad = attention_mask.to(dtype=attn_mask.dtype, device=device)
            total = offset + seq_len
            if pad.size(1) > total:
                pad = pad[:, :total]
            pad = pad[:, None, None, :]
            attn_mask = attn_mask.masked_fill(pad == 0, float("-inf"))

        new_kv_list: PastKeyValueList = []
        aux_loss_total = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list else None
            # 反向时重算激活值换显存，代价是额外一次前向 (~20% 训练时间)
            if self.training and self.use_gradient_checkpointing:
                x, current_kv = torch.utils.checkpoint.checkpoint(
                    layer,
                    x, self.rope_cos, self.rope_sin, attn_mask, past_kv,
                    use_reentrant=False,
                )
            else:
                x, current_kv = layer(x, self.rope_cos, self.rope_sin, attn_mask, past_kv)
            new_kv_list.append(current_kv)
            aux = layer.aux_loss
            if aux is not None:
                aux_loss_total = aux_loss_total + aux

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        if not use_cache:
            new_kv_list = []

        if output_hidden_states:
            return logits, new_kv_list, aux_loss_total, hidden

        return logits, new_kv_list, aux_loss_total, None

    def get_num_params(self) -> tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        total_buffers = sum(b.numel() for b in self.buffers())
        return total_params, total_params + total_buffers


# Backward-compatible aliases
apply_rotary_emb = apply_rope
_rotate_half = rotate_half
GroupedQueryAttention = GQA
SwiGLUFFN = MLP
DecoderBlock = DecoderLayer
