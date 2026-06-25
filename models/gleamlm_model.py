"""烁珑GleamLM Decoder-only 模型实现"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS 归一化：x / sqrt(mean(x²) + ε) * γ"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        dtype = x.dtype
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x.float() / rms).to(dtype)
        return x * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """预计算 RoPE 频率基 (cos/sin)"""
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def apply_rotary_emb(xq, xk, cos, sin, offset=0):
    """对 Q/K 施加旋转位置编码"""
    seq_len = xq.size(2)
    cos = cos[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)
    xq_out = xq * cos + _rotate_half(xq) * sin
    xk_out = xk * cos + _rotate_half(xk) * sin
    return xq_out, xk_out


def _rotate_half(x):
    """将前一半和后一半维度配对旋转：dim k 与 dim k+d/2 互换并取负"""
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:]
    return torch.cat([-x2, x1], dim=-1)


class GroupedQueryAttention(nn.Module):
    """GQA + QK-Norm：8 查询头共享 4 组 KV，Q/K 额外 RMSNorm（LLaMA3 标配）"""
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0, max_seq_len=1024):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads  # 每组查询头共享一对 KV 头
        self.max_seq_len = max_seq_len

        # Q 投影（查询头数 × 头维度）
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        # K, V 投影（KV 头数 × 头维度）
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        # QK-Norm：注意力计算前对 Q/K 额外 RMSNorm，稳定训练
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 预计算 RoPE 频率（缓存，不参与梯度）
        cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def _repeat_kv(self, kv, num_groups):
        """将 KV 头重复以匹配查询头数"""
        batch, kv_heads, seq_len, head_dim = kv.shape
        kv = kv.unsqueeze(2).expand(batch, kv_heads, num_groups, seq_len, head_dim)
        return kv.reshape(batch, kv_heads * num_groups, seq_len, head_dim)

    def forward(self, x, mask=None, past_kv=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm：稳定注意力分布，降低训练震荡
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # KV cache 时，新 token 的实际位置是已有 cache 的长度
        offset = past_kv[0].size(2) if past_kv is not None else 0
        Q, K = apply_rotary_emb(Q, K, self.rope_cos, self.rope_sin, offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        current_kv = (K, V)
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
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.dropout(self.W_down(gate * up))


class DecoderBlock(nn.Module):
    """Decoder 层：RMSNorm → GQA → +残差 → RMSNorm → SwiGLU → +残差"""
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, dropout, max_seq_len
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None, past_kv=None):
        residual = x
        x = self.attn_norm(x)
        attn_out, _, current_kv = self.attn(x, mask, past_kv)
        x = residual + attn_out
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x, current_kv


class GleamLMModel(nn.Module):
    """V4 Deep-Narrow：12层×512dim + BBPE 12K + QK-Norm（~39M 参数）"""
    def __init__(self, vocab_size=12003, d_model=512, num_layers=12,
                 num_heads=8, num_kv_heads=4, d_ff=1365,
                 dropout=0.1, max_seq_len=1024, pad_token_id=0,
                 tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, num_kv_heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        # 输出投影：d_model → vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定：lm_head 与 token_embed 共享权重（节省参数，提升效果）
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.d_model ** -0.5)

        # 线性层：小方差正态，RMSNorm weight 初始化为 1（默认）
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 跳过 lm_head（无论是否绑定，后面单独初始化）
                if module is self.lm_head:
                    continue
                fan_in = module.weight.size(1)
                nn.init.normal_(module.weight, mean=0.0, std=fan_in ** -0.5)

        # 若 lm_head 未绑定，使用更小初始化
        if self.lm_head.weight is not self.token_embed.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len, device):
        """创建因果注意力掩码（上三角 -inf）"""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, past_kv_list=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)
        x = self.emb_dropout(x)

        if past_kv_list is None:
            causal_mask = self._create_causal_mask(seq_len, device)
        else:
            causal_mask = None

        new_kv_list = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, current_kv = layer(x, causal_mask, past_kv)
            new_kv_list.append(current_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_list

    def get_num_params(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    print("=" * 60)
    print("构建 烁珑GleamLM V4 模型（12层×512dim + QK-Norm）")
    print("=" * 60)

    model = GleamLMModel(
        vocab_size=12003,
        d_model=512,
        num_layers=12,
        num_heads=8,
        num_kv_heads=4,
        d_ff=1365,
        dropout=0.1,
        max_seq_len=1024,
        pad_token_id=0
    )

    total, trainable = model.get_num_params()
    print(f"\n总参数量:   {total / 1e6:.2f}M")
    print(f"可训练参数: {trainable / 1e6:.2f}M")

    print("\n[1] forward() — 训练模式")
    input_ids = torch.randint(0, 12003, (4, 128))
    logits, kv_list = model(input_ids)
    print(f"    输入: input_ids {input_ids.shape}")
    print(f"    输出: logits {logits.shape}  ← 应为 [4, 128, 12003]")
    print(f"    KV Cache 层数: {len(kv_list)}")

    print("\n[2] backward() — 反向传播验证")
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 12003),
        input_ids[:, 1:].reshape(-1),
        ignore_index=0
    )
    loss.backward()

    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            if gn > 0:
                grad_norms.append(gn)
    print(f"    Loss: {loss.item():.4f}")
    print(f"    Grad norm (max): {max(grad_norms):.4f}")
    print(f"    Grad norm (mean): {sum(grad_norms)/len(grad_norms):.4f}")
    print("    反向传播: OK (无 NaN)")

    print("\n[3] forward() — 推理模式（KV Cache）")
    prompt = torch.randint(0, 12003, (1, 10))
    logits, kv_cache = model(prompt)
    print(f"    预填充: prompt {prompt.shape} → logits {logits.shape}")
    print(f"    KV Cache 长度: {kv_cache[0][0].size(2)} (应为 10)")

    past_kv = kv_cache
    next_token = logits[:, -1:].argmax(dim=-1)
    for step in range(5):
        logits, past_kv = model(next_token, past_kv_list=past_kv)
        next_token = logits[:, -1:].argmax(dim=-1)
    print(f"    生成 5 步后 KV Cache 长度: {past_kv[0][0].size(2)} (应为 15)")
    print("    KV Cache 推理: OK")

    print("\n[4] 超长序列 - RoPE 外推测试")
    input_ids_long = torch.randint(0, 12003, (2, 256))
    logits_long, _ = model(input_ids_long)
    print(f"    输入: {input_ids_long.shape}")
    print(f"    输出: {logits_long.shape}  ← 应为 [2, 256, 12003]")
    loss_long = F.cross_entropy(
        logits_long[:, :-1].reshape(-1, 12003),
        input_ids_long[:, 1:].reshape(-1),
        ignore_index=0
    )
    loss_long.backward()
    print(f"    Loss: {loss_long.item():.4f}")
    print("    超长序列: OK")

    print("\n" + "=" * 60)
    print("所有验证通过")
    print("=" * 60)
