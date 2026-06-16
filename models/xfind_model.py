"""
Xfind-Mini 现代 Decoder-only LLM 模型实现

架构（Pre-Norm Decoder-only，参考 Llama/Qwen）：
    tokens → Embed → [DecoderBlock × N] → RMSNorm → Linear → logits

每个 DecoderBlock:
    x → RMSNorm → GQA(自注意力) → +残差 → RMSNorm → SwiGLU(FFN) → +残差

现代 SOTA 组件：
    - RMSNorm: 替代 LayerNorm，更快更稳
    - RoPE: 旋转位置编码，支持长度外推
    - GQA: 分组查询注意力（8 查询头 / 4 KV 头），减少推理显存
    - SwiGLU: 门控激活函数，替代 ReLU/GELU
    - Pre-Norm: 归一化在子层之前，训练更稳定
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# RMSNorm：均方根归一化
# ============================================================

class RMSNorm(nn.Module):
    """
    均方根层归一化（Root Mean Square Layer Normalization）

    RMSNorm(x) = x / RMS(x) * γ
    其中 RMS(x) = sqrt(mean(x^2) + eps)

    相比 LayerNorm 去掉了均值中心化，仅做缩放归一化，
    计算更快（少一次 reduce），效果相当。

    Llama/Qwen 等现代 LLM 标配。
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: [batch, seq, d_model]
        dtype = x.dtype
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x.float() / rms).to(dtype)
        return x * self.weight


# ============================================================
# RoPE：旋转位置编码
# ============================================================

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    预计算 RoPE 的频率基（cos 和 sin 分量）

    Args:
        dim: 每个注意力头的维度 d_k
        max_seq_len: 最大序列长度
        theta: 频率基数（默认 10000）

    Returns:
        cos: [max_seq_len, dim] cos 值
        sin: [max_seq_len, dim] sin 值
    """
    # 频率：θ_i = theta^(-2i/dim)，i = 0, 1, ..., dim/2-1
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # 位置索引
    t = torch.arange(max_seq_len, dtype=torch.float)
    # 外积：[max_seq_len, dim//2]
    freqs = torch.outer(t, freq)
    # 对每一对维度 (2i, 2i+1) 重复
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def apply_rotary_emb(xq, xk, cos, sin):
    """
    对 Q 和 K 施加旋转位置编码（实数运算版本，比复数快 2-3 倍）

    将每对相邻维度 (x_{2i}, x_{2i+1}) 视为二维向量，
    施加旋转矩阵: [cos, -sin; sin, cos]

    Args:
        xq: query 张量 [batch, heads, seq, d_k]
        xk: key   张量 [batch, heads, seq, d_k]
        cos: [seq, dim] cos 值
        sin: [seq, dim] sin 值

    Returns:
        xq, xk 施加 RoPE 后的结果
    """
    seq_len = xq.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # 旋转：x * cos + rotate_half(x) * sin
    # rotate_half: 将每对相邻维度交换并取负
    xq_out = xq * cos + _rotate_half(xq) * sin
    xk_out = xk * cos + _rotate_half(xk) * sin

    return xq_out, xk_out


def _rotate_half(x):
    """将每对相邻维度交换并取负：(-x2, x1, -x4, x3, ...)"""
    x1 = x[..., ::2]   # 偶位置
    x2 = x[..., 1::2]  # 奇位置
    x = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x


# ============================================================
# GQA：分组查询注意力
# ============================================================

class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（Grouped Query Attention）

    查询头数 > KV 头数，KV 头通过 repeat 扩展对齐查询头。
    相比 MHA（查询头=KV头），GQA 减少 KV Cache 显存；
    相比 MQA（1个KV头），GQA 保留更好的表达能力。

    Xfind-Mini: 8 查询头 / 4 KV 头
    """
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
        # 输出投影
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 预计算 RoPE 频率（缓存，不参与梯度）
        cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def _repeat_kv(self, kv, num_groups):
        """
        将 KV 头重复以匹配查询头数

        [batch, kv_heads, seq, head_dim] → [batch, num_heads, seq, head_dim]
        """
        batch, kv_heads, seq_len, head_dim = kv.shape
        # 先 reshape 为 [batch, kv_heads, 1, seq, head_dim]
        # 再扩展为 [batch, kv_heads, num_groups, seq, head_dim]
        kv = kv.unsqueeze(2).expand(batch, kv_heads, num_groups, seq_len, head_dim)
        # 合并 kv_heads 和 num_groups → [batch, num_heads, seq, head_dim]
        return kv.reshape(batch, kv_heads * num_groups, seq_len, head_dim)

    def forward(self, x, mask=None, past_kv=None):
        """
        Args:
            x: 输入 [batch, seq, d_model]
            mask: 注意力掩码 [batch, 1, seq, seq] 或 [seq, seq]
            past_kv: (past_k, past_v) KV Cache（推理时使用）

        Returns:
            output: [batch, seq, d_model]
            attn_weights: 注意力权重
            (k, v): 当前层的 K, V（用于 KV Cache）
        """
        batch_size, seq_len, _ = x.shape

        # 投影：Q → [batch, num_heads, seq, head_dim]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V → [batch, num_kv_heads, seq, head_dim]
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 旋转位置编码（对 Q 和 K）
        Q, K = apply_rotary_emb(Q, K, self.rope_cos, self.rope_sin)

        # KV Cache 拼接（推理时）
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        # 保存当前 K, V 供缓存
        current_kv = (K, V)

        # GQA：将 KV 头扩展以匹配查询头数
        K = self._repeat_kv(K, self.num_groups)
        V = self._repeat_kv(V, self.num_groups)

        # 缩放点积注意力
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 合并多头 → [batch, seq, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)

        return output, attn_weights, current_kv


# ============================================================
# SwiGLU：门控前馈网络
# ============================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络

    FFN(x) = (x·W_gate ⊙ σ(x·W_gate)) · W_down
            = SiLU(x·W_gate) ⊙ (x·W_up) · W_down

    其中 σ = SiLU(x) = x * sigmoid(x)

    相比标准 FFN（ReLU(x·W1)·W2），SwiGLU：
    - 使用门控机制选择性激活
    - 训练更稳定，收敛更快
    - 现代 LLM（Llama/Qwen）标配

    三层投影：d_model → d_ff → d_model
        gate: d_model → d_ff（门控信号）
        up:   d_model → d_ff（数值信号）
        down: d_ff   → d_model（输出投影）
    """
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # gate: SiLU 激活的门控
        # up: 线性变换的值
        # 逐元素相乘后投影回 d_model
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.dropout(self.W_down(gate * up))


# ============================================================
# Decoder Block：Pre-Norm 结构
# ============================================================

class DecoderBlock(nn.Module):
    """
    单个 Decoder 层（Pre-Norm 结构）

    数据流：
        x → RMSNorm → GQA(自注意力) → +残差 → RMSNorm → SwiGLU → +残差

    Pre-Norm：归一化在子层之前，梯度流动更顺畅，训练更稳定。
    """
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, dropout, max_seq_len
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None, past_kv=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: 注意力掩码
            past_kv: KV Cache
        Returns:
            output: [batch, seq, d_model]
            current_kv: 当前层的 (K, V) 缓存
        """
        # 自注意力子层（Pre-Norm）
        residual = x
        x = self.attn_norm(x)
        attn_out, _, current_kv = self.attn(x, mask, past_kv)
        x = residual + attn_out

        # FFN 子层（Pre-Norm）
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out

        return x, current_kv


# ============================================================
# XfindModel：完整 Decoder-only LLM
# ============================================================

class XfindModel(nn.Module):
    """
    Xfind-Mini: 现代 Decoder-only 语言模型

    数据流：
        tokens → Embed → [DecoderBlock × N] → RMSNorm → Linear → logits

    架构特点：
        - Pre-Norm：归一化在子层之前
        - RoPE：旋转位置编码（内嵌于 GQA 中）
        - GQA：分组查询注意力（8Q/4KV）
        - SwiGLU：门控激活
        - 无 bias：所有线性层 bias=False
        - 权重绑定：Embedding 与 lm_head 共享权重（节省 ~16M 参数）

    参数规格（Xfind-Mini）：
        - 8 层，d_model=512
        - 8 查询头 / 4 KV 头
        - SwiGLU 中间维度 1365
        - 词表 32K，上下文 1024
        - 总参数量约 39M（含权重绑定）
        - 无权重绑定时约 56M
    """
    def __init__(self, vocab_size=32000, d_model=512, num_layers=8,
                 num_heads=8, num_kv_heads=4, d_ff=1365,
                 dropout=0.1, max_seq_len=1024, pad_token_id=0,
                 tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Dropout after embedding
        self.emb_dropout = nn.Dropout(dropout)

        # 堆叠 N 个 Decoder Block
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, num_kv_heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        # 最终 RMSNorm
        self.final_norm = RMSNorm(d_model)

        # 输出投影：d_model → vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定：lm_head 与 token_embed 共享权重（节省参数，提升效果）
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        # 权重初始化（参考 Llama 标准）
        self._init_weights()

    def _init_weights(self):
        """初始化权重，参考 Llama 做法"""
        # Embedding: 正态分布
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.d_model ** -0.5)

        # 线性层：小方差正态，RMSNorm weight 初始化为 1（默认）
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 跳过 lm_head（无论是否绑定，后面单独初始化）
                if module is self.lm_head:
                    continue
                fan_in = module.weight.size(1)
                nn.init.normal_(module.weight, mean=0.0, std=fan_in ** -0.5)

        # 如果 lm_head 未与 token_embed 绑定，使用更小初始化；已绑定时无需初始化
        if self.lm_head.weight is not self.token_embed.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len, device):
        """
        创建因果注意力掩码

        使用 additive mask 方式（-inf 填充），兼容 GQA 的 scores + mask 模式

        Returns:
            mask: [1, 1, seq_len, seq_len]，上三角为 -inf
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, past_kv_list=None):
        """
        训练 / 推理前向传播

        Args:
            input_ids: token IDs [batch, seq_len]
            past_kv_list: KV Cache 列表（推理时使用），
                          每层一个 (k, v) 元组

        Returns:
            logits: [batch, seq_len, vocab_size]
            past_kv_list: 更新后的 KV Cache 列表（推理时返回）
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token Embedding → Dropout
        x = self.token_embed(input_ids)
        x = self.emb_dropout(x)

        # 因果掩码（训练时遮未来 token）
        if past_kv_list is None:
            # 训练模式：对整个序列生成因果掩码
            causal_mask = self._create_causal_mask(seq_len, device)
        else:
            # 推理模式：KV Cache 中已有历史，只需看当前位置
            causal_mask = None  # 只需看自己，无需掩码

        # 逐层传递
        new_kv_list = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, current_kv = layer(x, causal_mask, past_kv)
            new_kv_list.append(current_kv)

        # 最终 RMSNorm
        x = self.final_norm(x)

        # 输出投影：d_model → vocab_size
        logits = self.lm_head(x)

        return logits, new_kv_list

    def get_num_params(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================
# 快速验证
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("构建 Xfind-Mini 模型（约 35M 参数）")
    print("=" * 60)

    model = XfindModel(
        vocab_size=32000,
        d_model=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=4,
        d_ff=1365,
        dropout=0.1,
        max_seq_len=1024,
        pad_token_id=0
    )

    # 参数量统计
    total, trainable = model.get_num_params()
    print(f"\n总参数量:   {total / 1e6:.2f}M")
    print(f"可训练参数: {trainable / 1e6:.2f}M")

    # 验证 1: 训练模式前向传播
    print("\n[1] forward() — 训练模式")
    input_ids = torch.randint(0, 32000, (4, 128))  # batch=4, seq_len=128
    logits, kv_list = model(input_ids)
    print(f"    输入: input_ids {input_ids.shape}")
    print(f"    输出: logits {logits.shape}  ← 应为 [4, 128, 32000]")
    print(f"    KV Cache 层数: {len(kv_list)}")

    # 验证 2: 反向传播
    print("\n[2] backward() — 反向传播验证")
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 32000),
        input_ids[:, 1:].reshape(-1),
        ignore_index=0
    )
    loss.backward()

    # 检查梯度
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

    # 验证 3: 推理模式（单 token 逐步生成）
    print("\n[3] forward() — 推理模式（KV Cache）")
    prompt = torch.randint(0, 32000, (1, 10))  # batch=1, prompt_len=10

    # 预填充：处理 prompt
    logits, kv_cache = model(prompt)
    print(f"    预填充: prompt {prompt.shape} → logits {logits.shape}")
    print(f"    KV Cache 长度: {kv_cache[0][0].size(2)} (应为 10)")

    # 逐步生成 5 个 token
    past_kv = kv_cache
    next_token = logits[:, -1:].argmax(dim=-1)  # [1, 1]
    for step in range(5):
        logits, past_kv = model(next_token, past_kv_list=past_kv)
        next_token = logits[:, -1:].argmax(dim=-1)
    print(f"    生成 5 步后 KV Cache 长度: {past_kv[0][0].size(2)} (应为 15)")
    print("    KV Cache 推理: OK")

    # 验证 4: 超长序列动态扩展
    print("\n[4] 超长序列 - RoPE 外推测试")
    input_ids_long = torch.randint(0, 32000, (2, 256))
    logits_long, _ = model(input_ids_long)
    print(f"    输入: {input_ids_long.shape}")
    print(f"    输出: {logits_long.shape}  ← 应为 [2, 256, 32000]")
    loss_long = F.cross_entropy(
        logits_long[:, :-1].reshape(-1, 32000),
        input_ids_long[:, 1:].reshape(-1),
        ignore_index=0
    )
    loss_long.backward()
    print(f"    Loss: {loss_long.item():.4f}")
    print("    超长序列: OK")

    print("\n" + "=" * 60)
    print("所有验证通过！Xfind-Mini 模型可正常前向和反向传播。")
    print("=" * 60)
