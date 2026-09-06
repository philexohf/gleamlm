"""模型 前向/反向/KV Cache/参数量/关键路径 测试"""

import math

import pytest
import torch
import torch.nn.functional as F

from gleamlm.models.attention_variants import AliBiGQA, NoPEGQA, SlidingWindowGQA
from gleamlm.models.model import (
    GQA,
    MLP,
    DecoderLayer,
    GleamLMModel,
    RMSNorm,
    apply_rope,
    precompute_freqs_cis,
    repeat_kv,
    rotate_half,
)

VOCAB_SIZE = 12002


# 基础组件


def test_rms_norm_shape():
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64)
    out = norm(x)
    assert out.shape == (4, 16, 64)


def test_rms_norm_numerics():
    norm = RMSNorm(64, eps=1e-6)
    x = torch.ones(2, 8, 64) * 3.0
    out = norm(x)
    rms = math.sqrt(3.0**2 + 1e-6)
    expected = (3.0 / rms) * 1.0
    assert abs(out[0, 0, 0].item() - expected) < 1e-4


def test_swiglu_ffn_shape():
    ffn = MLP(64, 256)
    x = torch.randn(4, 16, 64)
    out, aux = ffn(x)
    assert aux is None
    assert out.shape == (4, 16, 64)


def test_swiglu_ffn_gate_structure():
    ffn = MLP(64, 256)
    x = torch.randn(2, 8, 64)
    gate = F.silu(ffn.W_gate(x))
    up = ffn.W_up(x)
    assert gate.shape == (2, 8, 256)
    assert up.shape == (2, 8, 256)


def test_precompute_freqs_cis_shape():
    cos, sin = precompute_freqs_cis(64, 128)
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)


def test_apply_rope_shape():
    cos, sin = precompute_freqs_cis(64, 128)
    xq = torch.randn(2, 8, 10, 64)
    xk = torch.randn(2, 4, 10, 64)
    q_out, k_out = apply_rope(xq, xk, cos, sin, offset=0)
    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape


def test_apply_rope_offset():
    cos, sin = precompute_freqs_cis(64, 128)
    xq = torch.randn(1, 8, 5, 64)
    xk = torch.randn(1, 4, 5, 64)
    q_out, k_out = apply_rope(xq, xk, cos, sin, offset=10)
    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape


def test_apply_rope_extension():
    """cos/sin 可以由调用方预扩展后传入"""
    cos, sin = precompute_freqs_cis(64, 30)
    xq = torch.randn(1, 8, 20, 64)
    xk = torch.randn(1, 4, 20, 64)
    q_out, k_out = apply_rope(xq, xk, cos, sin, offset=0)
    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape


def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.equal(out, expected)


# GQA 注意力


def test_gqa_repeat_kv():
    kv = torch.randn(2, 2, 10, 32)
    repeated = repeat_kv(kv, 4)
    assert repeated.shape == (2, 8, 10, 32)


def test_gqa_forward_shape():
    attn = GQA(256, 8, 2)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(4, 16, 256)
    out, weights, kv = attn(x, cos, sin)
    assert out.shape == (4, 16, 256)
    assert weights is not None
    assert weights.shape == (4, 8, 16, 16)
    assert kv[0].shape == (4, 2, 16, 32)
    assert kv[1].shape == (4, 2, 16, 32)


def test_gqa_forward_flash():
    attn = GQA(256, 8, 2, use_flash_attn=True)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(2, 16, 256)
    out, weights, kv = attn(x, cos, sin)
    assert out.shape == (2, 16, 256)
    assert weights is None
    assert kv[0].shape == (2, 2, 16, 32)


def test_gqa_kv_cache():
    attn = GQA(256, 8, 2)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(1, 10, 256)
    _, _, past_kv = attn(x, cos, sin)
    assert past_kv[0].shape[2] == 10
    x2 = torch.randn(1, 1, 256)
    _, _, new_kv = attn(x2, cos, sin, past_kv=past_kv)
    assert new_kv[0].shape[2] == 11


def test_gqa_with_mask():
    attn = GQA(256, 8, 2)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(2, 8, 256)
    mask = torch.full((1, 1, 8, 8), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    out, weights, kv = attn(x, cos, sin, mask=mask)
    assert out.shape == (2, 8, 256)
    assert weights is not None


# DecoderLayer


def test_decoder_layer_shape():
    block = DecoderLayer(256, 8, 2, 682)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(4, 16, 256)
    out, kv = block(x, cos, sin)
    assert out.shape == (4, 16, 256)
    assert kv[0].shape == (4, 2, 16, 32)


def test_decoder_layer_residual():
    block = DecoderLayer(256, 8, 2, 682, dropout=0.0)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(2, 8, 256)
    out, _ = block(x, cos, sin)
    assert not torch.isnan(out).any()


# GleamLMModel


def test_parameter_count(small_model):
    total, trainable = small_model.get_num_params()
    assert 3_000_000 < total < 8_000_000, f"Unexpected: {total / 1e6:.1f}M"


def test_forward_shape(small_model):
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 64))
    logits, kv_list, _, _ = small_model(input_ids)
    assert logits.shape == (4, 64, VOCAB_SIZE)
    assert len(kv_list) == 4


def test_backward_no_nan(small_model):
    small_model.train()
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 64))
    logits, _, _, _ = small_model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB_SIZE),
        input_ids[:, 1:].reshape(-1),
        ignore_index=0,
    )
    loss.backward()
    for name, p in small_model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
    small_model.zero_grad()
    small_model.eval()


def test_kv_cache_forward(small_model):
    prompt = torch.randint(0, VOCAB_SIZE, (1, 10))
    with torch.no_grad():
        logits, kv_cache, _, _ = small_model(prompt)
    assert kv_cache[0][0].size(2) == 10

    past_kv = kv_cache
    next_token = logits[:, -1:].argmax(dim=-1)
    with torch.no_grad():
        for _ in range(5):
            logits, past_kv, _, _ = small_model(next_token, past_kv_list=past_kv)
            next_token = logits[:, -1:].argmax(dim=-1)
    assert past_kv[0][0].size(2) == 15


def test_long_sequence(small_model):
    """序列长度在预分配的 rope_max_len 内，应正常前向"""
    long_input = torch.randint(0, VOCAB_SIZE, (2, 200))
    with torch.no_grad():
        logits, _, _, _ = small_model(long_input)
    assert logits.shape == (2, 200, VOCAB_SIZE)


# 权重绑定 (Weight Tying)


def test_weight_tying_enabled():
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        tie_weights=True,
    )
    assert model.lm_head.weight is model.token_embed.weight


def test_weight_tying_disabled():
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        tie_weights=False,
    )
    assert model.lm_head.weight is not model.token_embed.weight


def test_weight_tying_param_count():
    tied = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        tie_weights=True,
    )
    untied = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        tie_weights=False,
    )
    tied_total, _ = tied.get_num_params()
    untied_total, _ = untied.get_num_params()
    embed_params = 12002 * 256
    assert untied_total - tied_total == embed_params


def test_weight_tying_get_num_params_dedup(small_model):
    total, trainable = small_model.get_num_params()
    raw_total = sum(p.numel() for p in small_model.parameters())
    assert total == raw_total


# Flash Attention 路径


def test_flash_attn_model_forward():
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        use_flash_attn=True,
    )
    model.eval()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 32))
    with torch.no_grad():
        logits, kv_list, _, _ = model(input_ids)
    assert logits.shape == (2, 32, VOCAB_SIZE)
    assert len(kv_list) == 2


def test_flash_attn_kv_cache():
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        use_flash_attn=True,
    )
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, 10))
    with torch.no_grad():
        logits, kv_cache, _, _ = model(prompt)
    assert kv_cache[0][0].size(2) == 10


# QK-Norm


def test_qk_norm_enabled():
    """QK-Norm is always applied."""
    attn = GQA(256, 8, 2)
    x = torch.randn(2, 8, 256)
    assert isinstance(attn.q_norm, RMSNorm)
    assert isinstance(attn.k_norm, RMSNorm)
    Q = attn.W_q(x).view(2, 8, 8, 32).transpose(1, 2)
    K = attn.W_k(x).view(2, 8, 2, 32).transpose(1, 2)
    Q_normed = attn.q_norm(Q)
    K_normed = attn.k_norm(K)
    assert Q_normed.shape == Q.shape
    assert K_normed.shape == K.shape
    assert not torch.isnan(Q_normed).any()
    assert not torch.isnan(K_normed).any()


# output 一致性验证


def test_same_input_same_output():
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        tie_weights=True,
    )
    model.eval()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 16))
    with torch.no_grad():
        logits1, _, _, _ = model(input_ids)
        logits2, _, _, _ = model(input_ids)
    assert torch.allclose(logits1, logits2)


def test_model_device_consistency():
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
    )
    device = next(model.parameters()).device
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 16), device=device)
    with torch.no_grad():
        logits, _, _, _ = model(input_ids)
    assert logits.device == device


# ---- Flash Attention + padding 回归 ----


def test_flash_attn_with_attention_mask():
    """attention_mask + Flash Attn 不应崩溃（Bug 1 回归防护）"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        use_flash_attn=True,
        pad_token_id=0,
    )
    model.eval()
    input_ids = torch.tensor([[5, 3, 2, 0], [1, 4, 0, 0]])
    with torch.no_grad():
        logits, _, _, _ = model(input_ids)
    assert logits.shape == (2, 4, 12002)


# ---- 因果掩码回归测试 ----


def test_causal_mask_matrix_values():
    """_create_causal_mask 返回值：未来位置应被 mask"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
    )
    mask = model._create_causal_mask(seq_len=5, device="cpu")
    assert mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 2, 3] == float("-inf"), "position 2 should not attend to position 3"
    assert mask[0, 0, 2, 2] == 0.0, "position 2 should attend to itself"
    assert mask[0, 0, 4, 0] == 0.0, "position 4 should attend to position 0"


def test_causal_mask_blocks_future_tokens():
    """修改未来位置的 token 不影响前面位置的 logits"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        use_flash_attn=False,
    )
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (1, 5))
    with torch.no_grad():
        logits1, _, _, _ = model(x)
    x_modified = x.clone()
    x_modified[0, 3:] = torch.randint(0, VOCAB_SIZE, (2,))
    with torch.no_grad():
        logits2, _, _, _ = model(x_modified)
    assert torch.allclose(logits1[0, :3], logits2[0, :3], atol=1e-5)


# ---- 梯度流完整性 ----


def test_all_parameters_have_gradient():
    """所有可训练参数在 backward 后必须有非 None 且无 NaN 的梯度"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
    )
    model.train()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 16))
    logits, _, _, _ = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB_SIZE),
        input_ids[:, 1:].reshape(-1),
        ignore_index=0,
    )
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
    model.zero_grad()


# ---- KV Cache 增量一致性 ----


def test_kv_cache_incremental_equivalence():
    """增量解码 final token 的 logits 与全序列前向等价"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        use_flash_attn=False,
    )
    model.eval()
    full_ids = torch.randint(0, VOCAB_SIZE, (1, 5))

    with torch.no_grad():
        logits_full, _, _, _ = model(full_ids)

    first_ids = full_ids[:, :3]
    with torch.no_grad():
        _, past_kv, _, _ = model(first_ids)

    for i in range(3, 5):
        next_id = full_ids[:, i : i + 1]
        with torch.no_grad():
            logits_inc, past_kv, _, _ = model(next_id, past_kv_list=past_kv)
        assert torch.allclose(logits_full[0, i], logits_inc[0, -1], atol=1e-4)


# ---- RoPE 缓存越界 ----


def test_rope_cache_exceeds_preallocation():
    """序列长度超过 rope_max_len 时应抛出 ValueError"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=32,
    )
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (1, 257))
    with pytest.raises(ValueError), torch.no_grad():
        model(x)


# ---- chunked prefill ----


def test_chunked_prefill_causal():
    """chunked prefill 的 logits 与全序列前向等价"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        max_seq_len=128,
        use_flash_attn=False,
    )
    model.eval()
    full_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    with torch.no_grad():
        logits_full, _, _, _ = model(full_ids)
    first_ids = full_ids[:, :5]
    with torch.no_grad():
        _, past_kv, _, _ = model(first_ids)
    second_ids = full_ids[:, 5:]
    with torch.no_grad():
        logits_chunked, _, _, _ = model(second_ids, past_kv_list=past_kv)
    assert torch.allclose(logits_full[0, 5], logits_chunked[0, 0], atol=1e-4)


# ---- Attention 变体测试 ----


def test_nope_gqa_forward_shape():
    """NoPEGQA 前向 shape 与 GQA 一致（QK-Norm 保留，仅去 RoPE）"""
    attn = NoPEGQA(256, 8, 2)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(4, 16, 256)
    out, attn_weights, kv = attn(x, cos, sin)
    assert out.shape == (4, 16, 256)
    assert attn_weights is not None
    assert kv[0].shape == (4, 2, 16, 32)


def test_nope_gqa_no_rope_dependence():
    """NoPEGQA 输出不受 cos/sin 变化影响（因为不使用 RoPE）"""
    attn = NoPEGQA(256, 8, 2)
    cos_a, sin_a = precompute_freqs_cis(32, 128)
    cos_b, sin_b = precompute_freqs_cis(32, 256)
    x = torch.randn(2, 8, 256)
    out_a, _, _ = attn(x, cos_a, sin_a)
    out_b, _, _ = attn(x, cos_b, sin_b)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_nope_gqa_kv_cache():
    """NoPEGQA 支持 KV Cache 增量解码"""
    attn = NoPEGQA(256, 8, 2)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(1, 10, 256)
    _, _, past_kv = attn(x, cos, sin)
    assert past_kv[0].shape[2] == 10
    x2 = torch.randn(1, 1, 256)
    _, _, new_kv = attn(x2, cos, sin, past_kv=past_kv)
    assert new_kv[0].shape[2] == 11


def test_alibi_gqa_forward_shape():
    """AliBiGQA 前向 shape 正确"""
    attn = AliBiGQA(256, 8, 2)
    x = torch.randn(4, 16, 256)
    out, attn_weights, kv = attn(x)
    assert out.shape == (4, 16, 256)
    assert attn_weights is not None
    assert kv[0].shape == (4, 2, 16, 32)


def test_alibi_gqa_no_rope_required():
    """AliBiGQA 不依赖 cos/sin 参数（用 ALiBi bias 替代）"""
    attn = AliBiGQA(256, 8, 2)
    x = torch.randn(2, 8, 256)
    out, _, _ = attn(x)
    assert out.shape == (2, 8, 256)
    assert not torch.isnan(out).any()


def test_alibi_gqa_kv_cache():
    """AliBiGQA 支持 KV Cache 增量解码"""
    attn = AliBiGQA(256, 8, 2)
    x = torch.randn(1, 10, 256)
    _, _, past_kv = attn(x)
    assert past_kv[0].shape[2] == 10
    x2 = torch.randn(1, 1, 256)
    _, _, new_kv = attn(x2, None, None, None, past_kv)
    assert new_kv[0].shape[2] == 11


def test_sliding_window_gqa_forward_shape():
    """SlidingWindowGQA 前向 shape 正确"""
    attn = SlidingWindowGQA(256, 8, 2, dropout=0.0)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(4, 16, 256)
    out, attn_weights, kv = attn(x, cos, sin)
    assert out.shape == (4, 16, 256)
    assert attn_weights is not None
    assert kv[0].shape == (4, 2, 16, 32)


def test_sliding_window_gqa_window_behavior():
    """窗口内 token 应能 attend，窗外 token 被 mask"""
    attn = SlidingWindowGQA(256, 4, 2, dropout=0.0, window_size=3)
    cos, sin = precompute_freqs_cis(64, 128)
    x = torch.randn(2, 8, 256)
    out, weights, _ = attn(x, cos, sin)
    assert out.shape == (2, 8, 256)
    assert not torch.isnan(out).any()
    # attention weights 不应全零
    assert weights.sum() > 0


def test_sliding_window_gqa_kv_cache():
    """SlidingWindowGQA 支持 KV Cache 增量解码"""
    attn = SlidingWindowGQA(256, 8, 2, dropout=0.0)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(1, 10, 256)
    _, _, past_kv = attn(x, cos, sin)
    assert past_kv[0].shape[2] == 10
    x2 = torch.randn(1, 1, 256)
    _, _, new_kv = attn(x2, cos, sin, past_kv=past_kv)
    assert new_kv[0].shape[2] == 11


def test_sliding_window_size_persisted():
    """window_size 参数通过 kwargs 正确传入"""
    attn = SlidingWindowGQA(128, 4, 2, dropout=0.0, window_size=2048)
    assert attn.window_size == 2048
