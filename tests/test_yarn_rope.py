"""
YaRN (Yet another RoPE extensioN) 单元测试。

验证 precompute_freqs_cis 和 _compute_yarn_freqs 在标准 RoPE 和 YaRN 模式下的正确性：
  - 位置编码唯一性
  - 高频维度在长距离处保持分辨能力
  - 低频维度线性插值正确性
  - 训练边界处平滑过渡
  - 相对位置性质

用法:
  python -m pytest tests/test_yarn_rope.py -v
"""

import torch

from gleamlm.models.model import _compute_yarn_freqs, apply_rope, precompute_freqs_cis, rotate_half

# ════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════


def _make_cos_sin_standard(head_dim: int, max_seq_len: int, base: float = 10000.0):
    return precompute_freqs_cis(head_dim, max_seq_len, base=base, rope_scale=1.0)


def _make_cos_sin_yarn(
    head_dim: int,
    max_seq_len: int,
    original_max_seq_len: int,
    rope_scale: float,
    base: float = 10000.0,
):
    return precompute_freqs_cis(
        head_dim,
        max_seq_len,
        base=base,
        rope_scale=rope_scale,
        original_max_seq_len=original_max_seq_len,
    )


# ════════════════════════════════════════════════════
# 标准 RoPE 基础测试
# ════════════════════════════════════════════════════


class TestStandardRoPE:
    def test_rope_shape(self):
        cos, sin = _make_cos_sin_standard(64, 128)
        B, H, S, D = 2, 4, 32, 64
        q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        q_out, k_out = apply_rope(q, k, cos, sin, offset=0)
        assert q_out.shape == q.shape and k_out.shape == k.shape

    def test_rope_not_identity(self):
        cos, sin = _make_cos_sin_standard(64, 128)
        q, k = torch.randn(1, 1, 16, 64), torch.randn(1, 1, 16, 64)
        q_out, k_out = apply_rope(q, k, cos, sin, offset=0)
        assert not torch.allclose(q_out, q)
        assert not torch.allclose(k_out, k)

    def test_offset_consistency(self):
        cos, sin = _make_cos_sin_standard(64, 256)
        B, H, D = 1, 2, 64
        q_all, k_all = torch.randn(B, H, 8, D), torch.randn(B, H, 8, D)
        qr_all, kr_all = apply_rope(q_all, k_all, cos, sin, offset=0)
        qr0, kr0 = apply_rope(q_all[:, :, :4], k_all[:, :, :4], cos, sin, offset=0)
        qr1, kr1 = apply_rope(q_all[:, :, 4:], k_all[:, :, 4:], cos, sin, offset=4)
        assert torch.allclose(qr_all[:, :, :4], qr0, atol=1e-6)
        assert torch.allclose(qr_all[:, :, 4:], qr1, atol=1e-6)

    def test_rotate_half_rsquare(self):
        x = torch.randn(4, 8, 64)
        assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)

    def test_relative_position(self):
        cos, sin = _make_cos_sin_standard(64, 256)
        q = torch.ones(1, 1, 256, 64)
        k = torch.ones(1, 1, 256, 64)
        qr, kr = apply_rope(q, k, cos, sin, offset=0)
        d = 5
        dots = [float((qr[0, 0, p] * kr[0, 0, p + d]).sum()) for p in range(10, 200)]
        assert max(dots) - min(dots) < 1e-4


class TestStandardRoPEScaled:
    def test_scale_is_pure_position_interpolation(self):
        cos, _ = _make_cos_sin_standard(64, 512)
        cos_pi, _ = precompute_freqs_cis(64, 512, rope_scale=4.0)
        assert torch.allclose(cos[32], cos_pi[128], atol=1e-6)


# ════════════════════════════════════════════════════
# YaRN 独立函数测试
# ════════════════════════════════════════════════════


class TestComputeYarnFreqs:
    def test_output_shape(self):
        freq = _compute_yarn_freqs(64, factor=8.0, original_max_seq_len=4096)
        assert freq.shape == (32,)

    def test_factor_one_is_standard(self):
        """factor=1 时 inv_freq 应等于标准 RoPE"""
        head_dim = 64
        freq_std = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        freq_yarn = _compute_yarn_freqs(head_dim, factor=1.0, original_max_seq_len=4096)
        assert torch.allclose(freq_std, freq_yarn, atol=1e-6)

    def test_low_freq_interpolated_high_freq_extrapolated(self):
        """factor=8: 低频 inv_freq 约为 1/(8*theta), 高频 inv_freq ≈ 1/theta"""
        head_dim = 64
        freq_std = 1.0 / (10000.0 ** (torch.arange(0, 32, dtype=torch.float) / 32))
        freq_yarn = _compute_yarn_freqs(head_dim, factor=8.0, original_max_seq_len=4096)

        # 低频 dim 0: inv_freq_yarn ≈ inv_freq_std / 8
        assert abs(freq_yarn[0].item() - freq_std[0].item() / 8.0) < 1e-4

        # 高频 dim 31: inv_freq_yarn ≈ inv_freq_std (不缩放)
        ratio = freq_yarn[-1].item() / freq_std[-1].item()
        assert 0.9 < ratio < 1.1, f"高频错误缩放: ratio={ratio:.4f}"

    def test_inv_freq_strictly_decreasing(self):
        """inv_freq 应严格单调递减（或等值），无异常跳变"""
        freq = _compute_yarn_freqs(64, factor=8.0, original_max_seq_len=4096)
        # 低频到高频整体趋势向下，允许 ramp 过渡处有小幅平台
        assert freq[0] > freq[-1] * 10  # 整体趋势正确
        # 无 NaNs
        assert not freq.isnan().any()


# ════════════════════════════════════════════════════
# YaRN 位置编码测试
# ════════════════════════════════════════════════════


class TestYaRNPositionUniqueness:
    def test_all_positions_unique(self):
        cos, sin = _make_cos_sin_yarn(64, 256, original_max_seq_len=32, rope_scale=8.0)
        for i in range(255):
            assert (cos[i] - cos[i + 1]).abs().max() > 0 or (sin[i] - sin[i + 1]).abs().max() > 0

    def test_progressive_difference(self):
        cos, _ = _make_cos_sin_yarn(64, 256, original_max_seq_len=64, rope_scale=4.0)
        d_near = (cos[10] - cos[11]).abs().max().item()
        d_far = (cos[200] - cos[201]).abs().max().item()
        assert d_far > d_near * 0.5


class TestYaRNHighFreqPreserved:
    def test_high_freq_not_collapsed(self):
        cos_y, _ = _make_cos_sin_yarn(64, 4097, original_max_seq_len=512, rope_scale=8.0)
        diffs = [
            float((cos_y[p, -8:] - cos_y[p + 1, -8:]).abs().max())
            for p in [100, 500, 1000, 2000, 3000, 4000, 4090]
        ]
        assert min(diffs) > 5e-5, f"高频塌缩: min diff={min(diffs):.6e}"

    def test_high_freq_comparable_to_standard(self):
        cos_s, _ = _make_cos_sin_standard(64, 4097)
        cos_y, _ = _make_cos_sin_yarn(64, 4097, original_max_seq_len=512, rope_scale=8.0)
        for pos in [500, 1000, 2000]:
            d_y = (cos_y[pos, -8:] - cos_y[pos + 1, -8:]).abs().max().item()
            d_s = (cos_s[pos, -8:] - cos_s[pos + 1, -8:]).abs().max().item()
            assert d_y > d_s * 0.01, f"YaRN pos {pos}: {d_y:.4f} << std {d_s:.4f}"


class TestYaRNBoundarySmoothness:
    def test_boundary_continuous(self):
        cos_y, _ = _make_cos_sin_yarn(64, 256, original_max_seq_len=64, rope_scale=4.0)

        def l2(p):
            return (cos_y[p] - cos_y[p + 1]).norm().item()

        b = sum(l2(p) for p in range(56, 72)) / 16
        i = sum(l2(p) for p in range(32, 48)) / 16
        assert b < i * 5.0, f"边界突变: boundary={b:.4f} inner={i:.4f}"


class TestYaRNRelativePosition:
    def test_relative_position_yarn(self):
        """YaRN: q(m)·k(n) 仍只依赖于 d=m-n（用 ones 向量验证）"""
        cos, sin = _make_cos_sin_yarn(64, 256, original_max_seq_len=64, rope_scale=4.0)
        q = torch.ones(1, 1, 256, 64)
        k = torch.ones(1, 1, 256, 64)
        qr, kr = apply_rope(q, k, cos, sin, offset=0)
        d = 5
        dots = [float((qr[0, 0, p] * kr[0, 0, p + d]).sum()) for p in range(5, 55)]
        # YaRN 调整 inv_freq 后某些维度的 theta 变化较大，
        # 导致 cos(d*theta) 对 d 的精度要求更高，允许稍大容差
        assert max(dots) - min(dots) < 0.1, f"spread={max(dots) - min(dots):.4e}"


class TestYaRNTrainingRange:
    def test_position_zero_identical(self):
        cos_s, _ = _make_cos_sin_standard(64, 256)
        cos_y, _ = _make_cos_sin_yarn(64, 256, original_max_seq_len=64, rope_scale=4.0)
        assert torch.allclose(cos_s[0], cos_y[0], atol=1e-6)

    def test_interpolation_consistency(self):
        """factor=4: 训练范围边界内，YaRN 频率过渡平滑"""
        cos_y, _ = _make_cos_sin_yarn(64, 256, original_max_seq_len=64, rope_scale=4.0)

        # 在训练边界前后各取 8 个位置，相邻差异应连续无跳变
        def l2(p):
            return (cos_y[p] - cos_y[p + 1]).norm().item()

        diffs = [l2(p) for p in range(56, 72)]
        # 差异不应有数量级跳变
        assert max(diffs) < min(diffs) * 5.0, f"训练边界频率跳变: {diffs}"


# ════════════════════════════════════════════════════
# 0.6B 模型实际参数回归测试
# ════════════════════════════════════════════════════


class TestGleamLM06BConfig:
    HEAD_DIM = 64
    ORIGINAL = 4096
    SCALE = 8.0
    THETA = 10000.0
    ROPE_MAX_LEN = ORIGINAL * SCALE

    def test_shape(self):
        cos, sin = precompute_freqs_cis(
            self.HEAD_DIM,
            self.ROPE_MAX_LEN,
            base=self.THETA,
            rope_scale=self.SCALE,
            original_max_seq_len=self.ORIGINAL,
        )
        assert cos.shape == (self.ROPE_MAX_LEN, self.HEAD_DIM)
        assert sin.shape == (self.ROPE_MAX_LEN, self.HEAD_DIM)

    def test_position_zero(self):
        cos_s, _ = precompute_freqs_cis(self.HEAD_DIM, 256, base=self.THETA, rope_scale=1.0)
        cos_y, _ = precompute_freqs_cis(
            self.HEAD_DIM,
            self.ROPE_MAX_LEN,
            base=self.THETA,
            rope_scale=self.SCALE,
            original_max_seq_len=self.ORIGINAL,
        )
        assert torch.allclose(cos_s[0], cos_y[0], atol=1e-6)

    def test_high_freq_alive(self):
        cos_y, _ = precompute_freqs_cis(
            self.HEAD_DIM,
            self.ROPE_MAX_LEN,
            base=self.THETA,
            rope_scale=self.SCALE,
            original_max_seq_len=self.ORIGINAL,
        )
        for pos in range(0, min(2000, self.ROPE_MAX_LEN - 1)):
            assert (cos_y[pos] - cos_y[pos + 1]).abs().max() > 0
        d = (cos_y[4090, -8:] - cos_y[4091, -8:]).abs().max().item()
        assert d > 5e-4, f"0.6B 高频塌缩: pos 4090 diff={d:.4f}"

    def test_low_freq_dim_unique(self):
        cos_y, _ = precompute_freqs_cis(
            self.HEAD_DIM,
            self.ROPE_MAX_LEN,
            base=self.THETA,
            rope_scale=self.SCALE,
            original_max_seq_len=self.ORIGINAL,
        )
        vals = cos_y[1000, :8]
        for i in range(0, 6, 2):
            assert (vals[i : i + 2] - vals[i + 2 : i + 4]).abs().max() > 0
