"""端到端集成测试 — 少量训练步骤，验证 model/optimizer/scaler/trainer 联动正确。

不依赖完整训练脚本，只验证核心路径：前向 → 反向 → 优化器更新。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gleamlm.models.attention_variants import NoPEGQA, SlidingWindowGQA
from gleamlm.models.model import DecoderLayer, GleamLMModel, MoE, precompute_freqs_cis
from gleamlm.trainer.base_trainer import create_scaler, optimizer_step
from gleamlm.trainer.schedulers import get_lr_cosine
from gleamlm.utils.torch_utils import safe_autocast

VOCAB_SIZE = 12002


# ── 训练核心路径：前向 → loss → backward → optimizer step ──────────────


def test_training_step_gradient_decreases_loss():
    """一个完整的 training step 应使同一个 batch 的 loss 下降"""
    model = GleamLMModel(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,
    )
    # 固定参数以复现
    torch.manual_seed(42)

    batch = torch.randint(0, VOCAB_SIZE, (4, 32))
    labels = batch.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = create_scaler()

    model.train()
    # ── 记录 step 前 loss ───────────────────────────────────────────
    with torch.no_grad(), safe_autocast():
        logits_before, _, _, _ = model(batch)
        loss_before = F.cross_entropy(
            logits_before[:, :-1].reshape(-1, VOCAB_SIZE),
            labels[:, 1:].reshape(-1),
            ignore_index=0,
        )

    # ── 一个 training step ──────────────────────────────────────────
    with safe_autocast():
        logits, _, _, _ = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE),
            labels[:, 1:].reshape(-1),
            ignore_index=0,
        )
    scaler.scale(loss).backward()
    optimizer_step(optimizer, scaler, parameters=model.parameters(), clip_grad=1.0)

    # ── 验证 loss 下降 ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad(), safe_autocast():
        logits_after, _, _, _ = model(batch)
        loss_after = F.cross_entropy(
            logits_after[:, :-1].reshape(-1, VOCAB_SIZE),
            labels[:, 1:].reshape(-1),
            ignore_index=0,
        )
    assert loss_after < loss_before, (
        f"Expected loss to decrease after training step ({loss_before:.4f} -> {loss_after:.4f})"
    )


def test_training_step_multiple_steps():
    """连续多个 training step 不产生 NaN"""
    torch.manual_seed(7)
    model = GleamLMModel(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = create_scaler()

    model.train()
    losses = []
    for _ in range(10):
        batch = torch.randint(0, VOCAB_SIZE, (4, 32))
        labels = batch.clone()

        with safe_autocast():
            logits, _, _, _ = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                labels[:, 1:].reshape(-1),
                ignore_index=0,
            )
        scaler.scale(loss).backward()
        optimizer_step(optimizer, scaler, parameters=model.parameters(), clip_grad=1.0)
        losses.append(loss.item())

    assert not any(torch.isnan(torch.tensor(losses))), "NaN loss detected"
    # 损失应在合理范围内（10 步随机 batch 不一定单调下降）
    assert min(losses) > 0, "Loss should be positive"
    assert max(losses) < 20.0, f"Loss is unreasonably high: {max(losses):.2f}"


# ── MoE 训练：验证 aux_loss 正常流动 ──────────────────────────────────


def test_moe_aux_loss_in_training():
    """MoE 模型训练时 aux_loss 通过 return 元组正常传递并加入 total loss"""
    torch.manual_seed(3)
    model = GleamLMModel(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,
        ffn_variant=MoE,
        num_experts=4,
        top_k=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = create_scaler()

    model.train()
    batch = torch.randint(0, VOCAB_SIZE, (4, 32))
    labels = batch.clone()

    with safe_autocast():
        logits, _, aux_loss, _ = model(batch)
    assert isinstance(aux_loss, torch.Tensor), f"aux_loss should be a tensor, got {type(aux_loss)}"
    assert aux_loss.item() > 0, "MoE aux_loss should be > 0"

    # aux_loss should be attached to computation graph
    ppl_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB_SIZE),
        labels[:, 1:].reshape(-1),
        ignore_index=0,
    )
    total_loss = ppl_loss + 0.01 * aux_loss
    scaler.scale(total_loss).backward()
    optimizer_step(optimizer, scaler, parameters=model.parameters(), clip_grad=1.0)

    # 每层的 aux_loss 通过 DecoderLayer.aux_loss 可访问
    for i, layer in enumerate(model.layers):
        assert hasattr(layer, "aux_loss"), f"Layer {i} missing aux_loss attribute"
        assert layer.aux_loss is not None, f"Layer {i} aux_loss should not be None"


# ── LR 调度集成 ──────────────────────────────────────────────────────


def test_manual_lr_schedule_matches_cosine():
    """手动 LR 调度（pretrain 风格）计算一致"""
    total_steps = 100
    warmup_ratio = 0.1

    def manual_lr(step, base_lr=1.0):
        mult = get_lr_cosine(step, total_steps, warmup_ratio, min_lr_ratio=0.05)
        return base_lr * mult

    # warmup 阶段 lr 递增
    assert manual_lr(0) < manual_lr(10)
    # warmup 结束后 lr 递减
    assert manual_lr(20) > manual_lr(90)
    # 终点 lr = base_lr * min_lr_ratio
    assert abs(manual_lr(total_steps - 1) - 0.05) < 0.01


# ── 变体 DecoderLayer 训练 ────────────────────────────────────────────


def test_decoder_layer_nope_training_step():
    """NoPEGQA DecoderLayer 训练不崩溃"""
    torch.manual_seed(1)
    layer = DecoderLayer(128, 4, 2, 256, attn_variant=NoPEGQA)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(2, 16, 128)

    for _ in range(3):
        out, _ = layer(x, cos, sin)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert not torch.isnan(out).any()


def test_decoder_layer_sliding_window_training_step():
    """SlidingWindowGQA DecoderLayer 训练不崩溃"""
    torch.manual_seed(2)
    layer = DecoderLayer(128, 4, 2, 256, attn_variant=SlidingWindowGQA)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    cos, sin = precompute_freqs_cis(32, 128)
    x = torch.randn(2, 16, 128)

    for _ in range(3):
        out, _ = layer(x, cos, sin)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert not torch.isnan(out).any()
