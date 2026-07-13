"""纯函数单元测试 — get_lr_cosine/wsd, dpo_loss, compute_log_probs, format_chatml, assert_same_architecture"""

import math

import torch

from gleamlm.inference.chatml import format_chatml
from gleamlm.training.dpo_trainer import compute_log_probs, dpo_loss
from gleamlm.utils.checkpoint import assert_same_architecture
from gleamlm.utils.torch_utils import get_lr_cosine, get_lr_wsd


# ---- LR 调度 ----


def test_lr_cosine_warmup():
    lr = get_lr_cosine(step=5, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    # step=5, warmup_steps=10, 5/10=0.5
    assert lr == 0.5


def test_lr_cosine_end():
    lr = get_lr_cosine(step=999, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    assert abs(lr - 0.1) < 1e-4


def test_lr_cosine_midpoint():
    lr = get_lr_cosine(step=500, total_steps=1000, warmup_ratio=0.01, min_lr_ratio=0.1)
    assert abs(lr - 0.55) < 0.01


def test_lr_wsd_warmup():
    lr = get_lr_wsd(step=5, total_steps=1000, warmup_ratio=0.02, stable_ratio=0.8, min_lr_ratio=0.05)
    assert lr == 5 / 20  # step=5, warmup_steps=20


def test_lr_wsd_stable():
    lr = get_lr_wsd(step=400, total_steps=1000, warmup_ratio=0.02, stable_ratio=0.8, min_lr_ratio=0.05)
    assert lr == 1.0


def test_lr_wsd_decay_end():
    lr = get_lr_wsd(step=999, total_steps=1000, warmup_ratio=0.02, stable_ratio=0.8, min_lr_ratio=0.05)
    assert abs(lr - 0.05) < 1e-4


# ---- DPO ----


def test_dpo_loss_equal():
    """chosen=rejected 时 loss 应为 -log sigmoid(0) = -log(0.5)"""
    policy_cho = torch.tensor([-2.0])
    policy_rej = torch.tensor([-2.0])
    ref_cho = torch.tensor([-3.0])
    ref_rej = torch.tensor([-3.0])
    loss = dpo_loss(policy_cho, policy_rej, ref_cho, ref_rej, beta=1.0)
    expected = -math.log(torch.sigmoid(torch.tensor(0.0)).item())
    assert abs(loss.item() - expected) < 1e-5


def test_dpo_loss_chosen_preferred():
    """policy 偏好 chosen 时 loss 应低于 policy 偏好 rejected"""
    # policy prefers chosen: p_cho > r_cho, p_rej = r_rej → term > 0 → low loss
    loss_correct = dpo_loss(
        torch.tensor([0.0]), torch.tensor([-2.0]),
        torch.tensor([-5.0]), torch.tensor([-2.0]),
        beta=1.0,
    )
    # policy prefers rejected: p_rej > r_rej, p_cho = r_cho → term < 0 → high loss
    loss_wrong = dpo_loss(
        torch.tensor([-5.0]), torch.tensor([0.0]),
        torch.tensor([-5.0]), torch.tensor([-2.0]),
        beta=1.0,
    )
    assert loss_correct.item() < loss_wrong.item()


def test_dpo_loss_beta_effect():
    """更大的 beta 放大偏好信号，使正确模型的 loss 更低"""
    loss_low_beta = dpo_loss(
        torch.tensor([0.0]), torch.tensor([-2.0]),
        torch.tensor([-5.0]), torch.tensor([-2.0]),
        beta=0.1,
    )
    loss_high_beta = dpo_loss(
        torch.tensor([0.0]), torch.tensor([-2.0]),
        torch.tensor([-5.0]), torch.tensor([-2.0]),
        beta=1.0,
    )
    assert loss_high_beta.item() < loss_low_beta.item()


# ---- compute_log_probs ----


def test_compute_log_probs_basic():
    logits = torch.zeros(1, 3, 5)  # [B=1, seq=3, vocab=5]
    input_ids = torch.tensor([[1, 1, 1]])
    mask = torch.tensor([[1.0, 1.0]])
    result = compute_log_probs(logits, input_ids, mask)
    assert result.shape == (1,)
    assert abs(result.item() - 2 * math.log(0.2)) < 1e-5


def test_compute_log_probs_mask():
    logits = torch.zeros(2, 3, 5)
    input_ids = torch.tensor([[1, 1, 1], [1, 1, 1]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = compute_log_probs(logits, input_ids, mask)
    assert result.shape == (2,)
    assert abs(result[0].item() - math.log(0.2)) < 1e-5
    assert abs(result[1].item() - math.log(0.2)) < 1e-5


# ---- ChatML ----


def test_chatml_single_message():
    result = format_chatml([{"role": "user", "content": "hi"}])
    assert result == "<|im_start|><|user|>\nhi<|im_end|>\n"


def test_chatml_with_generation_prompt():
    result = format_chatml([{"role": "system", "content": "Be helpful."}], add_generation_prompt=True)
    assert result == (
        "<|im_start|><|system|>\nBe helpful.<|im_end|>\n"
        "<|im_start|><|assistant|>\n"
    )


def test_chatml_multi_turn():
    result = format_chatml(
        [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    )
    assert result == (
        "<|im_start|><|user|>\nQ<|im_end|>\n"
        "<|im_start|><|assistant|>\nA<|im_end|>\n"
    )


# ---- assert_same_architecture ----


def test_assert_same_architecture_match():
    # 不应抛出异常
    assert_same_architecture({"vocab_size": 12002, "d_model": 512}, {"vocab_size": 12002, "d_model": 512})


def test_assert_same_architecture_mismatch():
    import pytest
    with pytest.raises(ValueError):
        assert_same_architecture({"vocab_size": 12002}, {"vocab_size": 999})


def test_assert_same_architecture_partial():
    # 一方缺 key 时不报错
    assert_same_architecture({"d_model": 512}, {"vocab_size": 12002, "d_model": 512})
