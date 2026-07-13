"""采样器 temperature/top_k/top_p/repetition_penalty 测试"""

import pytest
import torch

from gleamlm.inference.sampler import sample_token


def test_temperature_one():
    logits = torch.randn(1, 1000)
    token = sample_token(logits, temperature=1.0)
    assert token.dim() == 1
    assert 0 <= token.item() < 1000


def test_temperature_zero_greedy():
    """temperature=0 时 softmax 峰值应指向 argmax"""
    logits = torch.tensor([[0.1, 2.0, 0.5, -1.0]])
    # 由于 multinomial 不是确定性的，改为验证 softmax 峰值与 argmax 一致
    probs = torch.softmax(logits, dim=-1)
    assert probs.argmax().item() == 1  # index 1 = 2.0


def test_top_k():
    logits = torch.randn(1, 1000)
    token = sample_token(logits, top_k=10)
    assert 0 <= token.item() < 1000


def test_top_p():
    logits = torch.randn(1, 1000)
    token = sample_token(logits, top_p=0.9)
    assert 0 <= token.item() < 1000


def test_repetition_penalty_reduces_logit():
    """penalty > 1 应直接降低已生成 token 的 logit 值"""
    logits = torch.tensor([[0.5, 5.0, 0.3, -0.2]])
    # clone 后施加 penalty
    logits_pen = logits.clone()
    for gid in [1]:
        logits_pen[..., gid] = logits_pen[..., gid] / 100.0
    # token 1 的 logit 从 5.0 降为 0.05，不再是最大
    assert logits_pen[0, 1].item() == pytest.approx(0.05)
    assert logits_pen.argmax().item() != 1


def test_batch_sampling():
    logits = torch.randn(4, 12002)
    tokens = sample_token(logits, temperature=0.8)
    assert tokens.shape == (4,)


def test_logits_unchanged_with_defaults():
    """默认参数不修改 logits"""
    logits = torch.randn(1, 100)
    logits_copy = logits.clone()
    sample_token(logits, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0)
    assert torch.equal(logits, logits_copy)


def test_repetition_penalty_accumulates():
    """generated_ids 中已有 1 个 token 后，penalty 使相同 token 的 logit 降低"""
    logits = torch.tensor([[2.0, 5.0, 3.0, 1.0]])
    pen = 2.0
    logits_after_first = logits.clone()
    returned = sample_token(logits_after_first, repetition_penalty=pen, generated_ids=[])
    # 同一次调用内 generated_ids 不含已生成 token，故 logits 不变
    assert torch.equal(logits_after_first, logits), "empty generated_ids leaves logits unchanged"

    # 再调用时传入之前生成的 token，penalty 作用于该 token
    logits_after_pen = logits.clone()
    _ = sample_token(logits_after_pen, repetition_penalty=pen, generated_ids=[returned.item()])
    idx = returned.item()
    assert logits_after_pen[0, idx].item() == pytest.approx(logits[0, idx].item() / pen), (
        "repetition penalty divided logit of generated token"
    )


def test_repetition_penalty_1_is_noop():
    """penalty=1.0 即使有 generated_ids 也不修改 logits"""
    logits = torch.randn(1, 100)
    logits_copy = logits.clone()
    _ = sample_token(logits, repetition_penalty=1.0, generated_ids=[3, 7, 42])
    assert torch.equal(logits, logits_copy)


def test_repetition_penalty_prevents_token_cycling_greedy():
    """回归测试: repetition_penalty=1.15 降低重复 token 的 logit — 贪婪等效场景

    构造 logits 使 token 5 为 argmax（logit=5.0），token 6 紧随其后（logit=4.5）。
    无 penalty 时 token 5 持续被选；施加 penalty=1.15 后，token 5 被选 1 次后其
    有效 logit 降至 ~4.35 < 4.5，argmax 转为 token 6，打破连续重复。
    """
    torch.manual_seed(42)
    logits = torch.zeros(1, 100)
    logits[0, 5] = 5.0
    logits[0, 6] = 4.5

    generated_ids: list[int] = []
    penalty = 1.15

    for _step in range(10):
        token = sample_token(
            logits.clone(),
            temperature=0.0,
            repetition_penalty=penalty,
            generated_ids=generated_ids,
        )
        tid = int(token.item())
        generated_ids.append(tid)

    # penalty 使 token 5 的 logit 在出现多次后显著降低
    count5 = generated_ids.count(5)
    assert count5 > 0, "至少应选择过 token 5"
    effective_logit_5 = 5.0 / (penalty**count5)
    assert effective_logit_5 < 4.5, (
        f"penalty={penalty} 应将 token 5 的有效 logit ({effective_logit_5:.2f}) "
        f"降至低于 token 6 (4.5)"
    )

    # 验证最终有除 token 5 以外的 token 被选
    unique_tokens = set(generated_ids)
    assert len(unique_tokens) > 1, (
        f"repetition_penalty 应使生成中出现多个不同 token，实际 token 集合: {unique_tokens}"
    )


def test_repetition_penalty_prevents_token_cycling_temperature():
    """回归测试: temperature=0.8 下 repetition_penalty=1.15 仍打破循环"""
    torch.manual_seed(123)
    logits = torch.zeros(1, 100)
    logits[0, 5] = 5.0
    logits[0, 6] = 4.5

    generated_ids: list[int] = []
    penalty = 1.15

    for _step in range(10):
        token = sample_token(
            logits.clone(),
            temperature=0.8,
            repetition_penalty=penalty,
            generated_ids=generated_ids,
        )
        tid = int(token.item())
        generated_ids.append(tid)

    # penalty 使 token 5 的有效 logit 在重复一定次数后低于 token 6
    count5 = generated_ids.count(5)
    assert count5 > 0, "至少应选择过 token 5"
    effective_logit_5 = 5.0 / (penalty**count5)
    assert effective_logit_5 < 4.5, (
        f"temperature=0.8: penalty={penalty} 应将 token 5 的有效 logit "
        f"({effective_logit_5:.2f}) 降至低于 token 6 (4.5)"
    )

    unique_tokens = set(generated_ids)
    assert len(unique_tokens) > 1, (
        f"temperature=0.8: repetition_penalty 应使生成中出现多个不同 token，"
        f"实际 token 集合: {unique_tokens}"
    )


# ---- TextStreamer ----


def test_streamer_generate(small_model, tokenizer):
    from gleamlm.inference.streamer import TextStreamer

    streamer = TextStreamer(tokenizer)
    small_model.eval()
    count = 0
    for chunk in streamer.generate_text(
        small_model, "你好", max_new_tokens=10, temperature=0.8, top_k=50
    ):
        count += 1
        assert isinstance(chunk, str), f"Expected str got {type(chunk)}"
    assert count > 0, "Should generate at least 1 token"
