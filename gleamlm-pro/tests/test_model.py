"""GleamLM-Pro 126M 模型前向/反向测试 — pytest"""
import pytest
import torch

from gleamlm.models.model import GleamLMModel

VOCAB_SIZE = 12002
D_MODEL = 768
MAX_SEQ_LEN = 256


@pytest.fixture(scope="module")
def model():
    return GleamLMModel(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=18,
        num_heads=12, num_kv_heads=6, d_ff=2048,
        dropout=0.0, max_seq_len=MAX_SEQ_LEN, pad_token_id=0,
    )


def test_parameter_count(model):
    total, trainable = model.get_num_params()
    assert 120_000_000 < total < 140_000_000, f"Unexpected param count: {total / 1e6:.1f}M"


def test_forward_shape(model):
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 128))
    logits, kv_list = model(input_ids)
    assert logits.shape == (4, 128, VOCAB_SIZE)
    assert len(kv_list) == 18


def test_backward_no_nan(model):
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 128))
    logits, _ = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB_SIZE),
        input_ids[:, 1:].reshape(-1),
        ignore_index=0,
    )
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


def test_kv_cache_forward(model):
    prompt = torch.randint(0, VOCAB_SIZE, (1, 10))
    with torch.no_grad():
        logits, kv_cache = model(prompt)
    assert kv_cache[0][0].size(2) == 10
    past_kv = kv_cache
    next_token = logits[:, -1:].argmax(dim=-1)
    with torch.no_grad():
        for _ in range(5):
            logits, past_kv = model(next_token, past_kv_list=past_kv)
            next_token = logits[:, -1:].argmax(dim=-1)
    assert past_kv[0][0].size(2) == 15


def test_long_sequence(model):
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 256))
    logits, _ = model(input_ids)
    assert logits.shape == (2, 256, VOCAB_SIZE)


def test_flash_attn():
    flash_model = GleamLMModel(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=18,
        num_heads=12, num_kv_heads=6, d_ff=2048,
        dropout=0.0, max_seq_len=MAX_SEQ_LEN,
        use_flash_attn=True,
    )
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 64))
    with torch.no_grad():
        logits, _ = flash_model(input_ids)
    assert logits.shape == (2, 64, VOCAB_SIZE)
