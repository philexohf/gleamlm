"""Model forward/backward shape tests"""

import pytest
import torch

from gleamlm.models.model import GleamLMModel

VOCAB_SIZE = 12002
D_MODEL = 512
MAX_SEQ_LEN = 256


@pytest.fixture(scope="module")
def model():
    model = GleamLMModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=12,
        num_heads=8,
        num_kv_heads=4,
        d_ff=1365,
        dropout=0.0,
        max_seq_len=MAX_SEQ_LEN,
        pad_token_id=0,
    )
    return model


def test_parameter_count(model):
    total, trainable = model.get_num_params()
    assert 35_000_000 < total < 42_000_000, f"Unexpected param count: {total / 1e6:.1f}M"


def test_forward_shape(model):
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 128))
    logits, kv_list = model(input_ids)
    batch, seq, vocab = logits.shape
    assert batch == 4
    assert seq == 128
    assert vocab == VOCAB_SIZE
    assert len(kv_list) == 12  # 12 layers


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
    logits, kv_cache = model(prompt)
    assert kv_cache[0][0].size(2) == 10  # seq_len

    past_kv = kv_cache
    next_token = logits[:, -1:].argmax(dim=-1)
    for _ in range(5):
        logits, past_kv = model(next_token, past_kv_list=past_kv)
        next_token = logits[:, -1:].argmax(dim=-1)
    assert past_kv[0][0].size(2) == 15  # 10 prefill + 5 decode


def test_long_sequence(model):
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 256))
    logits, _ = model(input_ids)
    assert logits.shape == (2, 256, VOCAB_SIZE)
