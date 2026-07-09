"""Tokenizer encode/decode round-trip tests"""

import pytest

from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


@pytest.fixture(scope="module")
def tokenizer():
    return BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)


def test_vocab_size(tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size > 12000, f"Expected >12000, got {vocab_size}"


def test_encode_decode_roundtrip_chinese(tokenizer):
    text = "你好，世界！"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Round-trip failed: '{text}' → ids={ids} → '{decoded}'"


def test_encode_decode_roundtrip_english(tokenizer):
    text = "Hello, World!"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Round-trip failed: '{text}' → ids={ids} → '{decoded}'"


def test_encode_decode_roundtrip_mixed(tokenizer):
    text = "AI人工智能"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Round-trip failed: '{text}' → ids={ids} → '{decoded}'"


def test_special_tokens_present(tokenizer):
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<pad>", "<unk>"]:
        assert tok in tokenizer.special_tokens, f"Missing special token: {tok}"
        tokenizer.token_to_id(tok)
        encoded = tokenizer.encode(tok, add_bos=False, add_eos=False)
        assert len(encoded) == 1, f"'{tok}' should encode to single token, got {len(encoded)}"


def test_add_bos_eos(tokenizer):
    text = "你好"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id, "BOS not added"
    assert ids[-1] == tokenizer.eos_id, "EOS not added"


def test_pad_id(tokenizer):
    assert tokenizer.pad_id == tokenizer.special_tokens["<pad>"]


def test_encode_empty_string(tokenizer):
    ids = tokenizer.encode("", add_bos=False, add_eos=False)
    assert ids == [], f"Empty string should encode to empty list, got {ids}"
