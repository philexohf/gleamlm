"""Run nano module tests without pytest (direct execution)"""

import sys
import traceback

import torch

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        traceback.print_exc()
        failed += 1


# Model Tests
print("=" * 50)
print("Model Tests")
print("=" * 50)


def test_parameter_count():
    from gleamlm.models.model import GleamLMModel

    model = GleamLMModel(
        vocab_size=12002,
        d_model=512,
        num_layers=12,
        num_heads=8,
        num_kv_heads=4,
        d_ff=1365,
        dropout=0.0,
        max_seq_len=256,
        pad_token_id=0,
    )
    total, trainable = model.get_num_params()
    print(f"    Params: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")
    assert 35_000_000 < total < 42_000_000, f"Unexpected param count: {total / 1e6:.1f}M"
    return model


test("test_parameter_count", test_parameter_count)

# Create model once for reuse
from gleamlm.models.model import GleamLMModel

model = GleamLMModel(
    vocab_size=12002,
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=4,
    d_ff=1365,
    dropout=0.0,
    max_seq_len=256,
    pad_token_id=0,
)


def test_forward_shape():
    input_ids = torch.randint(0, 12002, (4, 128))
    logits, kv_list = model(input_ids)
    assert logits.shape == (4, 128, 12002), f"Expected (4,128,12002) got {logits.shape}"
    assert len(kv_list) == 12, f"Expected 12 KV layers got {len(kv_list)}"


test("test_forward_shape", test_forward_shape)


def test_backward_no_nan():
    input_ids = torch.randint(0, 12002, (4, 128))
    logits, _ = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 12002),
        input_ids[:, 1:].reshape(-1),
        ignore_index=0,
    )
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
    model.zero_grad()


test("test_backward_no_nan", test_backward_no_nan)


def test_kv_cache_forward():
    prompt = torch.randint(0, 12002, (1, 10))
    with torch.no_grad():
        logits, kv_cache = model(prompt)
    assert kv_cache[0][0].size(2) == 10, f"Expected KV seq_len=10 got {kv_cache[0][0].size(2)}"
    past_kv = kv_cache
    next_token = logits[:, -1:].argmax(dim=-1)
    with torch.no_grad():
        for _ in range(5):
            logits, past_kv = model(next_token, past_kv_list=past_kv)
            next_token = logits[:, -1:].argmax(dim=-1)
    assert past_kv[0][0].size(2) == 15, f"Expected KV seq_len=15 got {past_kv[0][0].size(2)}"


test("test_kv_cache_forward", test_kv_cache_forward)


def test_long_sequence():
    with torch.no_grad():
        input_ids = torch.randint(0, 12002, (2, 256))
        logits, _ = model(input_ids)
    assert logits.shape == (2, 256, 12002), f"Expected (2,256,12002) got {logits.shape}"


test("test_long_sequence", test_long_sequence)

# Tokenizer Tests
print("")
print("=" * 50)
print("Tokenizer Tests")
print("=" * 50)

from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)


def test_vocab_size():
    vocab_size = tokenizer.get_vocab_size()
    print(f"    Vocab size: {vocab_size}")
    assert vocab_size > 12000, f"Expected >12000, got {vocab_size}"


test("test_vocab_size", test_vocab_size)


def test_encode_decode_roundtrip_chinese():
    text = "你好，世界！"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Round-trip failed: '{text}' -> '{decoded}'"


test("test_encode_decode_roundtrip_chinese", test_encode_decode_roundtrip_chinese)


def test_encode_decode_roundtrip_english():
    text = "Hello, World!"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Round-trip failed: '{text}' -> '{decoded}'"


test("test_encode_decode_roundtrip_english", test_encode_decode_roundtrip_english)


def test_special_tokens():
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<pad>", "<unk>"]:
        assert tok in tokenizer.special_tokens, f"Missing: {tok}"
        tokenizer.token_to_id(tok)
        encoded = tokenizer.encode(tok, add_bos=False, add_eos=False)
        assert len(encoded) == 1, f"'{tok}' should be single token, got {len(encoded)}"
    # <|user|> / <|assistant|> 可能存在于旧版 checkpoint，仅在新训练时移除


test("test_special_tokens", test_special_tokens)


def test_add_bos_eos():
    ids = tokenizer.encode("你好", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id, "BOS not added"
    assert ids[-1] == tokenizer.eos_id, "EOS not added"


test("test_add_bos_eos", test_add_bos_eos)


def test_empty_string():
    ids = tokenizer.encode("", add_bos=False, add_eos=False)
    assert ids == [], f"Empty should be [], got {ids}"


test("test_empty_string", test_empty_string)

# Inference Tests
print("")
print("=" * 50)
print("Inference Tests (Streamer + Sampler)")
print("=" * 50)


def test_sampler_shape():
    from gleamlm.inference.sampler import sample_token

    logits = torch.randn(1, 12002)
    token = sample_token(logits, temperature=0.8, top_k=50, top_p=0.9)
    assert token.dim() == 1, f"Expected 1D tensor got {token.dim()}D"
    assert 0 <= token.item() < 12002, f"Token out of range: {token.item()}"


test("test_sampler_shape", test_sampler_shape)


def test_streamer_generate():
    from gleamlm.inference.streamer import TextStreamer

    streamer = TextStreamer(tokenizer)
    prompt_ids = torch.tensor([[tokenizer.bos_id]], dtype=torch.long)
    model.eval()
    count = 0
    with torch.no_grad():
        for token_id in streamer.generate(
            model, prompt_ids, max_new_tokens=10, temperature=0.8, top_k=50
        ):
            count += 1
            assert isinstance(token_id, int), f"Expected int got {type(token_id)}"
    assert count > 0, "Should generate at least 1 token"


test("test_streamer_generate", test_streamer_generate)

# Summary
print("")
print("=" * 50)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(0 if failed == 0 else 1)
