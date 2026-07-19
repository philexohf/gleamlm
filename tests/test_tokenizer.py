"""分词器 编解码/特殊token/边界/训练 测试"""

import os
import tempfile

import pytest

from gleamlm.tokenizer.tokenizer import BBPETokenizer


def test_vocab_size(tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size > 12000, f"Expected >12000, got {vocab_size}"


def test_encode_decode_chinese(tokenizer):
    text = "你好，世界！"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Round-trip failed: '{text}' → '{decoded}'"


def test_encode_decode_english(tokenizer):
    text = "Hello, World!"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Round-trip failed: '{text}' → '{decoded}'"


def test_encode_decode_mixed(tokenizer):
    text = "AI人工智能"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_special_tokens_single_id(tokenizer):
    specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    for tok in specials:
        assert tok in tokenizer.special_tokens, f"Missing: {tok}"
        encoded = tokenizer.encode(tok, add_bos=False, add_eos=False)
        assert len(encoded) == 1, f"'{tok}' should be 1 token, got {len(encoded)}"


def test_add_bos_eos(tokenizer):
    ids = tokenizer.encode("你好", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id, "BOS not added"
    assert ids[-1] == tokenizer.eos_id, "EOS not added"


def test_encode_empty_string(tokenizer):
    ids = tokenizer.encode("", add_bos=False, add_eos=False)
    assert ids == []


def test_chatml_format(tokenizer):
    prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt)
    decoded = tokenizer.decode(ids)
    assert "你好" in decoded
    assert tokenizer.special_tokens["<|im_start|>"] in ids
    assert tokenizer.special_tokens["<|im_end|>"] in ids


def test_decode_skip_special(tokenizer):
    text = "你好<|endoftext|>世界"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids, skip_special=True)
    assert "<|endoftext|>" not in decoded
    assert decoded == "你好世界"


def test_train_tokenizer_small():
    """从零训练小词表 tokenizer，验证编解码和特殊 token"""
    test_text = (
        "这是中文测试文本用于训练BBPE分词器。\n"
        "This is English test text for BBPE tokenizer training.\n"
        "人工智能正在改变世界。Artificial intelligence is changing the world.\n"
        "你好，世界！Hello, World!\n"
    ) * 100

    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt")
    tmp.write(test_text)
    tmp_path = tmp.name
    tmp.close()

    try:
        save_dir = tempfile.mkdtemp()
        trained = BBPETokenizer.train_from_files(
            [tmp_path],
            vocab_size=500,
            save_dir=save_dir,
            max_train_chars=500_000,
        )

        assert trained.get_vocab_size() >= 500
        assert len(trained.special_tokens) >= 13

        test_cases = ["你好，世界！", "人工智能", "Hello World"]
        for text in test_cases:
            ids = trained.encode(text, add_bos=False, add_eos=False)
            decoded = trained.decode(ids)
            assert decoded == text, f"Round-trip failed: '{text}' → '{decoded}'"

        specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        for tok in specials:
            assert tok in trained.special_tokens
            encoded = trained.encode(tok, add_bos=False, add_eos=False)
            assert len(encoded) == 1, f"'{tok}' should be 1 token, got {len(encoded)}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_train_save_load_roundtrip():
    """训练→保存→加载→编码，验证持久化正确性"""
    test_text = "测试文本。" * 200
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt")
    tmp.write(test_text)
    tmp_path = tmp.name
    tmp.close()

    save_dir = tempfile.mkdtemp()
    try:
        trained = BBPETokenizer.train_from_files(
            [tmp_path],
            vocab_size=500,
            save_dir=save_dir,
            max_train_chars=200_000,
        )
        original_ids = trained.encode("你好世界", add_bos=False, add_eos=False)

        loaded = BBPETokenizer.load(save_dir)
        assert loaded.get_vocab_size() == trained.get_vocab_size()
        assert loaded.special_tokens == trained.special_tokens
        assert loaded.merges == trained.merges
        reloaded_ids = loaded.encode("你好世界", add_bos=False, add_eos=False)
        assert reloaded_ids == original_ids, "Save/load round-trip 后编码不一致"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_encode_invalid_type():
    tok = BBPETokenizer()
    with pytest.raises(TypeError, match="Expected str"):
        tok.encode(12345)


def test_token_to_id_unknown():
    tok = BBPETokenizer()
    tok._add_special_tokens()
    assert tok.token_to_id("__nonexistent__") == tok.unk_id


def test_decode_unknown_id():
    tok = BBPETokenizer()
    result = tok.decode([99999])
    assert result is not None
    assert "?" in result or result == ""


def test_train_multiple_files():
    """多文件+权重训练"""
    tmp1 = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt")
    tmp1.write("中文AAA测试序列用于BPE训练。中文分词器需要大量中文文本。" * 500)
    tmp1_path = tmp1.name
    tmp1.close()

    tmp2 = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt")
    tmp2.write("English text for BPE tokenizer training with sufficient repetition." * 500)
    tmp2_path = tmp2.name
    tmp2.close()

    save_dir = tempfile.mkdtemp()
    try:
        trained = BBPETokenizer.train_from_files(
            [tmp1_path, tmp2_path],
            vocab_size=500,
            save_dir=save_dir,
            max_train_chars=200_000,
            ratios=[0.5, 0.5],
        )
        assert trained.get_vocab_size() >= 400
    finally:
        for p in [tmp1_path, tmp2_path]:
            if os.path.exists(p):
                os.unlink(p)


def test_encode_decode_very_long(tokenizer):
    text = "测试" * 5000
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_encode_decode_special_token_adjacent(tokenizer):
    text = "<|im_start|><|im_end|>"
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    assert len(ids) == 2
    assert ids[0] == tokenizer.special_tokens["<|im_start|>"]
    assert ids[1] == tokenizer.special_tokens["<|im_end|>"]
