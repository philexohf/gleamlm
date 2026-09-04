"""数据管线测试 — 覆盖流式引擎（清洗 / 去重 / 质量 / QA / 配比切分）+ 端到端 7 阶段。

用法:
  python -m pytest tests/test_data_pipeline.py -v
"""

import os
import sys

import pytest

from gleamlm.data import (
    clean_file,
    clean_text,
    dedup_file,
    filter_qa,
    minhash_dedup_file,
    normalize,
    parse_qa,
    score_quality_file,
    score_text,
    stream_split,
)
from gleamlm.data.dataset import tokenize_and_group
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

GOOD_ZH = (
    "深度学习是机器学习的一个分支，它基于人工神经网络的学习方法，通过多层非线性"
    "变换从数据中自动学习特征表示。深度学习在计算机视觉、自然语言处理、语音识别"
    "等领域取得了突破性进展，成为人工智能的核心技术之一。"
)
GOOD_EN = (
    "Large language models are trained on massive amounts of text data scraped from "
    "the web, books, and other sources. The training process typically involves two "
    "phases: unsupervised pretraining on general text, followed by supervised "
    "fine-tuning on task-specific data to align the model with human instructions."
)
JUNK_SYMBOL = "!!!!!@@@@@#####$$$$$%%%%%^^^^^^^^^&&&&&*****((((())))))~~~~~:::::;;;;;;<<<<<>>>>>|||||"

Z2 = (
    "人工神经网络由大量相互连接的神经元组成，每个神经元接收输入信号，通过激活"
    "函数产生输出，网络通过反向传播算法调整连接权重以最小化损失函数。随着计算"
    "能力的提升和大规模数据的可用，神经网络的规模迅速扩大。"
)
Z3 = (
    "Transformer 由编码器和解码器两部分组成，编码器负责将输入序列映射为上下文"
    "相关的表示，解码器则根据这些表示逐步生成输出序列。每个编码器和解码器层都"
    "包含多头自注意力子层和前馈神经网络子层，并采用残差连接和层归一化。"
)
E2 = (
    "The transformer architecture introduced in 2017 revolutionized natural "
    "language processing by replacing recurrent neural networks with "
    "self-attention mechanisms. Self-attention allows each token in a sequence "
    "to directly attend to all other tokens, capturing long-range dependencies."
)


def _write_txt(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ════════════════════════════════════════════════════
# 归一化与清洗
# ════════════════════════════════════════════════════


def test_normalize_fullwidth():
    # NFKC: 全角→半角、大写→小写、空白压缩（指纹用，不输出改写）
    assert normalize("ＡＢＣｄｅｆ１２３，测试") == "abcdef123,测试"
    assert normalize("  Hello   World\t! ") == "hello world !"


def test_clean_text_filters():
    assert clean_text("", min_len=5) is None
    assert clean_text("短文本", min_len=10) is None
    # 纯符号（中英字符占比 < 30%）剔除
    assert clean_text(JUNK_SYMBOL) is None
    # 广告过滤
    ads = "咨询热线：13800138000 特价促销"
    assert clean_text(ads, filter_ads=True) is None
    assert clean_text(ads, filter_ads=False) is not None
    # URL 剥离
    url = "更多内容请访问 https://example.com 查看详情"
    cleaned = clean_text(url)
    assert cleaned is not None and "https://" not in cleaned


def test_clean_file_stream(tmp_path):
    f_in = str(tmp_path / "in.txt")
    f_out = str(tmp_path / "out.txt")
    _write_txt(f_in, [GOOD_ZH, "短", GOOD_EN])
    clean_file(f_in, f_out, min_len=10)
    assert _read_lines(f_out) == [GOOD_ZH, GOOD_EN]


# ════════════════════════════════════════════════════
# 去重
# ════════════════════════════════════════════════════


def test_dedup_exact_catches_fullwidth_variant(tmp_path):
    variant = GOOD_ZH.replace("，", ",")  # 全角逗号 → 半角逗号
    f_in = str(tmp_path / "in.txt")
    f_out = str(tmp_path / "out.txt")
    _write_txt(f_in, [GOOD_ZH, variant, GOOD_EN])
    dedup_file(f_in, f_out, mode="exact")
    lines = _read_lines(f_out)
    assert len(lines) == 2, "归一化去重应合并全角/半角变体"
    assert lines[0] == GOOD_ZH, "输出保留首个原文，不做 NFKC 改写"


def test_dedup_prefix_news(tmp_path):
    base = GOOD_EN
    f_in = str(tmp_path / "in.txt")
    f_out = str(tmp_path / "out.txt")
    _write_txt(f_in, [base, base + " 更多内容", base + " 不同结尾", GOOD_ZH])
    dedup_file(f_in, f_out, mode="prefix", prefix_len=100)
    lines = _read_lines(f_out)
    assert len(lines) == 2, "前三行前缀相同应合并为 1，GOOD_ZH 保留"


def test_dedup_simhash_returns_fingerprints(tmp_path):
    f_in = str(tmp_path / "in.txt")
    f_out = str(tmp_path / "out.txt")
    _write_txt(f_in, [GOOD_ZH, GOOD_EN])
    fps = dedup_file(f_in, f_out, mode="simhash", simhash_threshold=3)
    assert len(fps) == 2, "应返回保留文档的指纹集合（供跨源复用）"
    assert _read_lines(f_out) == [GOOD_ZH, GOOD_EN]


def test_minhash_dedup_file(tmp_path):
    near_dup = GOOD_ZH.replace("突破性进展", "重大进展")
    f_in = str(tmp_path / "in.txt")
    f_out = str(tmp_path / "out.txt")
    _write_txt(f_in, [GOOD_ZH, near_dup, GOOD_EN])
    minhash_dedup_file(f_in, f_out, threshold=0.8)
    lines = _read_lines(f_out)
    assert len(lines) == 2, "MinHash 应检出 1 条近似重复"
    assert GOOD_EN in lines, "无关文档不应被误删"


# ════════════════════════════════════════════════════
# 质量评分
# ════════════════════════════════════════════════════


def test_score_text():
    assert score_text(JUNK_SYMBOL) < 0.30, "纯符号垃圾应低分"
    assert score_text(GOOD_ZH) >= 0.30
    assert score_text(GOOD_EN) >= 0.30, "英文源不应被中文占比规则误杀"


def test_score_quality_file(tmp_path):
    f_in = str(tmp_path / "in.txt")
    f_out = str(tmp_path / "out.txt")
    _write_txt(f_in, [GOOD_ZH, GOOD_EN, JUNK_SYMBOL])
    score_quality_file(f_in, f_out, min_score=0.30)
    lines = _read_lines(f_out)
    assert len(lines) == 2
    assert JUNK_SYMBOL not in lines


# ════════════════════════════════════════════════════
# QA 专项
# ════════════════════════════════════════════════════


def test_parse_qa_formats():
    assert parse_qa("问题：什么是注意力？ 回答：注意力是一种加权机制。") == (
        "什么是注意力？",
        "注意力是一种加权机制。",
    )
    assert parse_qa("Q: 什么是梯度？ A: 梯度方向指向上升方向。") == (
        "什么是梯度？",
        "梯度方向指向上升方向。",
    )
    assert parse_qa("无关文本") == (None, None)


def test_filter_qa(tmp_path):
    f_in = str(tmp_path / "in.txt")
    f_out = str(tmp_path / "out.txt")
    good = "问题：什么是过拟合？ 回答：" + "模型在训练集表现好而测试集差。" * 5
    short = "问题：什么是过拟合？ 回答：不知道"
    url = "问题：什么是过拟合？ 回答：" + "模型记住了训练数据。" * 3 + " 详见 https://x.com/a"
    dup = "问题：什么是过拟合？ 回答：" + "训练集好测试集差。" * 5
    _write_txt(f_in, [good, short, url, dup])
    filter_qa(f_in, f_out, min_answer_len=20)
    assert len(_read_lines(f_out)) == 1, "短答/URL/重复问题应全部剔除"


# ════════════════════════════════════════════════════
# 配比切分
# ════════════════════════════════════════════════════


def test_stream_split_ratios(tmp_path):
    """max_chars 超过数据总量时全部行应被分派，train 占比 ≈ 90%。"""
    zh_f = str(tmp_path / "zh.txt")
    en_f = str(tmp_path / "en.txt")
    _write_txt(zh_f, [GOOD_ZH] * 10)
    _write_txt(en_f, [GOOD_EN] * 10)
    prefix = str(tmp_path / "mix")
    stats = stream_split(
        [zh_f, en_f],
        str(tmp_path),
        ratios=[0.7, 0.3],
        max_chars=100000,
        seed=42,
        output_prefix=prefix,
    )
    total = sum(len(_read_lines(f"{prefix}_{name}.txt")) for name in ("train", "valid", "test"))
    assert total == 20, "全部行应被分派"
    assert {s["rows"] for s in stats} == {10, 10}
    train = len(_read_lines(f"{prefix}_train.txt"))
    assert 0.8 < train / total < 0.98, "概率切分 train ≈ 90%"


def test_stream_split_cap_respects_ratios(tmp_path):
    """max_chars 是字符总量上限：达到即停止混合，且各源输出配比保持目标比例。

    数据总量 ≈4600 字符 > max_chars=2000，封顶触发时 zh:en 行数应接近 0.7:0.3
    （行长度不等，按字符加权后比例才精确；行数比例允许较大容差）。
    """
    zh_f = str(tmp_path / "zh.txt")
    en_f = str(tmp_path / "en.txt")
    _write_txt(zh_f, [GOOD_ZH] * 10)
    _write_txt(en_f, [GOOD_EN] * 10)
    prefix = str(tmp_path / "mix")
    stats = stream_split(
        [zh_f, en_f],
        str(tmp_path),
        ratios=[0.7, 0.3],
        max_chars=2000,
        seed=42,
        output_prefix=prefix,
    )
    total = sum(len(_read_lines(f"{prefix}_{name}.txt")) for name in ("train", "valid", "test"))
    assert total < 20, "max_chars=2000 应截断数据总量 (~4600 字符)"
    rows = {os.path.basename(s["source"]): s["rows"] for s in stats}
    assert rows["zh.txt"] == 10, "zh 源数据量小应先耗尽"
    zh_ratio = rows["zh.txt"] / (rows["zh.txt"] + rows["en.txt"])
    assert abs(zh_ratio - 0.7) < 0.15, f"封顶后 zh 行数占比应接近 70%，实际 {zh_ratio:.2f}"


def test_stream_split_ratios_mismatch(tmp_path):
    f = str(tmp_path / "a.txt")
    _write_txt(f, ["x"])
    with pytest.raises(ValueError):
        stream_split([f], str(tmp_path), ratios=[0.5, 0.5])


# ════════════════════════════════════════════════════
# tokenize 打包（读取端不变）
# ════════════════════════════════════════════════════


def test_tokenize_and_group_empty_protection(tmp_path):
    """全部短于 seq_len 时应返回空数据集而不是崩溃。"""
    f = str(tmp_path / "short.txt")
    _write_txt(f, ["短", "更短"])
    tok = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
    out = tokenize_and_group(f, tok, seq_len=32, num_proc=1)
    assert len(out) == 0


# ════════════════════════════════════════════════════
# 端到端：7 阶段一条线 + 断点续跑
# ════════════════════════════════════════════════════


def test_pipeline_e2e(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    docs = [
        "梯度下降是最常用的优化算法之一，它通过计算损失函数对参数的偏导数来更新权重，"
        "在深度学习中广泛用于训练神经网络模型，其变体包括带动量的梯度下降和 Adam 等自适应方法。",
        "注意力机制的核心思想是让模型在处理序列时动态地关注重要的位置，"
        "自注意力通过查询、键、值的矩阵运算计算位置间的关联权重，是 Transformer 的基础组件。",
        "词嵌入将离散的词语映射为稠密的实数向量，使得语义相近的词在向量空间中距离较近，"
        "预训练语言模型在此基础上进一步学习上下文相关的动态表示。",
        "正则化技术用于缓解过拟合，常见方法包括 L1 和 L2 权重衰减、Dropout 随机失活、"
        "早停以及数据增强，其目标是在训练误差和泛化误差之间取得平衡。",
    ]
    _write_txt(str(raw / "wiki_raw.txt"), [GOOD_ZH] * 4 + [docs[0]] + [JUNK_SYMBOL])
    _write_txt(str(raw / "news_raw.txt"), [Z2] * 4 + docs[1:4] + [JUNK_SYMBOL])

    args = [
        "pipeline",
        "--input", str(raw),
        "--sources", "wiki", "news",
        "--output-prefix", str(tmp_path / "out"),
        "--tokenizer", "bbpe",
        "--tokenizer-path", DEFAULT_TOKENIZER_PATH,
        "--minhash",
        "--workers", "1",
        "--skip-verify",
    ]
    monkeypatch.setattr(sys, "argv", args)
    import data_tools.pretrain.run_pipeline as pl

    pl.main()

    # step1-4 产物（wiki: 6 行 → 粗去重 3 → 清洗 2 → 质量 2 → 细去重 2）
    assert (raw / "wiki_raw_dedup.txt").exists()
    assert (raw / "wiki_clean.txt").exists()
    assert (raw / "wiki_quality.txt").exists()
    assert (raw / "wiki_dedup.txt").exists()
    assert len(_read_lines(str(raw / "wiki_clean.txt"))) == 2, "JUNK 应在清洗阶段剔除"
    assert len(_read_lines(str(raw / "wiki_dedup.txt"))) == 2
    # news: 6 → 粗去重 5 → 清洗 4 → 质量 4 → 细去重 4
    assert len(_read_lines(str(raw / "news_dedup.txt"))) == 4

    # step5-6 产物（小数据概率切分下 valid/test 可能为空，存在则必已打包）
    assert (tmp_path / "out" / "train.txt").exists()
    assert (tmp_path / "out" / "train.bin").exists()
    assert (tmp_path / "out" / "train.idx").exists()
    for split in ("valid", "test"):
        txt = tmp_path / "out" / f"{split}.txt"
        if txt.exists() and txt.stat().st_size > 0:  # 空文件 = 概率切分无行，跳过
            assert (tmp_path / "out" / f"{split}.bin").exists()
            assert (tmp_path / "out" / f"{split}.idx").exists()

    # 断点续跑：再跑一遍，所有阶段跳过、产物不变
    pl.main()
    assert len(_read_lines(str(raw / "wiki_dedup.txt"))) == 2
    assert len(_read_lines(str(raw / "news_dedup.txt"))) == 4
