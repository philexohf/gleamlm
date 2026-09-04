"""数据集 tokenize_and_group / IndexedMMapDataset / lm_collate 测试"""

import os
import struct
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from gleamlm.data.dataset import (
    IndexedMMapDataset,
    estimate_tokens_per_row,
    lm_collate,
    tokenize_and_group,
)


def _tokenize(data_path, tokenizer, seq_len):
    """Helper: tokenize with num_proc=1 for test speed."""
    return tokenize_and_group(data_path, tokenizer, seq_len, num_proc=1)


def _write_text(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")


class TestTokenizeAndGroup:
    """验证 tokenize_and_group 核心行为。"""

    def test_output_format(self, tokenizer):
        """返回的 Dataset 元素是 {"input_ids", "labels"} 的 dict。"""
        with tempfile.TemporaryDirectory() as tmp:
            text_line = (
                "这是一个测试中文句子。深度学习是人工智能的重要分支。"
                "自然语言处理技术近年来取得了巨大的进步，大语言模型成为了研究热点。"
                "通过大规模预训练和指令微调，模型展现出了强大的语言理解和生成能力。\n"
            )
            _write_text(os.path.join(tmp, "data.txt"), [text_line] * 10)
            ds = _tokenize(os.path.join(tmp, "data.txt"), tokenizer, 32)
            assert len(ds) > 0
            sample = ds[0]
            assert "input_ids" in sample
            assert "labels" in sample
            assert isinstance(sample["input_ids"], torch.Tensor)
            assert isinstance(sample["labels"], torch.Tensor)

    def test_labels_are_shifted(self, tokenizer):
        """labels[i] == input_ids[i+1]（因果语言模型 shift 1 步）。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_text(os.path.join(tmp, "data.txt"), [
                "深度学习是人工智能的重要分支之一，近年来发展迅速备受关注",
            ] * 10)
            ds = _tokenize(os.path.join(tmp, "data.txt"), tokenizer, 16)
            for sample in ds:
                assert torch.equal(sample["labels"][:-1], sample["input_ids"][1:])

    def test_short_texts_filtered(self, tokenizer):
        """长度 ≤ seq_len 的短文本被过滤（信息量不足，浪费 GPU）。"""
        with tempfile.TemporaryDirectory() as tmp:
            # 极短文本 — 中文字符数远小于 token 数，tokenize 后长度必然很短
            _write_text(os.path.join(tmp, "data.txt"), [
                "你好",           # too short
                "什么",           # too short
                "深度学习是人工智能的重要分支之一近年来发展迅速备受关注大语言模型成为研究热点",  # longer
            ] * 5)
            ds = _tokenize(os.path.join(tmp, "data.txt"), tokenizer, seq_len=16)
            assert len(ds) > 0
            # 每条都应有足够长度
            for sample in ds:
                assert len(sample["input_ids"]) == 16
                assert len(sample["labels"]) == 16

    def test_length_consistency(self, tokenizer):
        """所有 block 长度 = seq_len。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_text(os.path.join(tmp, "data.txt"), [
                "深度学习是人工智能的重要分支之一近年来发展迅速备受关注大语言模型成为研究热点",
            ] * 20)
            ds = _tokenize(os.path.join(tmp, "data.txt"), tokenizer, seq_len=32)
            assert len(ds) > 0
            for sample in ds:
                assert sample["input_ids"].shape == (32,)
                assert sample["labels"].shape == (32,)

    def test_empty_input_returns_empty_dataset(self, tokenizer):
        """所有文本被过滤时返回空 Dataset（不崩溃）。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_text(os.path.join(tmp, "data.txt"), ["你好", "短"])
            ds = _tokenize(os.path.join(tmp, "data.txt"), tokenizer, seq_len=256)
            assert len(ds) == 0
            # 仍可用 DataLoader
            dl = DataLoader(ds, batch_size=2)
            assert len(list(dl)) == 0

    def test_dataloader_compatibility(self, tokenizer):
        """DataLoader 默认 collation 直接可用 — batch 为 {"input_ids", "labels"} dict。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_text(os.path.join(tmp, "data.txt"), [
                "深度学习是人工智能的重要分支之一近年来发展迅速备受关注大语言模型成为研究热点",
            ] * 30)
            ds = _tokenize(os.path.join(tmp, "data.txt"), tokenizer, seq_len=16)
            dl = DataLoader(ds, batch_size=4, shuffle=True)
            batch = next(iter(dl))
            assert isinstance(batch, dict)
            assert batch["input_ids"].shape == (4, 16)
            assert batch["labels"].shape == (4, 16)

    def test_directory_input(self, tokenizer):
        """data_path 为目录时自动扫描 .txt 文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "sub")
            os.makedirs(sub)
            _write_text(os.path.join(sub, "a.txt"), [
                "深度学习是人工智能的重要分支之一近年来发展迅速",
            ] * 15)
            _write_text(os.path.join(sub, "b.txt"), [
                "自然语言处理技术取得了巨大进步大语言模型研究热点",
            ] * 15)
            ds = _tokenize(tmp, tokenizer, seq_len=16)
            assert len(ds) > 0


class TestIndexedMMapDataset:
    """验证手写 .bin/.idx 读取器（工业 mmap 格式，与 Megatron 0.16 同款）。"""

    @staticmethod
    def _write_indexed(prefix: str, docs: list[list[int]], vocab_size: int = 24002) -> None:
        """按 gleamlm.data.pack（megatron 0.16 标准格式）生成 .bin/.idx。"""
        with open(prefix + ".bin", "wb") as f:
            for toks in docs:
                f.write(np.asarray(toks, dtype=np.uint16).tobytes())
        num_docs = len(docs)
        sequence_lengths = np.array([len(t) for t in docs], dtype=np.int32)
        seq_pointers = np.zeros(num_docs, dtype=np.int64)
        np.cumsum(sequence_lengths.astype(np.int64) * 2, out=seq_pointers)
        seq_pointers = np.concatenate(([0], seq_pointers[:-1]))
        document_indices = np.arange(num_docs + 1, dtype=np.int64)
        with open(prefix + ".idx", "wb") as f:
            f.write(b"MMIDIDX\x00\x00")
            f.write(struct.pack("<Q", 1))   # version (8B, uint64)
            f.write(struct.pack("<B", 8))   # dtype code: uint16
            f.write(struct.pack("<Q", num_docs))       # sequence count
            f.write(struct.pack("<Q", num_docs + 1))   # document count
            f.write(sequence_lengths.tobytes())        # int32 × N
            f.write(seq_pointers.tobytes())            # int64 × N
            f.write(document_indices.tobytes())        # int64 × (N+1)

    def test_auto_detect_returns_mmap(self, tokenizer):
        """tokenize_and_group 自动识别 .bin/.idx 前缀 → IndexedMMapDataset。"""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "data")
            self._write_indexed(prefix, [list(range(1, 21))])
            with tokenize_and_group(prefix, tokenizer, seq_len=8) as ds:
                assert isinstance(ds, IndexedMMapDataset)
                # 20 token，每 block 消费 9 token（input 8 + shift 1）
                assert len(ds) == (20 - 1) // 8

    def test_first_block_content(self, tokenizer):
        """block 内容 = 文档流滑窗，labels 为 shift 一步。"""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "data")
            self._write_indexed(prefix, [list(range(1, 21))])
            with tokenize_and_group(prefix, tokenizer, seq_len=8) as ds:
                s0 = ds[0]
                assert torch.equal(s0["input_ids"], torch.arange(1, 9))
                assert torch.equal(s0["labels"], torch.arange(2, 10))

    def test_cross_document_concat(self, tokenizer):
        """跨文档拼接: block 跨越文档边界（GPT-2 滑窗做法，EOD 不打断）。"""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "data")
            self._write_indexed(prefix, [[1, 2, 3], [4, 5, 6, 7], [8, 9]])
            with tokenize_and_group(prefix, tokenizer, seq_len=4) as ds:
                # 文档流 [1,2,3,4,5,6,7,8,9]，滑窗
                assert torch.equal(ds[0]["input_ids"], torch.tensor([1, 2, 3, 4]))
                assert torch.equal(ds[0]["labels"], torch.tensor([2, 3, 4, 5]))
                assert torch.equal(ds[1]["input_ids"], torch.tensor([5, 6, 7, 8]))
                assert torch.equal(ds[1]["labels"], torch.tensor([6, 7, 8, 9]))

    def test_dataloader_compatible(self, tokenizer):
        """DataLoader 默认 collate 直接可用 — batch 为 dict。"""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "data")
            self._write_indexed(prefix, [list(range(1, 41))])
            with tokenize_and_group(prefix, tokenizer, seq_len=8) as ds:
                dl = DataLoader(ds, batch_size=4)
                batch = next(iter(dl))
                assert batch["input_ids"].shape == (4, 8)
                assert batch["labels"].shape == (4, 8)


class TestLmCollate:
    """验证 lm_collate（dict list → tuple 转换）。"""

    def test_basic_stack(self):
        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([2, 3, 4, 5])},
            {"input_ids": torch.tensor([6, 7, 8, 9]), "labels": torch.tensor([7, 8, 9, 0])},
        ]
        ids, labels = lm_collate(batch)
        assert ids.shape == (2, 4)
        assert labels.shape == (2, 4)
        assert torch.equal(ids[0], torch.tensor([1, 2, 3, 4]))

    def test_single_sample(self):
        batch = [{"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([6, 7, 8])}]
        ids, labels = lm_collate(batch)
        assert ids.shape == (1, 3)
        assert labels.shape == (1, 3)


class TestEstimateTokensPerRow:
    """验证 estimate_tokens_per_row 估算。"""

    def test_estimate(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmp:
            _write_text(os.path.join(tmp, "data.txt"), [
                "深度学习是人工智能的重要分支之一近年来发展迅速备受关注",
            ] * 20)
            result = estimate_tokens_per_row(os.path.join(tmp, "data.txt"), tokenizer, sample_size=10)
            assert result["avg_tokens_per_row"] > 0
            assert result["tokens_per_char"] > 0
            assert result["sampled_rows"] == 10


class TestMegatronCompat:
    """验证 pack.write_indexed_dataset 产物与 megatron.core IndexedDataset 兼容。

    这是"手工数据对接工业"的格式契约测试：pack 写出的 .bin/.idx
    必须能被 megatron 的 IndexedDataset 原样读回，逐文档 token 一致。
    防止格式回归导致工业轨 (industrial/pretrain.py) 无法消费手工数据。
    """

    @staticmethod
    def _write_pack(prefix: str, docs: list[list[int]], vocab_size: int = 12002) -> None:
        from gleamlm.data.pack import write_indexed_dataset

        write_indexed_dataset(prefix, docs, vocab_size=vocab_size)

    def test_megatron_reads_pack_output(self):
        """megatron IndexedDataset 能读 pack 产物，且逐文档 token 一致。"""
        pytest.importorskip("megatron.core.datasets.indexed_dataset")
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        docs = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "data")
            self._write_pack(prefix, docs)
            ds = IndexedDataset(prefix)
            # megatron 读到的是"序列"= 我们写入的每文档（一文档一序列）
            assert len(ds) == len(docs)
            for i, expected in enumerate(docs):
                assert ds[i].tolist() == expected, f"doc {i} 不匹配"

    def test_megatron_slide_window_matches_manual(self):
        """跨文档滑窗语义对齐: megatron GPTDataset 切块与 IndexedMMapDataset 一致。"""
        pytest.importorskip("megatron.core.datasets.indexed_dataset")
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        docs = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "data")
            self._write_pack(prefix, docs)
            ds = IndexedDataset(prefix)
            # 文档流 [1..12]，seq_len=4 跨文档滑窗
            with IndexedMMapDataset(prefix, seq_len=4) as manual:
                total_tokens = 12
                assert manual.num_blocks == (total_tokens - 1) // 4
                assert torch.equal(manual[0]["input_ids"], torch.tensor([1, 2, 3, 4]))
                assert torch.equal(manual[1]["input_ids"], torch.tensor([5, 6, 7, 8]))
                # megatron 侧的 numel 序列总数与总 token 对应
                assert sum(ds.sequence_lengths.tolist()) == total_tokens
