"""数据加载 — streaming / 懒加载 / 自动缓存 + 工业 mmap 格式（.bin/.idx）。

大模型预训练数据必须切成固定长度 block:
  GPT-2 论文做法: 把整个文档流拼接成一个超长 token 序列，
  然后滑窗切成 [0:1024], [1024:2048], ... 边界可以跨文档。
  (语言模型 loss 只依赖前文，跨文档边界不影响训练)

两种输入格式（与 gleamlm.data.pack 统一）:
  A. 工业格式 .bin/.idx（推荐）: 一次预处理、多次复用，mmap 零内存加载。
     --data data/processed/wiki_zh   （gleamlm.data.pack 的输出前缀）
     tokenize_and_group 自动识别并走 IndexedMMapDataset（手写读取器，
     与 Megatron IndexedDataset 同格式，不依赖 megatron-core）
  B. 文本 .txt（小数据场景）: 每次训练现场 tokenize，纯 Python 逐行
     处理（零 datasets 依赖，与预处理引擎同范式）。

本实现步骤（B 分支）:
  1. 逐行读取文本 → tokenize → ids
  2. filter: 丢弃长度 ≤ seq_len 的短文本 (信息量不足，浪费 GPU)
  3. group: ids[:seq_len] 为 input, ids[1:seq_len+1] 为 label (shift 一步)
  4. 转 torch 格式 (DataLoader 直接可用)

为什么预训练统一用工业 mmap 格式？
  - tokenize 是最贵的步骤（BBPE 24K 几 GB 语料要数小时），一次预处理两轨共用
  - np.memmap 随机访问任意文档，内存占用 ≈ 0，TB 级语料也能训
  - 工业事实标准（Megatron/NeMo 同款 .bin/.idx）
  - 手写读取器即工业数据管道核心: mmap + 索引
"""

import bisect
import os
import random
import struct
from glob import glob
from typing import Any

import numpy as np
import torch

from torch.utils.data import Dataset as TorchDataset

# ── 工业 mmap 格式 (.bin/.idx) 常量 — 与 gleamlm.data.pack 写格式一致 ──
# megatron 0.16 header: 9 magic + 8 version + 1 dtype + 8 seq_count + 8 doc_count
_INDEX_MAGIC = b"MMIDIDX\x00\x00"
_IDX_HEADER_SIZE = 34
_DTYPE_CODE_UINT16 = 8


class IndexedMMapDataset(TorchDataset):
    """手写 .bin/.idx 读取器 — 工业 mmap 数据格式。

    与 megatron.core.datasets.indexed_dataset.IndexedDataset 完全同格式，
    但实现完全手写（纯 numpy/struct），零第三方依赖：

    .idx 布局:
      header 34B: 魔数(9) / version(8, uint64=1) / dtype_code(1) /
                  sequence_count(8, uint64) / document_count(8, uint64)
      sequence_lengths:   每序列 token 长度（N 个 int32）
      sequence_pointers:  每序列起始字节偏移（N 个 int64）
      document_indices:   文档边界序列序号（N+1 个 int64，以 0 开头）
    .bin 布局: 所有文档 token 连续写入（np.uint16）

    访问: 二分定位文档 → mmap 读 token 段 → 跨文档拼接成 block（GPT-2 滑窗）
    """

    def __init__(self, prefix: str, seq_len: int):
        idx_path = prefix + ".idx"
        bin_path = prefix + ".bin"
        if not (os.path.exists(idx_path) and os.path.exists(bin_path)):
            raise FileNotFoundError(f"IndexedDataset 缺失: {prefix}.bin/.idx")

        # ── 解析 .idx 头部 ───────────────────────────────────────
        with open(idx_path, "rb") as f:
            head = f.read(_IDX_HEADER_SIZE)
        assert head[:9] == _INDEX_MAGIC, f"非 .idx 文件: {idx_path}"
        assert head[9:17] == struct.pack("<Q", 1), f"不支持 .idx version: {idx_path}"
        dtype_code = head[17]
        assert dtype_code == _DTYPE_CODE_UINT16, f"仅支持 uint16 数据: code={dtype_code}"
        self.num_docs = struct.unpack("<Q", head[18:26])[0]

        # ── 解析索引区（header 后依次为 int32 lengths / int64 pointers(N) / int64 doc_idx(N+1)）──
        with open(idx_path, "rb") as f:
            f.seek(_IDX_HEADER_SIZE)
            seq_lengths = np.frombuffer(
                f.read(4 * self.num_docs), dtype=np.int32
            )
            seq_pointers = np.frombuffer(
                f.read(8 * self.num_docs), dtype=np.int64
            )
            # document_indices 有 N+1 个（含前导 0），这里只读前 N 个起始偏移即可
        # 每文档 token 起始偏移（字节 → token）
        self.doc_starts = seq_pointers // 2
        self.total_tokens = int(seq_lengths.sum())
        self.seq_len = seq_len

        # ── mmap 映射 .bin（不占进程内存，核心优势）──────────────
        self._mmap = np.memmap(bin_path, dtype=np.uint16, mode="r")

        # 每个 block 消费 seq_len+1 token（input=前 seq_len, label=shift 1）
        self.num_blocks = max((self.total_tokens - 1) // seq_len, 0)

    def __len__(self) -> int:
        return self.num_blocks

    def _read_tokens(self, start: int, length: int) -> np.ndarray:
        """从文档流读取 [start, start+length) 的 token，自动跨文档拼接。"""
        out = np.empty(length, dtype=np.uint16)
        pos, filled = start, 0
        while filled < length:
            i = bisect.bisect_right(self.doc_starts, pos) - 1
            doc_start = int(self.doc_starts[i])
            doc_end = int(self.doc_starts[i + 1]) if i + 1 < self.num_docs else self.total_tokens
            n = min(doc_end - pos, length - filled)
            out[filled:filled + n] = self._mmap[pos:pos + n]
            filled += n
            pos = doc_end  # 跳到下一文档继续（EOD 在文档尾，不打断拼接）
        return out

    def __getitem__(self, idx: int):
        span = self._read_tokens(idx * self.seq_len, self.seq_len + 1)
        input_ids = torch.from_numpy(span[:-1].copy()).long()
        labels = torch.from_numpy(span[1:].copy()).long()
        return {"input_ids": input_ids, "labels": labels}

    def close(self) -> None:
        """关闭底层 mmap — Windows 上不关闭会锁住 .bin 文件（删不掉/无法覆盖）。"""
        mm = getattr(self._mmap, "_mmap", None)
        if mm is not None:
            mm.close()

    def __enter__(self) -> "IndexedMMapDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _is_indexed_prefix(data_path: str) -> bool:
    """判断 --data 是否指向 .bin/.idx 前缀（工业格式）。"""
    if not os.path.exists(data_path + ".idx"):
        return False
    with open(data_path + ".idx", "rb") as f:
        return f.read(9) == _INDEX_MAGIC


def _resolve_data_files(data_path: str) -> list[str] | str:
    """将目录展开为 .txt 文件列表；已是文件则直接返回。"""
    if os.path.isdir(data_path):
        files = sorted(glob(os.path.join(data_path, "**", "*.txt"), recursive=True))
        if not files:
            raise ValueError(f"No .txt files found in directory: {data_path}")
        return files
    return data_path


def _read_text_lines(data_path: str | list[str]) -> list[str]:
    """读取 txt（路径/目录/文件列表）的所有非空行 — 零 datasets 依赖。"""
    files = data_path if isinstance(data_path, list) else _resolve_data_files(data_path)
    if isinstance(files, str):
        files = [files]
    lines: list[str] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    lines.append(s)
    return lines


class _TextTokenizeDataset(TorchDataset):
    """txt 分支 — 纯 Python 逐行 tokenize + 固定长度 block（零 datasets）。

    与原 datasets 版语义一致: 过滤短文本（≤ seq_len 丢弃），
    每行截取 ids[:seq_len] / ids[1:seq_len+1]（label shift 一步）。
    单进程急切加载（小数据场景，无需多进程）。
    """

    def __init__(self, data_path, tokenizer, seq_len, text_key="text", num_proc=8):
        _is_bbpe = hasattr(tokenizer, "encode") and hasattr(tokenizer, "get_vocab_size")
        blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for t in _read_text_lines(data_path):
            if _is_bbpe:
                ids = tokenizer.encode(t, add_bos=False, add_eos=False)
            else:
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            if len(ids) > seq_len:
                blocks.append(
                    (
                        torch.tensor(ids[:seq_len], dtype=torch.long),
                        torch.tensor(ids[1 : seq_len + 1], dtype=torch.long),
                    )
                )
        self._blocks = blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, idx: int):
        input_ids, labels = self._blocks[idx]
        return {"input_ids": input_ids, "labels": labels}


def tokenize_and_group(
    data_path: str | list[str],
    tokenizer: Any,
    seq_len: int,
    text_key: str = "text",
    num_proc: int = 8,
    streaming: bool = False,
):
    """
    统一预训练数据入口 — 工业格式 (.bin/.idx) 与文本格式自动识别。

    data_path 为 .bin/.idx 前缀（如 data/processed/wiki_zh）时:
      走 IndexedMMapDataset（mmap 零内存，与 Megatron/工业轨共用一份数据）
    data_path 为 .txt 路径/目录/文件列表时:
      走纯 Python 逐行 tokenize（小数据场景，零 datasets 依赖）

    返回值均支持 len()/getitem()/DataLoader，元素为
    {"input_ids", "labels"}（labels 为 shift 一步）。
    """
    if isinstance(data_path, str) and _is_indexed_prefix(data_path):
        return IndexedMMapDataset(data_path, seq_len)
    return _TextTokenizeDataset(data_path, tokenizer, seq_len, text_key=text_key, num_proc=num_proc)


def estimate_tokens_per_row(
    data_path: str | list[str],
    tokenizer: Any,
    text_key: str = "text",
    sample_size: int = 5000,
    seed: int = 42,
) -> dict:
    """采样估算行均 token 数 — 用于数据混合配比的 token/字符换算。

    输入为 txt 路径/目录/文件列表（纯 Python，零 datasets 依赖）。
    """
    lines = _read_text_lines(data_path)
    n = min(len(lines), sample_size)
    sample = random.Random(seed).sample(lines, n)
    _is_bbpe = hasattr(tokenizer, "encode") and hasattr(tokenizer, "get_vocab_size")

    total_tokens = 0
    total_chars = 0
    for t in sample:
        total_chars += len(t)
        if _is_bbpe:
            total_tokens += len(tokenizer.encode(t, add_bos=False, add_eos=False))
        else:
            total_tokens += len(tokenizer(t, add_special_tokens=False)["input_ids"])

    avg_tokens = total_tokens / max(n, 1)
    return {
        "avg_tokens_per_row": avg_tokens,
        "tokens_per_char": total_tokens / max(total_chars, 1),
        "sampled_rows": n,
        "total_chars": total_chars,
    }


def lm_collate(batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate for DataLoader: stack {"input_ids", "labels"} dicts → (input_ids, labels)."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return input_ids, labels
