"""
Xfind-Mini 语言模型数据集

实现纯文本续写数据集，支持滑动窗口切分、BPE 分词、动态 padding 和 numpy memmap。
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def _encode_chunk(model_path, chunk):
    """多线程分词辅助函数（SentencePiece C++ 底层释放 GIL）"""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp.encode(chunk, out_type=int)


class LMDataset(Dataset):
    """
    语言模型数据集（memmap 版本）

    分词后在磁盘上存为 numpy memmap，
    __getitem__ 时按索引切片返回，内存占用极小。

    数据格式：
        data_dir/
            train.txt       # 元始文本
            train_ids.npy   # 预分词 token ID（自动生成）
            valid.txt
            valid_ids.npy
            test.txt
            test_ids.npy
    """

    def __init__(self, data_dir, tokenizer, max_seq_len=1024, split="train", stride=None):
        """
        Args:
            data_dir: 数据目录
            tokenizer: XfindTokenizer 实例
            max_seq_len: 最大序列长度
            split: "train" / "valid" / "test"
            stride: 滑动窗口步长（默认 max_seq_len // 2）
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len * 3 // 4  # 75%，即 25% 重叠（减少模板强化）

        text_file = os.path.join(data_dir, f"{split}.txt")
        ids_file = os.path.join(data_dir, f"{split}_ids.npy")

        if not os.path.exists(text_file):
            raise FileNotFoundError(
                f"Data file not found: {text_file}\n"
                f"Please run tools/build_dataset.py first to prepare the data."
            )

        # 如果已有预分词文件，直接加载到内存（1.7GB 可放入 RAM，避免 memmap 随机 I/O）
        if os.path.exists(ids_file):
            print(f"Loading pre-tokenized {split} data from {ids_file}...")
            self.all_ids = np.load(ids_file, mmap_mode=None)  # 加载到 RAM，随机访问快
            self.total_tokens = len(self.all_ids)
            print(f"Loaded {split} data: {self.total_tokens} tokens")
        else:
            num_threads = min(os.cpu_count() or 4, 8)
            model_path = tokenizer.model_prefix + ".model"

            # 读取全部文本
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            total_chars = len(text)

            # 按线程数均分文本
            chunk_len = max(1, total_chars // num_threads)
            chunks = []
            for i in range(num_threads):
                start = i * chunk_len
                end = start + chunk_len if i < num_threads - 1 else total_chars
                chunks.append(text[start:end])
            del text  # 释放原始文本内存

            # 多线程并行编码（SentencePiece C++ 底层释放 GIL）
            encode_fn = partial(_encode_chunk, model_path)
            est_min = max(1, total_chars // (20_000_000 * num_threads))
            print(f"Tokenizing {split} data ({total_chars/1e9:.2f}B chars, "
                  f"{num_threads} threads, ~{est_min} min)...")
            print(f"  No console output expected during tokenization."
                  f"  Progress: ", end="", flush=True)

            all_ids = []
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(encode_fn, chunk) for chunk in chunks]
                for i, future in enumerate(futures):
                    all_ids.extend(future.result())
                    print(f"#{i+1} ", end="", flush=True)

            print(f"\n  Done: {len(all_ids)} tokens")

            # 保存为 numpy memmap 到磁盘
            ids_array = np.array(all_ids, dtype=np.uint32)
            np.save(ids_file, ids_array)
            self.all_ids = np.load(ids_file, mmap_mode='r')
            self.total_tokens = len(all_ids)
            del all_ids, ids_array
            print(f"  Saved to {ids_file}")

        # 计算样本数
        self.num_samples = max(0, (self.total_tokens - self.max_seq_len) // self.stride)
        print(f"Created {self.num_samples} samples for {split}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回单个样本（内存映射切片）

        Returns:
            tensor: [max_seq_len + 1]，前 max_seq_len 为输入，后 max_seq_len 为目标
        """
        start = idx * self.stride
        end = start + self.max_seq_len + 1
        ids = self.all_ids[start:end].astype(np.int64)
        return torch.from_numpy(ids)


def collate_fn(batch):
    """
    批次整理函数

    将变长序列 padding 到批次内最大长度，
    然后拆分为输入和目标（右移一位）。

    Args:
        batch: Tensor 列表，每个 [seq_len+1]

    Returns:
        input_ids: [batch, max_len]  输入 token IDs
        target_ids: [batch, max_len]  目标 token IDs（= input 右移一位）
    """
    max_len = max(len(sample) for sample in batch)

    padded = []
    for sample in batch:
        if len(sample) < max_len:
            padding = torch.zeros(max_len - len(sample), dtype=torch.long)
            sample = torch.cat([sample, padding])
        padded.append(sample)

    batch_tensor = torch.stack(padded)  # [batch, max_len + padding]

    # 输入：去掉最后一个 token
    # 目标：去掉第一个 token（右移一位）
    input_ids = batch_tensor[:, :-1]
    target_ids = batch_tensor[:, 1:]

    return input_ids, target_ids
