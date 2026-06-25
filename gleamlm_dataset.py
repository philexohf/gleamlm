"""烁珑GleamLM 数据集。滑动窗口切分 + 分词 + 动态 padding + numpy memmap。"""

import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from tqdm import tqdm


class LMDataset(Dataset):
    """LM 数据集（memmap 版本）。分词后存磁盘，按索引切片"""
    def __init__(self, data_dir, tokenizer, max_seq_len=1024, split="train",
                 stride=None, max_chars=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len * 3 // 4  # 75%即25%重叠

        text_file = os.path.join(data_dir, f"{split}.txt")
        ids_file = os.path.join(data_dir, f"{split}_ids.npy")

        if not os.path.exists(text_file):
            raise FileNotFoundError(
                f"Data file not found: {text_file}\n"
                f"Please run tools/build_dataset.py first to prepare the data."
            )

        # 如果已有预分词文件，直接加载到内存
        if os.path.exists(ids_file):
            print(f"Loading pre-tokenized {split} data from {ids_file}...")
            self.all_ids = np.load(ids_file, mmap_mode='r')
            self.total_tokens = len(self.all_ids)
            print(f"Loaded {split} data: {self.total_tokens} tokens")
        else:
            # 读取文本（可限制字符数以节省编码时间）
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            total_chars = len(text)

            # 分块编码（进度条显示）
            chunk_size = 256 * 1024  # 256K 字符每块
            all_ids = []
            n_chunks = (total_chars + chunk_size - 1) // chunk_size

            print(f"Tokenizing {split} data ({total_chars/1e9:.2f}B chars, "
                  f"{n_chunks} chunks)...")
            sys.stdout.flush()
            # 写状态文件，方便外部监控进度
            status_path = os.path.join(data_dir, f".tokenizing_{split}")
            for i in tqdm(range(0, total_chars, chunk_size), desc=f"  {split}",
                          file=sys.stdout):
                chunk = text[i:i + chunk_size]
                ids = tokenizer.encode(chunk, add_bos=False, add_eos=False)
                all_ids.extend(ids)
                # 每 10 chunk 更新状态文件
                if (i // chunk_size) % 10 == 0:
                    with open(status_path, "w") as sf:
                        sf.write(f"{i+chunk_size}/{total_chars}")

            print(f"  Done: {len(all_ids)} tokens")

            ids_array = np.array(all_ids, dtype=np.uint32)
            np.save(ids_file, ids_array)
            self.all_ids = ids_array
            self.total_tokens = len(all_ids)
            del all_ids, text
            print(f"  Saved to {ids_file}")

        self.num_samples = max(0, (self.total_tokens - self.max_seq_len - 1) // self.stride + 1)
        print(f"Created {self.num_samples} samples for {split}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_seq_len + 1
        ids = self.all_ids[start:end].astype(np.int64)
        tensor = torch.from_numpy(ids)

        # 随机截断增强：10% 概率随机缩短序列，让模型学会从任意位置续写
        if random.random() < 0.10:
            min_len = self.max_seq_len // 2
            trunc = random.randint(min_len, self.max_seq_len)
            tensor = tensor[:trunc]

        return tensor


def collate_fn(batch, pad_id):
    """Padding 到 batch 内最大长度，右移一位拆分为 input 和 target"""
    max_len = max(len(sample) for sample in batch)

    padded = []
    for sample in batch:
        if len(sample) < max_len:
            padding = torch.full((max_len - len(sample),), pad_id, dtype=torch.long)
            sample = torch.cat([sample, padding])
        padded.append(sample)

    batch_tensor = torch.stack(padded)

    input_ids = batch_tensor[:, :-1]
    target_ids = batch_tensor[:, 1:]

    return input_ids, target_ids
