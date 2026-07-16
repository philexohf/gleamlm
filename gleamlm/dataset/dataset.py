"""GleamLM 数据集。滑动窗口切分 + 分词 + 动态 padding + numpy memmap。"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gleamlm.tokenizer.tokenizer import BBPETokenizer


class LMDataset(Dataset):
    """LM 数据集（memmap 版本）。分词后存磁盘，按索引切片"""

    def __init__(
        self,
        data_dir: str,
        tokenizer: BBPETokenizer,
        max_seq_len: int,
        split: str = "train",
        stride: int | None = None,
        max_chars: int | None = None,
        ids_prefix: str = "",
        augment: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len * 3 // 4
        self.augment = augment

        text_file = os.path.join(data_dir, f"{split}.txt")
        id_prefix = f"{ids_prefix}_" if ids_prefix else ""
        ids_file = os.path.join(data_dir, f"{split}_{id_prefix}ids.npy")

        if not os.path.exists(text_file):
            raise FileNotFoundError(
                f"Data file not found: {text_file}\n"
                f"Please run data_tools/pretrain/build.py first to prepare the data."
            )

        if os.path.exists(ids_file):
            print(f"Loading pre-tokenized {split} data from {ids_file}...")
            self.all_ids = np.load(ids_file, mmap_mode="r", allow_pickle=False)
            self.total_tokens = len(self.all_ids)
            print(f"Loaded {split} data: {self.total_tokens} tokens")
        else:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                with open(text_file, encoding="utf-8") as f:
                    text = f.read() if max_chars is None else f.read(max_chars)
                total_chars = len(text)

                chunk_size = 256 * 1024
                n_chunks = (total_chars + chunk_size - 1) // chunk_size

                print(
                    f"Tokenizing {split} data ({total_chars / 1e9:.2f}B chars, "
                    f"~{n_chunks} chunks)..."
                )
                sys.stdout.flush()
                # 预分配 numpy 数组，避免 Python list 的 28 字节/元素 OOM
                # BBPE CJK 分词 ~1.5 字符/token，取 2x 余量安全上限
                estimated = int(total_chars * 2.0)
                all_ids = np.empty(estimated, dtype=np.uint32)
                pos = 0
                pos_c = 0
                pbar = tqdm(total=total_chars, desc=f"  {split}", unit="chars")
                while pos_c < total_chars:
                    end_c = min(pos_c + chunk_size, total_chars)
                    if end_c < total_chars:
                        nl = text.rfind("\n", pos_c, end_c)
                        if nl >= 0:
                            end_c = nl + 1
                    chunk = text[pos_c:end_c]
                    ids = tokenizer.encode(chunk, add_bos=False, add_eos=False)
                    n = len(ids)
                    if pos + n > len(all_ids):
                        all_ids = np.resize(all_ids, len(all_ids) * 2)
                    all_ids[pos : pos + n] = ids
                    pos += n
                    pbar.update(end_c - pos_c)
                    pos_c = end_c
                pbar.close()

                all_ids = all_ids[:pos]
                print(f"  Done: {pos} tokens")

                np.save(ids_file, all_ids)
                del all_ids, text
                print(f"  Saved to {ids_file}")

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            self.all_ids = np.load(ids_file, mmap_mode="r", allow_pickle=False)
            self.total_tokens = len(self.all_ids)
            print(f"Loaded {split} data: {self.total_tokens} tokens")

        self.num_samples = max(0, (self.total_tokens - self.max_seq_len - 1) // self.stride + 1)
        if self.num_samples == 0:
            print(
                f"*** WARNING: 0 samples for {split}! "
                f"total_tokens={self.total_tokens}, max_seq_len={self.max_seq_len}. "
                f"Data may be too small or max_seq_len too large."
            )
        print(f"Created {self.num_samples} samples for {split}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        end = start + self.max_seq_len + 1
        ids = self.all_ids[start:end].astype(np.int64)
        tensor = torch.from_numpy(ids)

        if self.augment and random.random() < 0.10:
            min_len = self.max_seq_len // 2
            trunc = random.randint(min_len, self.max_seq_len)
            tensor = tensor[:trunc]

        return tensor


def collate_fn(
    batch: list[torch.Tensor], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padding to max within-batch length → input / target / attention_mask"""
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
    attention_mask = (input_ids != pad_id).to(dtype=torch.long)

    return input_ids, target_ids, attention_mask
