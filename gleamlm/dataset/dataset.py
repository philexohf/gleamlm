"""GleamLM 数据集。滑动窗口切分 + 分词 + 动态 padding + numpy memmap。"""

from __future__ import annotations

import contextlib
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
        self.stride = stride or max_seq_len * 3 // 4
        self.augment = augment

        text_file = os.path.join(data_dir, f"{split}.txt")
        id_prefix = f"{ids_prefix}_" if ids_prefix else ""
        ids_file = os.path.join(data_dir, f"{split}_{id_prefix}ids.npy")

        if not os.path.exists(text_file):
            raise FileNotFoundError(
                f"Data file not found: {text_file}\n"
                f"Please run data_tools/build_dataset.py first to prepare the data."
            )

        if os.path.exists(ids_file):
            print(f"Loading pre-tokenized {split} data from {ids_file}...")
            self.all_ids = np.load(ids_file, mmap_mode="r")
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
                all_ids: list[int] = []
                n_chunks = (total_chars + chunk_size - 1) // chunk_size

                print(
                    f"Tokenizing {split} data ({total_chars / 1e9:.2f}B chars, "
                    f"{n_chunks} chunks)..."
                )
                sys.stdout.flush()
                status_path = os.path.join(data_dir, f".tokenizing_{split}")
                for i in tqdm(
                    range(0, total_chars, chunk_size), desc=f"  {split}", file=sys.stdout
                ):
                    chunk = text[i : i + chunk_size]
                    ids = tokenizer.encode(chunk, add_bos=False, add_eos=False)
                    all_ids.extend(ids)
                    if (i // chunk_size) % 10 == 0:
                        with open(status_path, "w") as sf:
                            sf.write(f"{i + chunk_size}/{total_chars}")

                print(f"  Done: {len(all_ids)} tokens")

                ids_array = np.array(all_ids, dtype=np.uint32)
                np.save(ids_file, ids_array)
                del all_ids, text
                print(f"  Saved to {ids_file}")

                with contextlib.suppress(OSError):
                    os.remove(status_path)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            self.all_ids = np.load(ids_file, mmap_mode="r")
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


def collate_fn(batch: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
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
