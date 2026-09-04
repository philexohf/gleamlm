"""RLHF 数据集 — 每行一个 prompt (jsonl)，RL 阶段不需要标签。

GRPO / PPO 共用。
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset


class RLHFDataset(Dataset):
    """RLHF 基础数据集 — 每行一个 prompt (jsonl)。RL 阶段不需要标签。

    支持可选 ground_truth 列（规则 reward 用），无则 None。
    返回 dict {"prompt": str, "ground_truth": str|None}。
    """

    def __init__(self, data_path: str, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len
        self.data: list[dict] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = item.get("prompt", item.get("instruction", ""))
                if prompt:
                    self.data.append({
                        "prompt": prompt,
                        "ground_truth": item.get("ground_truth"),
                    })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def tokenize_prompts(prompts: list[str], tokenizer, max_seq_len: int) -> torch.Tensor:
    """批量 tokenize + pad prompts。

    预留 max_new_tokens 空间 (8 token buffer)，防止 prompt + 生成超过 max_seq_len。
    """
    if not prompts:
        raise ValueError("tokenize_prompts: 空 prompt 列表")
    ids = [tokenizer.encode(p, add_bos=False) for p in prompts]
    ids = [t[: max_seq_len - 8] for t in ids]
    max_len = max(len(t) for t in ids)
    padded = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in ids]
    return torch.tensor(padded)
