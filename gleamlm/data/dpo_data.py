"""DPO 数据集 — chosen/rejected pairs，prompt portion loss mask = 0。

Supports both single-turn and multi-turn preference data.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Dataset

from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.chatml import format_chatml


def dpad_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad chosen_ids and rejected_ids to max within-batch length + merge masks."""
    B = len(batch)
    pad_id = batch[0].get("_pad_id", 0)

    max_c = max(b["chosen_ids"].size(0) for b in batch)
    max_r = max(b["rejected_ids"].size(0) for b in batch)

    chosen_ids = torch.full((B, max_c), pad_id, dtype=torch.long)
    rejected_ids = torch.full((B, max_r), pad_id, dtype=torch.long)
    chosen_mask = torch.zeros(B, max_c - 1)
    rejected_mask = torch.zeros(B, max_r - 1)

    for i, b in enumerate(batch):
        Lc = b["chosen_ids"].size(0)
        Lr = b["rejected_ids"].size(0)
        chosen_ids[i, :Lc] = b["chosen_ids"]
        rejected_ids[i, :Lr] = b["rejected_ids"]
        chosen_mask[i, : b["chosen_mask"].size(0)] = b["chosen_mask"]
        rejected_mask[i, : b["rejected_mask"].size(0)] = b["rejected_mask"]

    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
        "_pad_id": pad_id,
    }


class DPODataset(Dataset):
    """DPO dataset: chosen/rejected pairs, prompt portion loss mask = 0.

    Supports two formats, auto-detected:

    Single-turn:
        {"instruction": "...", "chosen": "...", "rejected": "..."}

    Multi-turn:
        {"messages": [{"role":"user","content":"..."}, ...],
         "chosen": "...",
         "rejected": "..."}

    In multi-turn mode, `messages` provides the conversation history.
    `chosen` / `rejected` are the preferred / dispreferred continuations
    for the final assistant turn. Only the final answer tokens contribute
    to the DPO loss.
    """

    def __init__(self, data_path: str, tokenizer: BBPETokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        raw_samples: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    raw_samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping line {i} in {data_path}: {e}")

        if not raw_samples:
            raise ValueError(f"No valid samples in {data_path}")

        self.multiturn: bool = "messages" in raw_samples[0]

        self.samples: list[dict[str, Any]] = []
        for i, s in enumerate(raw_samples):
            has_messages = "messages" in s
            has_single = "instruction" in s
            has_pair = "chosen" in s and "rejected" in s

            if not has_pair:
                print(f"Warning: skipping line {i} in {data_path}: missing chosen/rejected")
                continue
            if not (has_messages or has_single):
                print(f"Warning: skipping line {i} in {data_path}: missing messages or instruction")
                continue

            self.samples.append(s)

        single_count = sum(1 for s in self.samples if "instruction" in s)
        multi_count = sum(1 for s in self.samples if "messages" in s)
        print(
            f"Loaded {len(self.samples)} DPO samples from {data_path} "
            f"({single_count} single-turn, {multi_count} multi-turn)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_bos=False, add_eos=False)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        chosen = s["chosen"]
        rejected = s["rejected"]

        if "messages" in s:
            messages = s["messages"]
            prompt_text = format_chatml(messages, add_generation_prompt=True)
            chosen_text = format_chatml(
                messages + [{"role": "assistant", "content": chosen}],
                add_generation_prompt=False,
            )
            rejected_text = format_chatml(
                messages + [{"role": "assistant", "content": rejected}],
                add_generation_prompt=False,
            )
        else:
            msgs = [{"role": "user", "content": s["instruction"]}]
            prompt_text = format_chatml(msgs, add_generation_prompt=True)
            chosen_text = format_chatml(
                msgs + [{"role": "assistant", "content": chosen}],
                add_generation_prompt=False,
            )
            rejected_text = format_chatml(
                msgs + [{"role": "assistant", "content": rejected}],
                add_generation_prompt=False,
            )

        prompt_ids = self._encode(prompt_text)
        chosen_ids = self._encode(chosen_text)
        rejected_ids = self._encode(rejected_text)

        P = len(prompt_ids)
        if len(chosen_ids) > self.max_seq_len:
            dropped = len(chosen_ids) - self.max_seq_len
            chosen_ids = chosen_ids[-self.max_seq_len :]
            P_c = max(0, P - dropped)
        else:
            P_c = P
        if len(rejected_ids) > self.max_seq_len:
            dropped = len(rejected_ids) - self.max_seq_len
            rejected_ids = rejected_ids[-self.max_seq_len :]
            P_r = max(0, P - dropped)
        else:
            P_r = P

        chosen_mask = torch.zeros(len(chosen_ids) - 1, dtype=torch.float32)
        rejected_mask = torch.zeros(len(rejected_ids) - 1, dtype=torch.float32)
        chosen_mask[max(0, min(P_c, len(chosen_ids)) - 1) :] = 1.0
        rejected_mask[max(0, min(P_r, len(rejected_ids)) - 1) :] = 1.0

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
            "_pad_id": self.tokenizer.pad_id,
        }
