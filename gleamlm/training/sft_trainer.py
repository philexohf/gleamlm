"""SFT (Supervised Fine-Tuning) shared module. Extracted from nano/sft.py and lite/sft.py.

Provides SFTDataset, train_one_epoch_sft, evaluate_sft, and generate_response_sft.

Supports both single-turn and multi-turn conversation formats.
"""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gleamlm.inference.chatml import format_chatml
from gleamlm.inference.generate import generate_response
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.torch_utils import safe_autocast

SYSTEM_PROMPTS = [
    "你是一个有帮助的AI助手。",
    "你是一个友善的中文对话助手，请用简洁清晰的语言回答问题。",
    "你是一个知识渊博的助手，请准确回答问题。",
    "You are a helpful AI assistant.",
]


class SFTDataset(Dataset):
    """SFT dataset: JSONL -> ChatML format -> loss mask.

    Supports two formats, auto-detected from the first line:

    Single-turn (backward-compatible):
        {"instruction": "...", "output": "..."}

    Multi-turn:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

    Loss mask: only the LAST assistant turn contributes to loss.
    In multi-turn mode, all prior turns (including earlier assistant replies)
    are treated as context and masked.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BBPETokenizer,
        max_seq_len: int = 512,
        inject_system_ratio: float = 0.2,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.inject_system_ratio = inject_system_ratio
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        self.multiturn: bool = False
        self.data: list[dict[str, Any]] = []

        raw_lines: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping line {i} in {data_path}: {e}")
                    continue
                raw_lines.append(item)

        if not raw_lines:
            raise ValueError(f"No valid samples in {data_path}")

        first_has_messages = "messages" in raw_lines[0]

        if first_has_messages:
            self.multiturn = True
            for i, item in enumerate(raw_lines):
                msgs = item.get("messages")
                if not isinstance(msgs, list) or len(msgs) < 2:
                    print(f"Warning: skipping line {i} in {data_path}: invalid messages")
                    continue
                has_assistant = any(m.get("role") == "assistant" for m in msgs)
                if not has_assistant:
                    print(f"Warning: skipping line {i} in {data_path}: no assistant turn")
                    continue
                self.data.append({"messages": msgs})
            print(f"Loaded {len(self.data)} multi-turn SFT samples from {data_path}")
        else:
            self.multiturn = False
            for i, item in enumerate(raw_lines):
                if "messages" in item:
                    msgs = item.get("messages")
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        has_assistant = any(m.get("role") == "assistant" for m in msgs)
                        if has_assistant:
                            self.data.append({"messages": msgs})
                            continue
                if "instruction" in item and "output" in item:
                    self.data.append({"instruction": item["instruction"], "output": item["output"]})
                else:
                    print(f"Warning: skipping line {i} in {data_path}: unknown format")
            single_count = sum(1 for d in self.data if "instruction" in d)
            multi_count = sum(1 for d in self.data if "messages" in d)
            print(
                f"Loaded {len(self.data)} SFT samples from {data_path} "
                f"({single_count} single-turn, {multi_count} multi-turn)"
            )

        rng = random.Random(42)
        self._system_prompts: list[str] = []
        for _ in range(len(self.data)):
            if rng.random() < inject_system_ratio:
                self._system_prompts.append(rng.choice(SYSTEM_PROMPTS))
            else:
                self._system_prompts.append("")

    def __len__(self) -> int:
        return len(self.data)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_bos=False, add_eos=False)

    def _single_turn_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        instruction = item["instruction"]
        output = item["output"]
        system_prompt = self._system_prompts[idx]

        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": instruction})

        prompt_text = format_chatml(msgs, add_generation_prompt=True)
        full_text = format_chatml(
            msgs + [{"role": "assistant", "content": output}],
            add_generation_prompt=False,
        )

        prompt_ids = self._encode(prompt_text)
        full_ids = self._encode(full_text)

        P = len(prompt_ids)

        if len(full_ids) > self.max_seq_len:
            dropped = len(full_ids) - self.max_seq_len
            full_ids = full_ids[-self.max_seq_len :]
            P = max(0, P - dropped)

        input_ids = full_ids[:-1]
        labels = list(full_ids[1:])

        mask_end = min(P, len(labels))
        for i in range(mask_end - 1):
            labels[i] = -100

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    # --- multi-turn helpers ---

    def _multi_turn_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self.data[idx]["messages"]
        last = messages[-1]

        full_ids = self._encode(
            format_chatml(messages, add_generation_prompt=False)
        )
        prompt_ids = self._encode(
            format_chatml(messages[:-1], add_generation_prompt=True)
        )

        P = len(prompt_ids)

        if len(full_ids) > self.max_seq_len:
            dropped = len(full_ids) - self.max_seq_len
            full_ids = full_ids[-self.max_seq_len :]
            P = max(0, P - dropped)

        input_ids = full_ids[:-1]
        labels = list(full_ids[1:])

        mask_end = min(P, len(labels))
        for i in range(mask_end - 1):
            labels[i] = -100

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    # --- dispatch ---

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        if "messages" in item:
            return self._multi_turn_item(idx)
        return self._single_turn_item(idx)

    def collate_fn(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        attention_mask = (input_ids != self.pad_id).to(dtype=torch.long)
        return input_ids, labels, attention_mask


def train_one_epoch_sft(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    args: Any,
    global_step: int,
    scaler: Any,
    log_interval: int = 50,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch}", mininterval=3)

    for batch_idx, (input_ids, labels, attention_mask) in enumerate(pbar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        with safe_autocast():
            logits, _ = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        loss = loss / args.accumulate_grad
        world_size = getattr(args, "world_size", 1)
        is_accum = (batch_idx + 1) % args.accumulate_grad == 0 or (batch_idx + 1) == len(train_loader)
        sync_ctx = model.no_sync() if (not is_accum and world_size > 1) else nullcontext()
        with sync_ctx:
            scaler.scale(loss).backward()

        if (batch_idx + 1) % args.accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item() * args.accumulate_grad
        num_batches += 1

        if batch_idx % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * args.accumulate_grad:.4f}",
                    "lr": f"{lr:.2e}",
                }
            )

    return total_loss / num_batches, global_step


def evaluate_sft(
    model: torch.nn.Module,
    tokenizer: BBPETokenizer,
    test_prompts: list[str],
) -> list[tuple[str, str]]:
    model.eval()
    print("\n" + "=" * 60)
    print("SFT 生成评估")
    print("=" * 60)
    results: list[tuple[str, str]] = []
    for prompt in test_prompts:
        response = generate_response(model, tokenizer, prompt)
        results.append((prompt, response))
        print(f"\n[User] {prompt}")
        print(f"[Assistant] {response}")
        print("-" * 40)
    return results
