"""DPO (Direct Preference Optimization) shared module. Extracted from nano/dpo.py and lite/dpo.py.

Provides DPODataset, dpad_collate, compute_log_probs, dpo_loss, get_reference_logps,
train_one_epoch_dpo, generate_response_dpo, evaluate_dpo.

Supports both single-turn and multi-turn preference data.
"""

from __future__ import annotations

import json
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


def compute_log_probs(
    logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-token log probabilities, masked. Returns [B]."""
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs_token = log_probs_all[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (log_probs_token * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    term = (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)
    return -F.logsigmoid(beta * term).mean()


@torch.no_grad()
def get_reference_logps(
    ref_model: torch.nn.Module,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute chosen and rejected log-probs from frozen reference model."""
    ref_model.eval()
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    with safe_autocast():
        c_logits, _ = ref_model(chosen_ids)
        r_logits, _ = ref_model(rejected_ids)
    ref_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
    ref_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)
    return ref_cho, ref_rej


def train_one_epoch_dpo(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    beta: float,
    device: torch.device,
    args: Any,
) -> float:
    model.train()
    ref_model.eval()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="DPO")
    for batch_idx, batch in enumerate(pbar):
        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)

        ref_cho, ref_rej = get_reference_logps(
            ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask
        )

        amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        with safe_autocast():
            c_logits, _ = model(
                chosen_ids,
                attention_mask=(chosen_ids != batch["_pad_id"]).to(dtype=torch.long),
            )
            r_logits, _ = model(
                rejected_ids,
                attention_mask=(rejected_ids != batch["_pad_id"]).to(dtype=torch.long),
            )

        policy_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
        policy_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)

        loss = dpo_loss(policy_cho, policy_rej, ref_cho.detach(), ref_rej.detach(), beta)

        loss = loss / args.accumulate_grad
        world_size = getattr(args, "world_size", 1)
        is_accum = (batch_idx + 1) % args.accumulate_grad == 0 or (batch_idx + 1) == len(dataloader)
        sync_ctx = model.no_sync() if (not is_accum and world_size > 1) else nullcontext()
        with sync_ctx:
            scaler.scale(loss).backward()

        if (batch_idx + 1) % args.accumulate_grad == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.accumulate_grad
        n_batches += 1

        if (batch_idx + 1) % args.accumulate_grad == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item() * args.accumulate_grad:.4f}", lr=f"{lr:.2e}")

    return total_loss / max(n_batches, 1)


def evaluate_dpo(model: torch.nn.Module, tokenizer: BBPETokenizer) -> None:
    eval_prompts = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？",
        "请用一句话描述北京的秋天。",
        "写一首关于春天的五言诗。",
        "请解释一下什么是光合作用。",
    ]
    model.eval()
    print("\n" + "=" * 60)
    print("DPO 生成评估")
    print("=" * 60)
    for prompt in eval_prompts:
        print(f"\n[User] {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"[Assistant] {response}")
        print("-" * 40)
