"""PPL (Perplexity) 评估"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.tokenizer.tokenizer import BBPETokenizer


@dataclass
class PPLResult:
    """PPL 评估结果"""

    loss: float
    ppl: float
    tokens: int
    batches: int
    dataset_name: str
    model_params_m: float = 0.0
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PPLResult({self.dataset_name}: "
            f"loss={self.loss:.4f}, ppl={self.ppl:.2f}, tokens={self.tokens})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "loss": round(self.loss, 6),
            "ppl": round(self.ppl, 2),
            "tokens": self.tokens,
            "batches": self.batches,
            "model_params_m": self.model_params_m,
            **self.extra,
        }


@torch.no_grad()
def _compute_raw_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    pad_token_id: int = 0,
    max_batches: int | None = None,
) -> tuple[float, int, int]:
    """Core PPL helper: iterate loader, accumulate sum(loss) and both total tokens.

    Returns (total_loss, total_tokens, n_batches). Callers handle avg_loss, ppl, and DDP all-reduce.
    """
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_token_id)

    pbar = tqdm(data_loader, desc="Eval", mininterval=5)
    for input_ids, target_ids, attention_mask in pbar:
        if max_batches and n_batches >= max_batches:
            break
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits, _ = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        total_loss += loss.item()
        total_tokens += (target_ids != pad_token_id).sum().item()
        n_batches += 1

        pbar.set_postfix(
            {"loss": f"{loss.item() / max(1, (target_ids != pad_token_id).sum().item()):.4f}"}
        )

    return total_loss, total_tokens, n_batches


def compute_ppl(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    pad_token_id: int = 0,
    max_batches: int | None = None,
) -> PPLResult:
    """Compute PPL: sum(loss) / sum(tokens) → exp()."""
    total_loss, total_tokens, n_batches = _compute_raw_loss(
        model,
        data_loader,
        device,
        pad_token_id,
        max_batches,
    )
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    return PPLResult(
        loss=avg_loss, ppl=ppl, tokens=total_tokens, batches=n_batches, dataset_name="eval"
    )


def evaluate_ppl(
    model: nn.Module,
    tokenizer: BBPETokenizer,
    data_dir: str,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    device: str = "cuda",
    dataset: str = "test",
    max_batches: int | None = None,
    ids_prefix: str = "",
    world_size: int = 1,
    local_rank: int = 0,
) -> PPLResult:
    """评估模型在指定数据集上的困惑度。"""
    ds = LMDataset(data_dir, tokenizer, max_seq_len, dataset, ids_prefix=ids_prefix, augment=False)

    if world_size > 1:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, num_replicas=world_size, rank=local_rank)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
            num_workers=0,
        )

    result = compute_ppl(model, dl, torch.device(device), tokenizer.pad_id, max_batches)
    result.dataset_name = dataset

    total, _ = model.get_num_params()
    result.model_params_m = total / 1e6

    if world_size > 1:
        total_loss_sum = result.loss * result.tokens
        loss_t = torch.tensor([total_loss_sum], device=device)
        tokens_t = torch.tensor([result.tokens], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_t, op=dist.ReduceOp.SUM)
        result.tokens = tokens_t.item()
        result.loss = loss_t.item() / max(1, result.tokens)
        result.ppl = math.exp(result.loss)

    return result


def evaluate_multiple(
    model: nn.Module,
    tokenizer: BBPETokenizer,
    data_dir: str,
    datasets: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, PPLResult]:
    """对多个数据集依次评估，返回 {name: PPLResult}。"""
    if datasets is None:
        datasets = ["valid", "test"]

    results: dict[str, PPLResult] = {}
    for ds_name in datasets:
        txt_path = os.path.join(data_dir, f"{ds_name}.txt")
        if not os.path.exists(txt_path):
            print(f"  Skip {ds_name}: no data file")
            continue
        result = evaluate_ppl(model, tokenizer, data_dir, dataset=ds_name, **kwargs)
        results[ds_name] = result
    return results
