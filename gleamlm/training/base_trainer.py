"""Pre-training shared utilities — seed, AMP, distributed wrapping,
optimizer/scheduler, evaluate, checkpoint save/load, training loop.
"""

from __future__ import annotations

import math
import random
from contextlib import nullcontext
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gleamlm.utils.torch_utils import get_lr_cosine, get_lr_wsd, safe_autocast


def set_seed(seed: int) -> None:
    """Fixed random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_scaler() -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:  # pyright: ignore[reportDeprecated]
    """AMP GradScaler with CPU fallback (compatible with PyTorch 1.x / 2.x)."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[name-defined]
    return torch.cuda.amp.GradScaler()  # pyright: ignore[reportDeprecated]


def wrap_for_distributed(model: nn.Module, args: Any) -> nn.Module:
    if args.world_size > 1:
        if getattr(args, "use_fsdp", False):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            from gleamlm.models.model import DecoderBlock

            model = FSDP(
                model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={DecoderBlock},
                ),
                use_orig_params=True,
            )
        else:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank]
            )
    return model


def create_optimizer_and_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
    args: Any,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(getattr(args, "betas", (0.9, 0.95))),
        eps=getattr(args, "eps", 1e-8),
        weight_decay=args.weight_decay,
    )
    total_steps = math.ceil(len(train_loader) / args.accumulate_grad) * args.epochs
    lr_type = getattr(args, "type", "cosine")
    warmup_ratio = getattr(args, "warmup_ratio", 0.02)
    min_lr_ratio = getattr(args, "min_lr_ratio", 0.1)
    if lr_type == "wsd":

        def lr_fn(step: int) -> float:
            return get_lr_wsd(
                step,
                total_steps,
                warmup_ratio,
                getattr(args, "stable_ratio", 0.80),
                min_lr_ratio,
            )
    else:

        def lr_fn(step: int) -> float:
            return get_lr_cosine(step, total_steps, warmup_ratio, min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    return optimizer, scheduler


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    pad_token_id: int = 0,
    world_size: int = 1,
) -> tuple[float, float]:
    """Validate and return (avg_loss, ppl). Aggregates across DDP ranks."""
    torch.cuda.empty_cache()

    from gleamlm.evaluation.ppl import _compute_raw_loss

    total_loss, total_tokens, _ = _compute_raw_loss(model, val_loader, device, pad_token_id)

    if world_size > 1 and dist.is_initialized():
        t_loss = torch.tensor(total_loss, device=device)
        t_tokens = torch.tensor(total_tokens, device=device)
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_tokens, op=dist.ReduceOp.SUM)
        total_loss = t_loss.item()
        total_tokens = int(t_tokens.item())

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    path: str,
    epoch: int,
    global_step: int,
    world_size: int,
    extra: dict[str, Any] | None = None,
) -> None:
    state_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        **(extra or {}),
    }
    torch.save(state_dict, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    path: str,
    device: torch.device,
    world_size: int,
) -> dict[str, Any]:
    """Returns {start_epoch, global_step, best_val_loss}."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if world_size > 1:
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return {
        "start_epoch": checkpoint.get("epoch", 0) + 1,
        "global_step": checkpoint.get("global_step", 0),
        "best_val_loss": checkpoint.get("val_loss", float("inf")),
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args: Any,
    global_step: int,
    writer: Any,
    scaler: Any,
) -> tuple[float, int]:
    """训练一个 epoch，支持 AMP + 梯度累积 + Z-Loss + DDP。

    criterion: 带 label_smoothing 的 CrossEntropyLoss（用于反向传播）。
    日志和返回的 loss 使用无平滑 raw CE，保证与 eval 可比。
    """
    model.train()
    total_raw_ce = 0.0
    num_batches = 0
    prev_loss: float | None = None
    accumulate_grad = args.accumulate_grad
    z_loss_weight = getattr(args, "z_loss_weight", 0.0)
    spike_threshold = getattr(args, "loss_spike_threshold", 6.0)
    pad_id: int = criterion.ignore_index  # type: ignore[assignment]
    amp_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16

    pbar = (
        tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5)
        if args.local_rank == 0
        else train_loader
    )

    for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)

        with safe_autocast(enabled=True, dtype=amp_dtype):
            logits, _ = model(input_ids, attention_mask=attention_mask)
            ce_loss = criterion(logits.reshape(-1, args.vocab_size), target_ids.reshape(-1))
            raw_ce = F.cross_entropy(
                logits.reshape(-1, args.vocab_size),
                target_ids.reshape(-1),
                ignore_index=pad_id,
                label_smoothing=0.0,
            )
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = z_loss_weight * (log_z**2).mean()
            loss = (ce_loss + z_loss) / accumulate_grad

        is_accum_step = (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(
            train_loader
        )
        sync_context = (
            model.no_sync() if (not is_accum_step and args.world_size > 1) else nullcontext()
        )
        with sync_context:
            scaler.scale(loss).backward()

        if is_accum_step:
            cur_loss = raw_ce.item()

            if torch.isnan(raw_ce) or torch.isinf(raw_ce):
                print(f"\n[FATAL] NaN/Inf loss at step {global_step}, aborting")
                raise RuntimeError(f"NaN/Inf loss at step {global_step}")

            if prev_loss is not None and cur_loss > prev_loss * spike_threshold:
                print(f"\n[WARN] Loss spike at step {global_step}: "
                      f"{prev_loss:.3f} -> {cur_loss:.3f}, skipping update")
                optimizer.zero_grad()
                prev_loss = cur_loss
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if grad_norm > 10.0:
                print(f"\n[WARN] High grad norm at step {global_step}: {grad_norm:.1f}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if writer is not None and args.local_rank == 0:
                writer.add_scalar("Train/Loss", cur_loss, global_step)
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

            global_step += 1
            prev_loss = cur_loss

            if isinstance(pbar, tqdm):
                pbar.set_postfix(
                    {
                        "loss": f"{raw_ce.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                    }
                )

        total_raw_ce += raw_ce.item()
        num_batches += 1

    return total_raw_ce / max(1, num_batches), global_step
