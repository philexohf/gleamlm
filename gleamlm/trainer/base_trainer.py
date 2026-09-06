"""Pre-training shared utilities — atomic building blocks for training scripts.

Design principle: each function does ONE thing. Training loop control lives in
manual/*.py scripts, not here. This avoids the "big ball of mud" pattern where
a single train_one_epoch() bundles AMP + DDP + accumulation + logging — making
it impossible to partially reuse.

Functions are grouped by concern:
  - Reproducibility: set_seed
  - AMP:            create_scaler, optimizer_step
  - Distributed:    ddp_setup, ddp_cleanup
  - Eval:           evaluate
  - Persistence:    save_checkpoint, load_checkpoint
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from gleamlm.inference.generate import generate_response
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer

# Reproducibility


def set_seed(seed: int) -> None:
    """Fixed random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# AMP: 前向/反向用 BF16/FP16 计算，梯度回缩 FP32。
# BF16 保留 8 位指数不丢量级；GradScaler 解决 FP16 underflow。


def create_scaler() -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:  # type: ignore[name-defined]  # pyright: ignore[reportDeprecated]
    """AMP GradScaler with CPU fallback (compatible with PyTorch 1.x / 2.x)."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    return torch.cuda.amp.GradScaler()  # pyright: ignore[reportDeprecated]


# 梯度累积: 等效 batch = micro_batch × accumulate_grad，省显存；
# loss 需除以 accumulate_grad 归一化 (PyTorch 默认求和)。


# AMP 更新顺序不可打乱: unscale → clip → step → update → zero_grad。
# clip 必须在 unscale 之后，否则阈值作用于 ×scale 的梯度会失准。
def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    parameters: Any | None = None,
    clip_grad: float | None = None,
) -> None:
    """Atomic AMP update: unscale → clip → step → update → zero_grad.

    Args:
        optimizer:  AdamW (or any torch optimizer)
        scaler:     torch.amp.GradScaler
        parameters: model.parameters() for grad clipping (optional).
                    If None, clips optimizer's first param group params.
        clip_grad:  max grad norm. None or 0 → skip clipping.
    """
    scaler.unscale_(optimizer)
    if clip_grad and clip_grad > 0:
        params = parameters if parameters is not None else optimizer.param_groups[0]["params"]
        torch.nn.utils.clip_grad_norm_(params, clip_grad)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


# DDP: torchrun 设置 LOCAL_RANK/RANK/WORLD_SIZE 等环境变量；
# 必须 set_device(local_rank) 绑定各自 GPU，否则全部挤在 cuda:0。


def build_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float,
    *,
    wd_exclude_embedding: bool = True,
    wd_exclude_norm: bool = True,
) -> list[dict[str, Any]]:
    """构造带 weight-decay 分组的优化器参数组。

    工业实践（SmolLM3 / Megatron / OLMo2 一致）: embedding 与 norm 权重
    不应施加 weight decay ——
      - embedding 去 wd 提升训练稳定性（norm 自然稳定，见 OLMo2 / SmolLM3）
      - norm/bias 是 1-D 参数，wd 只对矩阵作用显著（Megatron 默认跳过
        len==1 参数）
    其余矩阵权重（W_q/k/v/o、W_gate/up/down）正常加 wd。

    用法:
        groups = build_optimizer_param_groups(model, weight_decay=0.1)
        optimizer = torch.optim.AdamW(groups, lr=..., betas=(0.9, 0.95))
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        skip = False
        if (
            wd_exclude_embedding
            and "embed" in name
            or wd_exclude_norm
            and (param.ndim <= 1 or "norm" in name)
        ):
            skip = True
        (no_decay if skip else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # 无参数时返回空组（非空模型下不会发生，防御性处理）
    return [g for g in groups if g["params"]]


def ddp_setup() -> None:
    """Initialize DDP process group. Call once at script start when LOCAL_RANK is set."""
    # Windows 无 NCCL (gloo 仅支持 CPU 梯度同步，多卡 CUDA 训练请用 Linux)
    backend = "gloo" if sys.platform == "win32" else "nccl"
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_cleanup() -> None:
    """Destroy DDP process group. Call in finally block or at script end."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """True if rank 0 or running outside DDP."""
    return not dist.is_initialized() or dist.get_rank() == 0


# Evaluation


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


# Checkpoint 必须含 model + optimizer + scaler + step：
# 只存权重会重置 LR 调度与 Adam 动量，续训 loss 会跳；
# _config 供下游精确重建模型结构。


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    path: str,
    step: int,
    epoch: int = 0,
    world_size: int = 1,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint with model + optimizer + scaler state.

    Args:
        model:      raw model (not DDP-wrapped). Caller must unwrap if needed.
        optimizer:  AdamW optimizer
        scaler:     AMP GradScaler
        path:       save path
        step:       global training step
        epoch:      current epoch number
        world_size: >1 if DDP (used to unwrap DDP modules internally)
        extra:      additional dict to merge (e.g. _config, loss, val_loss)
    """
    sd = model.module.state_dict() if world_size > 1 else model.state_dict()  # type: ignore[union-attr]
    state_dict: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": sd,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if extra:
        state_dict.update(extra)
    torch.save(state_dict, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    path: str,
    device: torch.device,
    world_size: int = 1,
) -> dict[str, Any]:
    """Load training checkpoint. Returns {step, epoch, ...} for resume.

    Handles both old keys (optimizer_state_dict / scaler_state_dict) and
    new keys (optimizer / scaler) for backward compatibility.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if world_size > 1:
        model.module.load_state_dict(ckpt["model_state_dict"])  # type: ignore[union-attr]
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    opt_key = "optimizer_state_dict" if "optimizer_state_dict" in ckpt else "optimizer"
    if opt_key in ckpt:
        optimizer.load_state_dict(ckpt[opt_key])

    scl_key = "scaler_state_dict" if "scaler_state_dict" in ckpt else "scaler"
    if scl_key in ckpt:
        scaler.load_state_dict(ckpt[scl_key])

    return {
        "step": ckpt.get("step", ckpt.get("global_step", 0)),
        "epoch": ckpt.get("epoch", 0),
        "batch": ckpt.get("batch", 0),
    }


# 生成评估（SFT / DPO 共用）


def evaluate_generations(
    model: GleamLMModel,
    tokenizer: BBPETokenizer,
    prompts: list[str],
    title: str = "生成评估",
) -> list[tuple[str, str]]:
    """用一组 prompt 做推理评估，打印结果并返回 (prompt, response) 对。"""
    model.eval()
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    results: list[tuple[str, str]] = []
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        results.append((prompt, response))
        print(f"\n[User] {prompt}")
        print(f"[Assistant] {response}")
        print("-" * 40)
    return results


# 显存: 参数 2B + 梯度 2B + Adam 状态 12B (每参数 bytes)；
# 激活值是最大占位者——checkpointing 省的是激活值不是参数。
