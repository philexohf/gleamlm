"""GleamLM 训练脚本。支持 AMP + CosineAnnealing + AdamW + DDP + 断点续训"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard.writer import SummaryWriter

    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TB_AVAILABLE = False

import math
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

from gleamlm.dataset.dataset import LMDataset, collate_fn

# 添加当前目录和项目根目录到路径
from gleamlm.models.config import get_args
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.torch_utils import get_lr_cosine


def set_seed(seed):
    """固定随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    raw_criterion,
    device,
    epoch,
    args,
    global_step,
    writer,
    scaler,
):
    """训练一个 epoch，支持 AMP + 梯度累积。

    criterion: 带 label_smoothing 的 loss（用于反向传播）
    raw_criterion: 无平滑的 loss（用于日志，与 eval 可比）
    """
    model.train()
    total_smoothed = 0
    total_raw = 0
    num_batches = 0
    accumulate_grad = args.accumulate_grad

    pbar = (
        tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5, miniters=50)
        if args.local_rank == 0
        else train_loader
    )

    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # AMP 前向（兼容 PyTorch 1.x / 2.x）
        _amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        autocast_ctx = (
            torch.amp.autocast(_amp_device)
            if hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast()
        )
        with autocast_ctx:
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            # 无平滑 raw loss（仅用于日志，不参与反向传播）
            with torch.no_grad():
                raw_loss = raw_criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        loss = loss / accumulate_grad

        # DDP 梯度累积优化：非累加边界步骤跳过 all_reduce
        is_accum_step = (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(
            train_loader
        )
        sync_context = (
            model.no_sync() if (not is_accum_step and args.world_size > 1) else nullcontext()
        )
        with sync_context:
            scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if args.local_rank == 0 and writer is not None:
                current_loss = raw_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Train/Loss", current_loss, global_step)
                writer.add_scalar("Train/Learning_Rate", current_lr, global_step)

            global_step += 1

        total_smoothed += loss.item() * accumulate_grad
        total_raw += raw_loss.item()
        num_batches += 1

        if args.local_rank == 0 and batch_idx % args.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}", "lr": f"{lr:.6f}"})

    return total_raw / num_batches, global_step


@torch.no_grad()
def evaluate(model, val_loader, eval_criterion, device, pad_token_id=0, world_size=1):
    """验证集评估，返回 loss 和 PPL。DDP 下自动汇总所有 rank 的 loss"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for input_ids, target_ids in val_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = eval_criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        total_loss += loss.item()
        total_tokens += (target_ids != pad_token_id).sum().item()

    # DDP: 汇总所有 rank 的 loss 和 token 数
    if world_size > 1 and dist.is_initialized():
        loss_tensor = torch.tensor(total_loss, device=device)
        tokens_tensor = torch.tensor(total_tokens, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor.item()
        total_tokens = int(tokens_tensor.item())

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


def main():
    # 无 --config 时默认走 root configs/nano.yaml
    if "--config" not in sys.argv:
        _default_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "nano.yaml"
        )
        sys.argv = [sys.argv[0], "--config", _default_config] + sys.argv[1:]

    args = get_args()

    # DDP 初始化
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(args.local_rank)

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.local_rank == 0:
        print("WARNING: CUDA not available. Training on CPU will be extremely slow.")
        print(
            "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    set_seed(args.seed)

    if args.local_rank == 0:
        print("=" * 60)
        print("GleamLM 大模型训练")
        print("=" * 60)
        print(f"World size: {args.world_size} GPU(s)")
        print(
            f"Config: d_model={args.d_model}, layers={args.num_layers}, "
            f"heads={args.num_heads}(Q)/{args.num_kv_heads}(KV)"
        )
        print(f"Data dir: {args.data_dir}")

    train_txt = os.path.join(args.data_dir, "train.txt")
    os.path.join(args.data_dir, "valid.txt")

    if not os.path.exists(train_txt):
        raise FileNotFoundError(
            f"Training data not found: {train_txt}\n"
            f"Please prepare data first. For quick test, create a small text file."
        )

    tokenizer = BBPETokenizer.load(args.tokenizer_path)

    if args.local_rank == 0:
        print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    train_dataset = LMDataset(
        args.data_dir, tokenizer, args.max_seq_len, "train", max_chars=args.max_train_chars
    )
    val_dataset = LMDataset(args.data_dir, tokenizer, args.max_seq_len, "valid", augment=False)

    if args.world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
            pin_memory=True,
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
            num_workers=0,
            pin_memory=True,
        )

    model = GleamLMModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_id,
        tie_weights=True,
    ).to(device)

    if args.local_rank == 0:
        total, trainable = model.get_num_params()
        print(f"Model parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    # torch.compile 在 Windows 上暂无 Triton 支持
    # model = torch.compile(model)

    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=args.label_smoothing
    )

    # 无平滑 loss，用于日志显示，与 eval 直接可比
    raw_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    eval_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="sum")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay
    )

    # 总步数 = epochs * ceil(batches_per_epoch / accumulate_grad)
    total_steps = math.ceil(len(train_loader) / args.accumulate_grad) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: get_lr_cosine(step, total_steps, args.warmup_ratio)
    )

    # AMP 梯度缩放器（兼容 PyTorch 1.x / 2.x）
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)

        if args.world_size > 1:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))

        if args.local_rank == 0:
            print(f"Resuming from epoch {start_epoch}, step {global_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = None
    if args.local_rank == 0:
        if TB_AVAILABLE:
            log_dir = os.path.join(args.checkpoint_dir, "runs")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard: tensorboard --logdir {log_dir}")
        else:
            print("Warning: tensorboard not available")

    for epoch in range(start_epoch, args.epochs):
        if args.world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        # 训练一个 epoch
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            raw_criterion,
            device,
            epoch,
            args,
            global_step,
            writer,
            scaler,
        )

        # 验证（所有 rank 参与，DDP 下自动汇总）
        val_loss, val_ppl = evaluate(
            model.module if args.world_size > 1 else model,
            val_loader,
            eval_criterion,
            device,
            tokenizer.pad_id,
            args.world_size,
        )

        # DDP: 确保所有 rank 同步后再保存 checkpoint，防止其他 rank
        # 提前进入下一轮 set_epoch() 导致 all_reduce 死锁
        if args.world_size > 1:
            dist.barrier()

        if args.local_rank == 0:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_ppl={val_ppl:.2f}"
            )

            # TensorBoard
            if writer is not None:
                writer.add_scalar("Eval/Loss", val_loss, epoch)
                writer.add_scalar("Eval/Perplexity", val_ppl, epoch)
                writer.add_scalar("Eval/Train_Loss", train_loss, epoch)

            # 保存最佳模型
            if val_loss > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss

                ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.module.state_dict()
                        if args.world_size > 1
                        else model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "args": args,
                    },
                    ckpt_path,
                )
                print(f"  Saved best model (val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f})")

            # 每个 epoch 保存 checkpoint（方便对比各 epoch 生成质量）
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict()
                    if args.world_size > 1
                    else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
            )

    if args.world_size > 1:
        dist.destroy_process_group()

    if writer is not None:
        writer.close()

    if args.local_rank == 0:
        print("=" * 60)
        print("Training completed!")
        print(f"Best val_loss: {best_val_loss:.4f}, best val_ppl: {math.exp(best_val_loss):.2f}")
        print(f"Model saved to: {args.checkpoint_dir}")
        print(f"View TensorBoard: tensorboard --logdir {os.path.join(args.checkpoint_dir, 'runs')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
