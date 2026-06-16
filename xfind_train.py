"""Xfind-Mini 训练脚本。支持 AMP + CosineAnnealing + AdamW + DDP + 断点续训"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TB_AVAILABLE = False

import os
import sys
import random
import numpy as np
import math
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.xfind_config import get_args
from models.xfind_model import XfindModel
from xfind_dataset import LMDataset, collate_fn
from tokenizer.xfind_tokenizer import build_tokenizer


def set_seed(seed):
    """固定随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_cosine(step, total_steps, warmup_ratio=0.01, min_lr_ratio=0.1):
    """Cosine Annealing + Warmup 学习率调度，返回乘数 0~1"""
    warmup_steps = int(total_steps * warmup_ratio)

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device,
                    epoch, args, global_step, writer, scaler):
    """训练一个 epoch，支持 AMP + 梯度累积"""
    model.train()
    total_loss = 0
    num_batches = 0
    accumulate_grad = args.accumulate_grad

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5, miniters=50) if args.local_rank == 0 else train_loader

    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # AMP 前向
        with torch.amp.autocast('cuda'):
            logits, _ = model(input_ids)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

        loss = loss / accumulate_grad
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if args.local_rank == 0 and writer is not None:
                current_loss = loss.item() * accumulate_grad
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Train/Loss', current_loss, global_step)
                writer.add_scalar('Train/Learning_Rate', current_lr, global_step)

            global_step += 1

        total_loss += loss.item() * accumulate_grad
        num_batches += 1

        if args.local_rank == 0 and batch_idx % args.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item() * accumulate_grad:.4f}",
                "lr": f"{lr:.6f}"
            })

    return total_loss / num_batches, global_step


@torch.no_grad()
def evaluate(model, val_loader, device, pad_token_id=0):
    """验证集评估，返回 loss 和 PPL"""
    model.eval()
    total_loss = 0
    num_batches = 0

    # 验证用标准 CE（无 label_smoothing）确保 PPL 准确
    eval_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    for input_ids, target_ids in val_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = eval_criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


def main():
    args = get_args()

    # DDP 初始化
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.local_rank == 0:
        print("=" * 60)
        print("Xfind-Mini 大模型训练")
        print("=" * 60)
        print(f"World size: {args.world_size} GPU(s)")
        print(f"Config: d_model={args.d_model}, layers={args.num_layers}, "
              f"heads={args.num_heads}(Q)/{args.num_kv_heads}(KV)")
        print(f"Data dir: {args.data_dir}")

    # 构建分词器
    train_txt = os.path.join(args.data_dir, "train.txt")
    valid_txt = os.path.join(args.data_dir, "valid.txt")

    if not os.path.exists(train_txt):
        raise FileNotFoundError(
            f"Training data not found: {train_txt}\n"
            f"Please prepare data first. For quick test, create a small text file."
        )

    text_files = []
    for f in [train_txt, valid_txt]:
        if os.path.exists(f):
            text_files.append(f)

    tokenizer = build_tokenizer(
        text_files,
        vocab_size=args.vocab_size,
        model_prefix=args.tokenizer_path
    )

    if args.local_rank == 0:
        print(f"Tokenizer vocab size: {len(tokenizer)}")

    train_dataset = LMDataset(args.data_dir, tokenizer, args.max_seq_len, "train")
    val_dataset = LMDataset(args.data_dir, tokenizer, args.max_seq_len, "valid")

    # DataLoader
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = XfindModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_id,
        tie_weights=True
    ).to(device)

    if args.local_rank == 0:
        total, trainable = model.get_num_params()
        print(f"Model parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    # torch.compile 在 Windows 上暂无 Triton 支持
    # model = torch.compile(model)

    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    # 总步数 = epochs * batches_per_epoch / accumulate_grad
    total_steps = len(train_loader) * args.epochs // args.accumulate_grad
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(step, total_steps, args.warmup_ratio)
    )

    # AMP 梯度缩放器
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)

        if args.world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))

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
            model, train_loader, optimizer, scheduler, criterion, device,
            epoch, args, global_step, writer, scaler
        )

        # 验证
        if args.local_rank == 0:
            val_loss, val_ppl = evaluate(
                model.module if args.world_size > 1 else model,
                val_loader, device, tokenizer.pad_id
            ) if len(val_loader) > 0 else (0.0, 1.0)

            print(f"Epoch {epoch}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_ppl={val_ppl:.2f}")

            # TensorBoard
            if writer is not None:
                writer.add_scalar('Eval/Loss', val_loss, epoch)
                writer.add_scalar('Eval/Perplexity', val_ppl, epoch)
                writer.add_scalar('Eval/Train_Loss', train_loss, epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'args': args,
                }, save_path)
                print(f"  Saved best model (val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f})")

            # 每个 epoch 保存 checkpoint（方便对比各 epoch 生成质量）
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    if args.world_size > 1:
        dist.destroy_process_group()

    if writer is not None:
        writer.close()

    if args.local_rank == 0:
        print("=" * 60)
        print("Training completed!")
        print(f"Best val_loss: {best_val_loss:.4f}")
        print(f"Model saved to: {args.checkpoint_dir}")
        print(f"View TensorBoard: tensorboard --logdir {os.path.join(args.checkpoint_dir, 'runs')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
