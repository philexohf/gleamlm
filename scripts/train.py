"""GleamLM 统一预训练脚本。通过 --variant 选择配置。

用法:
    python scripts/train.py --variant nano
    python scripts/train.py --variant lite --epochs 3 --lr 0.001
    python scripts/train.py --variant nano --load_checkpoint checkpoints/nano/checkpoint_epoch_1.pt
"""

import argparse
import math
import os

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

from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.training.base_trainer import (
    create_optimizer_and_scheduler,
    create_scaler,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    wrap_for_distributed,
)
from gleamlm.utils.config import cfg_to_namespace, load_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="GleamLM 预训练")
    parser.add_argument(
        "--variant", type=str, choices=["nano", "lite", "pro"], required=True, help="模型变体"
    )
    parser.add_argument(
        "--config_dir", type=str, default=os.path.join(_ROOT_DIR, "configs"), help="YAML 配置目录"
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆写训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="覆写学习率")
    parser.add_argument("--batch_size", type=int, default=None, help="覆写 batch size")
    parser.add_argument("--accumulate_grad", type=int, default=None, help="覆写梯度累积步数")
    parser.add_argument("--max_seq_len", type=int, default=None, help="覆写序列最大长度")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="覆写 warmup 比例")
    parser.add_argument("--max_train_chars", type=int, default=None, help="覆写训练字符上限")
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="断点续训 checkpoint 路径"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="覆写数据目录")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="覆写 checkpoint 输出目录")

    cli_args = parser.parse_args()
    config_path = os.path.join(cli_args.config_dir, f"{cli_args.variant}.yaml")
    cfg = load_config(config_path)
    args = cfg_to_namespace(cfg, _ROOT_DIR)

    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.lr is not None:
        args.lr = cli_args.lr
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.accumulate_grad is not None:
        args.accumulate_grad = cli_args.accumulate_grad
    if cli_args.max_seq_len is not None:
        args.max_seq_len = cli_args.max_seq_len
    if cli_args.warmup_ratio is not None:
        args.warmup_ratio = cli_args.warmup_ratio
    if cli_args.max_train_chars is not None:
        args.max_train_chars = cli_args.max_train_chars
    if cli_args.load_checkpoint:
        args.load_checkpoint = cli_args.load_checkpoint
    if cli_args.data_dir:
        args.data_dir = cli_args.data_dir
    if cli_args.checkpoint_dir:
        args.checkpoint_dir = cli_args.checkpoint_dir

    set_seed(args.seed)

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    if args.world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)

    if device.type == "cpu" and args.local_rank == 0:
        print("WARNING: CUDA not available. Training on CPU will be extremely slow.")

    variant_name = cli_args.variant.upper()
    if args.local_rank == 0:
        print("=" * 60)
        print(f"GleamLM-{variant_name} {args.d_model}d x {args.num_layers}L 训练")
        print("=" * 60)
        print(
            f"  d_model={args.d_model}, layers={args.num_layers}, "
            f"heads={args.num_heads}(Q)/{args.num_kv_heads}(KV), "
            f"seq_len={args.max_seq_len}"
        )
        print(
            f"  lr={args.lr:.0e}, type={args.type}, "
            f"batch={args.batch_size}, accum={args.accumulate_grad}, "
            f"epochs={args.epochs}"
        )
        print(
            f"  Flash Attn: {args.use_flash_attn}, BF16: {args.bf16}, Z-Loss: {args.z_loss_weight}"
        )
        print(f"  Data: {args.data_dir}")
        print(f"  Checkpoint: {args.checkpoint_dir}")

    train_txt = os.path.join(args.data_dir, "train.txt")
    if not os.path.exists(train_txt):
        raise FileNotFoundError(
            f"Training data not found: {train_txt}\n"
            f"Please prepare data first or specify --data_dir."
        )

    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    if args.local_rank == 0:
        print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    train_dataset = LMDataset(
        args.data_dir,
        tokenizer,
        args.max_seq_len,
        "train",
        max_chars=args.max_train_chars,
        ids_prefix=args.ids_prefix,
    )
    val_dataset = LMDataset(
        args.data_dir,
        tokenizer,
        args.max_seq_len,
        "valid",
        max_chars=args.max_train_chars,
        augment=False,
        ids_prefix=args.ids_prefix,
    )

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
        tie_weights=args.tie_weights,
        use_flash_attn=args.use_flash_attn,
        use_qk_norm=args.use_qk_norm,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    ).to(device)

    if args.local_rank == 0:
        total, trainable = model.get_num_params()
        print(f"Model parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    if args.world_size > 1:
        model = wrap_for_distributed(model, args)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=args.label_smoothing
    )

    optimizer, scheduler = create_optimizer_and_scheduler(model, train_loader, args)
    scaler = create_scaler()

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.load_checkpoint}")
        ckpt_info = load_checkpoint(
            model, optimizer, scheduler, scaler, args.load_checkpoint, device, args.world_size
        )
        start_epoch = ckpt_info["start_epoch"]
        global_step = ckpt_info["global_step"]
        best_val_loss = ckpt_info["best_val_loss"]
        if args.local_rank == 0:
            print(f"Resumed from epoch {start_epoch}, step {global_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    writer = None
    if args.local_rank == 0 and TB_AVAILABLE:
        log_dir = os.path.join(args.checkpoint_dir, "runs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: tensorboard --logdir {log_dir}")

    for epoch in range(start_epoch, args.epochs):
        if args.world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            args,
            global_step,
            writer,
            scaler,
        )

        val_loss, val_ppl = evaluate(
            model.module if args.world_size > 1 else model,
            val_loader,
            device,
            tokenizer.pad_id,
            args.world_size,
        )

        if args.world_size > 1:
            dist.barrier()

        if args.local_rank == 0:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
            )

            if writer is not None:
                writer.add_scalar("Eval/Loss", val_loss, epoch)
                writer.add_scalar("Eval/Perplexity", val_ppl, epoch)
                writer.add_scalar("Eval/Train_Loss", train_loss, epoch)

            if val_loss > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    os.path.join(args.checkpoint_dir, "best_model.pt"),
                    epoch,
                    global_step,
                    args.world_size,
                    extra={
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "args": args,
                    },
                )
                print(f"  Saved best model (val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f})")

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                epoch,
                global_step,
                args.world_size,
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


if __name__ == "__main__":
    main()
