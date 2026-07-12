"""GleamLM-Lite 87M 训练脚本。Cosine LR + Flash Attention + Z-Loss + Dropout=0"""

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
)


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_script_dir)

    import argparse

    parser = argparse.ArgumentParser(description="GleamLM-Lite 87M Training")

    # 路径
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join(_root_dir, "data", "lite_data")
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.path.join(_root_dir, "gleamlm", "tokenizer", "checkpoints", "bbpe_12k"),
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=os.path.join(_script_dir, "checkpoints")
    )
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_root_dir, "configs", "lite.yaml"),
        help="YAML 配置文件路径",
    )

    # 模型结构
    parser.add_argument("--vocab_size", type=int, default=12002)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_kv_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", dest="use_flash_attn", action="store_false")
    parser.set_defaults(use_flash_attn=True)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate_grad", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--z_loss_weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # 精度与日志
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--max_train_chars", type=int, default=5_300_000_000)
    parser.add_argument(
        "--ids_prefix", type=str, default="", help="预分词文件前缀，用于区分不同分词器"
    )

    # 配置加载
    config_args, _ = parser.parse_known_args()

    if config_args.config:
        from gleamlm.utils.config import load_config_as_args

        args = load_config_as_args(config_args.config, cli_overrides=True)
        defaults = {
            a.dest: parser.get_default(a.dest)
            for a in parser._actions
            if a.dest != "help" and a.dest != "config"
        }
        for key, val in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, val)
    else:
        args = parser.parse_args()

    set_seed(args.seed)

    # DDP 初始化
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    if args.world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        print("=" * 60)
        print("GleamLM-Lite 87M 训练")
        print("=" * 60)
        print(
            f"  模型: d={args.d_model}, L={args.num_layers}, heads={args.num_heads}/{args.num_kv_heads}"
        )
        print(f"  词表: {args.vocab_size}, seq={args.max_seq_len}, dropout={args.dropout}")
        print(
            f"  批次: {args.batch_size} x accum {args.accumulate_grad} = effective {args.batch_size * args.accumulate_grad}"
        )
        print(
            f"  学习率: {args.lr:.0e}, Cosine (warmup={args.warmup_ratio}, min_lr_ratio={args.min_lr_ratio})"
        )
        print(f"  Flash Attn: {args.use_flash_attn}, Z-Loss: {args.z_loss_weight}")
        print(f"  设备: {device}")

    tokenizer = BBPETokenizer.load(args.tokenizer_path)

    if args.local_rank == 0:
        print(f"分词器词表大小: {tokenizer.get_vocab_size()}")

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
        tie_weights=True,
        use_flash_attn=args.use_flash_attn,
    ).to(device)

    if args.local_rank == 0:
        total, trainable = model.get_num_params()
        print(f"模型参数: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=args.label_smoothing
    )

    optimizer, scheduler = create_optimizer_and_scheduler(model, train_loader, args)
    scaler = create_scaler()

    # 断点续训
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        if args.local_rank == 0:
            print(f"加载 checkpoint: {args.load_checkpoint}")
        ckpt_info = load_checkpoint(model, optimizer, scheduler, scaler,
                                     args.load_checkpoint, device, args.world_size)
        start_epoch = ckpt_info["start_epoch"]
        global_step = ckpt_info["global_step"]
        best_val_loss = ckpt_info["best_val_loss"]
        if args.local_rank == 0:
            print(f"  从 epoch {start_epoch}, step {global_step} 续训")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    writer = None
    if args.local_rank == 0:
        if TB_AVAILABLE:
            log_dir = os.path.join(args.checkpoint_dir, "runs")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard: tensorboard --logdir {log_dir}")
        else:
            print("警告: tensorboard 不可用")

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

            if writer is not None:
                writer.add_scalar("Eval/Loss", val_loss, epoch)
                writer.add_scalar("Eval/Perplexity", val_ppl, epoch)
                writer.add_scalar("Eval/Train_Loss", train_loss, epoch)

            if val_loss > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    os.path.join(args.checkpoint_dir, "best_model.pt"),
                    epoch, global_step, args.world_size,
                    extra={"train_loss": train_loss, "val_loss": val_loss, "val_ppl": val_ppl, "args": args},
                )
                print(f"  保存最佳模型 (val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f})")

            save_checkpoint(
                model, optimizer, scheduler, scaler,
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                epoch, global_step, args.world_size,
            )

    if args.world_size > 1:
        dist.destroy_process_group()

    if writer is not None:
        writer.close()

    if args.local_rank == 0:
        print("=" * 60)
        print("训练完成!")
        print(f"最佳 val_loss: {best_val_loss:.4f}, 最佳 val_ppl: {math.exp(best_val_loss):.2f}")
        print(f"模型保存在: {args.checkpoint_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
