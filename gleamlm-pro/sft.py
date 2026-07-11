"""GleamLM-Pro 126M SFT 指令微调脚本。基于 best_model.pt，ChatML 格式 + loss mask

用法：
    python sft.py --data_path ./data/sft_data.jsonl --model_path ./checkpoints/best_model.pt
"""

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.training.base_trainer import create_scaler, set_seed
from gleamlm.training.sft_trainer import (
    SFTDataset,
    evaluate_sft,
    train_one_epoch_sft,
)
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH
from gleamlm.utils.torch_utils import get_lr_cosine

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")


# 参数解析
def get_sft_args():
    parser = argparse.ArgumentParser(description="GleamLM-Pro 126M SFT 指令微调")

    # 数据与模型路径
    parser.add_argument("--data_path", type=str, default="./gleamlm-pro/data/sft_data.jsonl",
                        help="SFT JSONL 数据路径")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(_CHECKPOINT_DIR, "best_model.pt"),
                        help="预训练模型路径")
    parser.add_argument("--tokenizer_path", type=str,
                        default=DEFAULT_TOKENIZER_PATH,
                        help="BBPE 分词器目录路径")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(_CHECKPOINT_DIR, "sft"),
                        help="SFT 模型保存目录")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate_grad", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6, help="SFT 学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    # 系统消息注入
    parser.add_argument("--inject_system_ratio", type=float, default=0.2,
                        help="系统消息随机注入比例")

    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 续训（如 ./checkpoints/sft/sft_epoch_0.pt）")

    # Pro 126M 模型架构（默认值）
    parser.add_argument("--vocab_size", type=int, default=12002)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_kv_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", dest="use_flash_attn", action="store_false")

    return parser.parse_args()


# 主函数
def main():
    args = get_sft_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("GleamLM-Pro 126M SFT 指令微调")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"LR: {args.lr:.1e}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

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

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded pretrained model: {args.model_path}")
    total, trainable = model.get_num_params()
    print(f"Model params: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    train_dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        inject_system_ratio=args.inject_system_ratio,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTDataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    total_steps = math.ceil(len(train_loader) / args.accumulate_grad) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(step, total_steps, args.warmup_ratio, min_lr_ratio=0.05),
    )
    scaler = create_scaler()

    # 断点续训
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        global_step = resume_ckpt.get("global_step", 0)
        best_loss = resume_ckpt.get("train_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, global_step={global_step}, best_loss={best_loss:.4f}")

    # 评估提示词
    eval_prompts = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？",
        "请用一句话描述北京的秋天。",
        "写一首关于春天的五言诗。",
        "请解释一下什么是光合作用。",
    ]

    print("\n--- SFT 前生成基线 ---")
    model.eval()
    evaluate_sft(model, tokenizer, eval_prompts)
    model.train()

    os.makedirs(args.save_dir, exist_ok=True)
    if not args.resume:
        global_step = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step = train_one_epoch_sft(
            model, train_loader, optimizer, scheduler, device,
            epoch, args, global_step, scaler,
        )

        print(f"\n--- SFT Epoch {epoch} 生成评估 ---")
        model.eval()
        evaluate_sft(model, tokenizer, eval_prompts)
        model.train()

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt_name = f"sft_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "args": args,
        }, os.path.join(args.save_dir, ckpt_name))

        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(args.save_dir, "sft_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": args,
            }, best_path)
            print(f"  Saved best SFT model (loss={train_loss:.4f}) -> {best_path}")

    print("\n" + "=" * 60)
    print("SFT 训练完成，最终生成评估")
    print("=" * 60)
    model.eval()
    evaluate_sft(model, tokenizer, eval_prompts)
    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
