"""GleamLM 统一 SFT 指令微调脚本。通过 --variant 选择配置。

用法:
    python scripts/sft.py --variant nano
    python scripts/sft.py --variant lite --model_path checkpoints/lite/best_model.pt
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
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, cfg_to_namespace, load_config
from gleamlm.utils.torch_utils import get_lr_cosine

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="GleamLM SFT 指令微调")
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
    parser.add_argument("--max_seq_len", type=int, default=None, help="覆写序列长度")
    parser.add_argument("--data_path", type=str, default=None, help="覆写 SFT 数据路径")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="预训练模型路径 (默认: checkpoints/{variant}/best_model.pt)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH, help="BBPE 分词器目录"
    )
    parser.add_argument("--save_dir", type=str, default=None, help="SFT 模型保存目录")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 续训")

    cli_args = parser.parse_args()

    config_path = os.path.join(cli_args.config_dir, f"{cli_args.variant}.yaml")
    cfg = load_config(config_path)
    args = cfg_to_namespace(cfg, _ROOT_DIR)

    model_path = cli_args.model_path or os.path.join(args.checkpoint_dir, "best_model.pt")
    data_path = cli_args.data_path or args.sft_data_path
    save_dir = cli_args.save_dir or os.path.join(args.checkpoint_dir, "sft")

    lr = cli_args.lr if cli_args.lr is not None else args.sft_lr
    epochs = cli_args.epochs if cli_args.epochs is not None else args.sft_epochs
    batch_size = cli_args.batch_size if cli_args.batch_size is not None else args.sft_batch_size
    accumulate_grad = (
        cli_args.accumulate_grad
        if cli_args.accumulate_grad is not None
        else args.sft_accumulate_grad
    )
    max_seq_len = cli_args.max_seq_len if cli_args.max_seq_len is not None else args.sft_max_seq_len
    warmup_ratio = args.sft_warmup_ratio
    weight_decay = args.sft_weight_decay
    inject_system_ratio = args.sft_inject_system_ratio
    clip_grad = args.clip_grad

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant_name = cli_args.variant.upper()
    print("=" * 60)
    print(f"GleamLM-{variant_name} SFT 指令微调")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"Model: {model_path}")
    print(f"LR: {lr:.1e}, Epochs: {epochs}, Batch: {batch_size}, Seq: {max_seq_len}")

    tokenizer = BBPETokenizer.load(cli_args.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    model = GleamLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_id,
        tie_weights=args.tie_weights,
        use_flash_attn=args.use_flash_attn,
        use_qk_norm=args.use_qk_norm,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded pretrained model: {model_path}")
    total, trainable = model.get_num_params()
    print(f"Model params: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    train_dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        inject_system_ratio=inject_system_ratio,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    total_steps = math.ceil(len(train_loader) / accumulate_grad) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(step, total_steps, warmup_ratio, min_lr_ratio=0.05),
    )
    scaler = create_scaler()

    start_epoch = 0
    best_loss = float("inf")

    if cli_args.resume:
        print(f"\nResuming from: {cli_args.resume}")
        resume_ckpt = torch.load(cli_args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        global_step = resume_ckpt.get("global_step", 0)
        best_loss = resume_ckpt.get("train_loss", float("inf"))
        print(
            f"  Resumed at epoch {start_epoch}, global_step={global_step}, "
            f"best_loss={best_loss:.4f}"
        )

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

    os.makedirs(save_dir, exist_ok=True)
    if not cli_args.resume:
        global_step = 0

    train_ns = argparse.Namespace(
        epochs=epochs,
        batch_size=batch_size,
        accumulate_grad=accumulate_grad,
        lr=lr,
        clip_grad=clip_grad,
        max_seq_len=max_seq_len,
    )

    for epoch in range(start_epoch, epochs):
        train_loss, global_step = train_one_epoch_sft(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            train_ns,
            global_step,
            scaler,
        )

        print(f"\n--- SFT Epoch {epoch} 生成评估 ---")
        model.eval()
        evaluate_sft(model, tokenizer, eval_prompts)
        model.train()

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt_name = f"sft_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_loss,
                "args": train_ns,
            },
            os.path.join(save_dir, ckpt_name),
        )

        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(save_dir, "sft_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": train_ns,
                },
                best_path,
            )
            print(f"  Saved best SFT model (loss={train_loss:.4f}) -> {best_path}")

    print("\n" + "=" * 60)
    print("SFT 训练完成，最终生成评估")
    print("=" * 60)
    model.eval()
    evaluate_sft(model, tokenizer, eval_prompts)
    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
