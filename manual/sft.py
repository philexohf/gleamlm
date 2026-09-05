"""GleamLM 统一 SFT 指令微调脚本。通过 --variant 选择配置。

用法:
    python manual/sft.py --variant nano
    python manual/sft.py --variant lite --model_path checkpoints/lite/best_model.pt
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.base_trainer import create_scaler, evaluate_generations, optimizer_step, set_seed
from gleamlm.data.sft_data import SFTDataset
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, cfg_to_namespace, load_config
from gleamlm.trainer.schedulers import get_lr_cosine, get_lr_wsd
from gleamlm.utils.torch_utils import clean_state_dict, safe_autocast

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
    parser.add_argument(
        "--lr_scheduler", type=str, choices=["cosine", "wsd"], default="cosine",
        help="学习率调度器类型"
    )
    parser.add_argument("--stable_ratio", type=float, default=0.80, help="WSD stable 阶段占比")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05, help="最小学习率比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (对齐 pretrain.py 的 --seed)")

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
    lr_scheduler = cli_args.lr_scheduler
    stable_ratio = cli_args.stable_ratio
    min_lr_ratio = cli_args.min_lr_ratio

    set_seed(cli_args.seed)

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
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]), strict=True)
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
    scaler = create_scaler()

    start_epoch = 0
    best_loss = float("inf")

    if cli_args.resume:
        print(f"\nResuming from: {cli_args.resume}")
        resume_ckpt = torch.load(cli_args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scaler.load_state_dict(resume_ckpt["scaler"])
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
    evaluate_generations(model, tokenizer, eval_prompts, "SFT 生成评估")
    model.train()

    os.makedirs(save_dir, exist_ok=True)
    if not cli_args.resume:
        global_step = 0

    train_ns = argparse.Namespace(
        epochs=epochs,
        batch_size=batch_size,
        accumulate_grad=accumulate_grad,
        lr=lr,
        warmup_ratio=warmup_ratio,
        stable_ratio=stable_ratio,
        min_lr_ratio=min_lr_ratio,
        total_steps=total_steps,
        clip_grad=clip_grad,
        max_seq_len=max_seq_len,
        lr_scheduler=lr_scheduler,
        # 模型结构字段: 供下游 grpo/ppo/opd 经 extract_checkpoint_config 精确重建
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        use_flash_attn=args.use_flash_attn,
    )

    log_interval = 50
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch}", mininterval=3)

        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with safe_autocast():
                logits, _, _, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            is_accum_step = (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(train_loader)
            # 残差批 (末尾不足 accumulate) 按实际批数除，避免归一化过头导致梯度偏小
            denom = ((batch_idx % accumulate_grad) + 1) if (batch_idx + 1) == len(train_loader) else accumulate_grad
            loss = loss / denom
            scaler.scale(loss).backward()
            if is_accum_step:
                if lr_scheduler == "wsd":
                    lr_mult = get_lr_wsd(global_step, total_steps, warmup_ratio, stable_ratio, min_lr_ratio)
                else:
                    lr_mult = get_lr_cosine(global_step, total_steps, warmup_ratio, min_lr_ratio)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * lr_mult
                optimizer_step(optimizer, scaler, parameters=model.parameters(), clip_grad=clip_grad)
                global_step += 1

            epoch_loss += loss.item() * denom
            n_batches += 1

            if batch_idx % log_interval == 0:
                if lr_scheduler == "wsd":
                    lr_mult = get_lr_wsd(global_step, total_steps, warmup_ratio, stable_ratio, min_lr_ratio)
                else:
                    lr_mult = get_lr_cosine(global_step, total_steps, warmup_ratio, min_lr_ratio)
                cur_lr = lr * lr_mult
                pbar.set_postfix({"loss": f"{loss.item() * denom:.4f}", "lr": f"{cur_lr:.6f}"})

        epoch_loss /= max(n_batches, 1)

        print(f"\n--- SFT Epoch {epoch} 生成评估 ---")
        model.eval()
        evaluate_generations(model, tokenizer, eval_prompts, "SFT 生成评估")
        model.train()

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}: train_loss={epoch_loss:.4f}, lr={cur_lr:.6f}")

        ckpt_name = f"sft_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "train_loss": epoch_loss,
                "args": train_ns,
            },
            os.path.join(save_dir, ckpt_name),
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(save_dir, "sft_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": train_ns,
                },
                best_path,
            )
            print(f"  Saved best SFT model (loss={epoch_loss:.4f}) -> {best_path}")

    print("\n" + "=" * 60)
    print("SFT 训练完成，最终生成评估")
    print("=" * 60)
    model.eval()
    evaluate_generations(model, tokenizer, eval_prompts, "SFT 生成评估")
    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
