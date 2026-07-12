"""GleamLM-Pro 126M DPO 偏好对齐脚本。基于 sft_best.pt 策略模型 + 冻结参考模型。

用法：
    python dpo.py --data_path ./data/dpo_data.jsonl \
                  --model_path ./checkpoints/sft/sft_best.pt
"""
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.training.base_trainer import create_scaler, set_seed
from gleamlm.training.dpo_trainer import (
    DPODataset,
    dpad_collate,
    evaluate_dpo,
    train_one_epoch_dpo,
)
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH
from gleamlm.utils.torch_utils import get_lr_cosine

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")


# 主函数
def main():
    parser = argparse.ArgumentParser(description="GleamLM-Pro 126M DPO")
    parser.add_argument("--data_path", default="./gleamlm-pro/data/dpo_data.jsonl")
    parser.add_argument("--model_path",
                        default=os.path.join(_CHECKPOINT_DIR, "sft", "sft_best.pt"))
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)

    # Pro 126M 模型架构
    parser.add_argument("--vocab_size", type=int, default=12002)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_kv_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--use_flash_attn", action="store_true", default=True,
                        help="Use PyTorch Flash Attention (sdpa)")
    parser.add_argument("--no_flash_attn", dest="use_flash_attn", action="store_false",
                        help="Disable Flash Attention")
    parser.set_defaults(use_flash_attn=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad", type=int, default=2)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir",
                        default=os.path.join(_CHECKPOINT_DIR, "dpo"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print("=" * 60)
    print("GleamLM-Pro 126M DPO 偏好对齐")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"LR: {args.lr:.1e}, Beta: {args.beta}, Epochs: {args.epochs}")

    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    sft_ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

    # 从 checkpoint 中读取模型架构参数，避免默认值与实际训练配置不一致
    if "args" in sft_ckpt:
        sft_args = sft_ckpt["args"]
        model_kwargs = {
            "vocab_size": getattr(sft_args, "vocab_size", args.vocab_size),
            "d_model": getattr(sft_args, "d_model", args.d_model),
            "num_layers": getattr(sft_args, "num_layers", args.num_layers),
            "num_heads": getattr(sft_args, "num_heads", args.num_heads),
            "num_kv_heads": getattr(sft_args, "num_kv_heads", args.num_kv_heads),
            "d_ff": getattr(sft_args, "d_ff", args.d_ff),
            "dropout": getattr(sft_args, "dropout", args.dropout),
            "max_seq_len": getattr(sft_args, "max_seq_len", args.max_seq_len),
            "pad_token_id": getattr(sft_args, "pad_token_id", 0),
        }
    else:
        model_kwargs = {
            "vocab_size": args.vocab_size, "d_model": args.d_model,
            "num_layers": args.num_layers, "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads, "d_ff": args.d_ff,
            "dropout": args.dropout, "max_seq_len": args.max_seq_len,
        }

    policy_model = GleamLMModel(**model_kwargs, use_flash_attn=args.use_flash_attn).to(device)
    policy_model.load_state_dict(sft_ckpt["model_state_dict" if "model_state_dict" in sft_ckpt else "model"])
    print(f"Policy model: {policy_model.get_num_params()[0]/1e6:.2f}M params")

    ref_model = GleamLMModel(**model_kwargs, use_flash_attn=args.use_flash_attn).to(device)
    ref_model.load_state_dict(sft_ckpt["model_state_dict" if "model_state_dict" in sft_ckpt else "model"])
    for p in ref_model.parameters():
        p.requires_grad = False
    print("Reference model: frozen")

    dataset = DPODataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    print(f"DPO pairs: {len(dataset)}")

    effective_batch = args.batch_size * args.accumulate_grad
    print(f"Batch: {args.batch_size} x {args.accumulate_grad} = {effective_batch}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=dpad_collate, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.95), eps=1e-8,
                                   weight_decay=0.01)

    total_steps = math.ceil(len(dataloader) / args.accumulate_grad) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(step, total_steps, warmup_ratio=0.01, min_lr_ratio=0.05),
    )
    scaler = create_scaler()

    # DPO 前基线
    print("\n--- DPO 前生成基线 ---")
    evaluate_dpo(policy_model, tokenizer)
    policy_model.train()

    avg_loss = float("inf")
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch_dpo(
            policy_model, ref_model, dataloader, optimizer, scheduler,
            scaler, args.beta, device, args)
        print(f"\nDPO Epoch {epoch}: loss={avg_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "dpo_best.pt")
    torch.save({
        "model_state_dict": policy_model.state_dict(),
        "dpo_loss": avg_loss,
        "args": args,
    }, save_path)
    print(f"Model saved: {save_path}")

    # DPO 后评估
    print("\n--- DPO 后最终评估 ---")
    evaluate_dpo(policy_model, tokenizer)


if __name__ == "__main__":
    main()
