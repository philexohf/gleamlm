"""GleamLM 统一 DPO 偏好对齐脚本。通过 --variant 选择配置。

用法:
    python scripts/dpo.py --variant nano
    python scripts/dpo.py --variant lite --model_path checkpoints/lite/sft/sft_best.pt
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
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, load_config_as_args
from gleamlm.utils.torch_utils import get_lr_cosine

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="GleamLM DPO 偏好对齐")
    parser.add_argument(
        "--variant", type=str, choices=["nano", "lite", "pro"], required=True, help="模型变体"
    )
    parser.add_argument(
        "--config_dir", type=str, default=os.path.join(_ROOT_DIR, "configs"), help="YAML 配置目录"
    )
    parser.add_argument("--data_path", type=str, default=None, help="覆写 DPO 数据路径")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="SFT 模型路径 (默认: checkpoints/{variant}/sft/sft_best.pt)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH, help="BBPE 分词器目录"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="DPO 模型保存目录")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 续训")

    cli_args = parser.parse_args()

    config_path = os.path.join(cli_args.config_dir, f"{cli_args.variant}.yaml")
    args = load_config_as_args(config_path, model_name=cli_args.variant, cli_overrides=True)

    # 从配置读取所有参数（dpo 块已加入 NO_PREFIX，自动展平到 namespace）
    model_path = cli_args.model_path or os.path.join(args.checkpoint_dir, "sft", "sft_best.pt")
    data_path = cli_args.data_path or getattr(args, "data_path", "")
    output_dir = cli_args.output_dir or os.path.join(args.checkpoint_dir, "dpo")

    lr = args.dpo_lr
    beta = args.dpo_beta
    epochs = args.dpo_epochs
    batch_size = args.dpo_batch_size
    accumulate_grad = args.dpo_accumulate_grad
    max_seq_len = args.dpo_max_seq_len
    warmup_ratio = getattr(args, "dpo_warmup_ratio", 0.02)
    min_lr_ratio = getattr(args, "dpo_min_lr_ratio", 0.05)
    clip_grad = getattr(args, "clip_grad", 1.0)

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant_name = cli_args.variant.upper()
    print("=" * 60)
    print(f"GleamLM-{variant_name} DPO 偏好对齐")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"Model: {model_path}")
    print(f"LR: {lr:.1e}, Beta: {beta}, Epochs: {epochs}, Batch: {batch_size}")

    tokenizer = BBPETokenizer.load(cli_args.tokenizer_path)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    sft_ckpt = torch.load(model_path, map_location=device, weights_only=False)

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
            "max_seq_len": getattr(sft_args, "max_seq_len", max_seq_len),
            "pad_token_id": getattr(sft_args, "pad_token_id", 0),
        }
    else:
        model_kwargs = {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "max_seq_len": max_seq_len,
            "pad_token_id": 0,
            "use_qk_norm": getattr(args, "use_qk_norm", True),
        }

    flash_attn = getattr(args, "use_flash_attn", False)

    policy_model = GleamLMModel(
        **model_kwargs,
        use_flash_attn=flash_attn,
    ).to(device)
    policy_model.load_state_dict(
        sft_ckpt["model_state_dict" if "model_state_dict" in sft_ckpt else "model"]
    )
    print(f"Policy model: {policy_model.get_num_params()[0] / 1e6:.2f}M params")

    ref_model = GleamLMModel(
        **model_kwargs,
        use_flash_attn=flash_attn,
    ).to(device)
    ref_model.load_state_dict(
        sft_ckpt["model_state_dict" if "model_state_dict" in sft_ckpt else "model"]
    )
    for p in ref_model.parameters():
        p.requires_grad = False
    print("Reference model: frozen")

    dataset = DPODataset(data_path, tokenizer, max_seq_len=max_seq_len)
    print(f"DPO pairs: {len(dataset)}")

    effective_batch = batch_size * accumulate_grad
    print(f"Batch: {batch_size} x {accumulate_grad} = {effective_batch}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dpad_collate,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    total_steps = math.ceil(len(dataloader) / accumulate_grad) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(step, total_steps, warmup_ratio, min_lr_ratio=min_lr_ratio),
    )
    scaler = create_scaler()

    print("\n--- DPO 前生成基线 ---")
    evaluate_dpo(policy_model, tokenizer)
    policy_model.train()

    dpo_ns = argparse.Namespace(
        batch_size=batch_size,
        accumulate_grad=accumulate_grad,
        clip_grad=clip_grad,
        lr=lr,
        epochs=epochs,
        max_seq_len=max_seq_len,
    )

    avg_loss = float("inf")
    for epoch in range(epochs):
        avg_loss = train_one_epoch_dpo(
            policy_model,
            ref_model,
            dataloader,
            optimizer,
            scheduler,
            scaler,
            beta,
            device,
            dpo_ns,
        )
        print(f"\nDPO Epoch {epoch}: loss={avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "dpo_best.pt")
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),
            "dpo_loss": avg_loss,
            "args": dpo_ns,
        },
        save_path,
    )
    print(f"Model saved: {save_path}")

    print("\n--- DPO 后最终评估 ---")
    evaluate_dpo(policy_model, tokenizer)


if __name__ == "__main__":
    main()
