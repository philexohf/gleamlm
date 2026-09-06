"""GleamLM 统一 DPO 偏好对齐脚本。通过 --variant 选择配置。

用法:
    python manual/dpo.py --variant nano
    python manual/dpo.py --variant lite --model_path checkpoints/lite/sft/sft_best.pt
"""

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gleamlm.data.dpo_data import DPODataset, dpad_collate
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.base_trainer import (
    create_scaler,
    evaluate_generations,
    optimizer_step,
    set_seed,
)
from gleamlm.trainer.dpo_loss import (
    compute_log_probs,
    dpo_loss,
    get_reference_logps,
)
from gleamlm.trainer.schedulers import get_lr_cosine, get_lr_wsd
from gleamlm.utils.config import (
    DEFAULT_TOKENIZER_PATH,
    extract_checkpoint_config,
    load_config,
)
from gleamlm.utils.torch_utils import clean_state_dict, safe_autocast

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
    parser.add_argument("--epochs", type=int, default=None, help="覆写训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="覆写学习率")
    parser.add_argument("--beta", type=float, default=None, help="覆写 DPO beta")
    parser.add_argument("--batch_size", type=int, default=None, help="覆写 batch size")
    parser.add_argument("--accumulate_grad", type=int, default=None, help="覆写梯度累积步数")
    parser.add_argument("--max_seq_len", type=int, default=None, help="覆写序列长度")
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
    parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "wsd"], default="cosine",
        help="学习率调度器类型"
    )
    parser.add_argument("--stable_ratio", type=float, default=0.80, help="WSD stable 阶段占比")
    parser.add_argument("--min_lr_ratio", type=float, default=None, help="覆写最小学习率比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="覆写权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (对齐 pretrain.py 的 --seed)")

    cli_args = parser.parse_args()

    config_path = os.path.join(cli_args.config_dir, f"{cli_args.variant}.yaml")
    # 单轨 Pydantic 配置: 字段校验/默认值唯一来源 (gleamlm/utils/config.py)
    cfg = load_config(config_path, _ROOT_DIR)

    model_path = cli_args.model_path or os.path.join(cfg.data.checkpoint_dir, "sft", "sft_best.pt")
    data_path = cli_args.data_path or cfg.dpo.data_path
    output_dir = cli_args.output_dir or os.path.join(cfg.data.checkpoint_dir, "dpo")

    lr = cli_args.lr if cli_args.lr is not None else cfg.dpo.lr
    beta = cli_args.beta if cli_args.beta is not None else cfg.dpo.beta
    epochs = cli_args.epochs if cli_args.epochs is not None else cfg.dpo.epochs
    batch_size = cli_args.batch_size if cli_args.batch_size is not None else cfg.dpo.batch_size
    accumulate_grad = (
        cli_args.accumulate_grad
        if cli_args.accumulate_grad is not None
        else cfg.dpo.accumulate_grad
    )
    max_seq_len = cli_args.max_seq_len if cli_args.max_seq_len is not None else cfg.dpo.max_seq_len
    warmup_ratio = cfg.dpo.warmup_ratio
    min_lr_ratio = cli_args.min_lr_ratio if cli_args.min_lr_ratio is not None else cfg.dpo.min_lr_ratio
    weight_decay = (
        cli_args.weight_decay if cli_args.weight_decay is not None else cfg.training.weight_decay
    )
    clip_grad = cfg.training.clip_grad
    lr_scheduler = cli_args.lr_scheduler
    stable_ratio = cli_args.stable_ratio

    set_seed(cli_args.seed)

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

    # 结构快照统一从 _config 读（纯 dict；旧 args 格式已不再兼容）
    sft_cfg = extract_checkpoint_config(sft_ckpt)
    model_kwargs = {
        "vocab_size": sft_cfg["vocab_size"],
        "d_model": sft_cfg["d_model"],
        "num_layers": sft_cfg["num_layers"],
        "num_heads": sft_cfg["num_heads"],
        "num_kv_heads": sft_cfg["num_kv_heads"],
        "d_ff": sft_cfg["d_ff"],
        "dropout": sft_cfg.get("dropout", cfg.model.dropout),
        "max_seq_len": sft_cfg.get("max_seq_len", max_seq_len),
        "pad_token_id": sft_cfg.get("pad_token_id", 0),
    }

    flash_attn = cfg.model.use_flash_attn

    policy_model = GleamLMModel(
        **model_kwargs,
        use_flash_attn=flash_attn,
    ).to(device)
    sft_state = clean_state_dict(
        sft_ckpt["model_state_dict" if "model_state_dict" in sft_ckpt else "model"]
    )
    policy_model.load_state_dict(sft_state)
    print(f"Policy model: {policy_model.get_num_params()[0] / 1e6:.2f}M params")

    ref_model = GleamLMModel(
        **model_kwargs,
        use_flash_attn=flash_attn,
    ).to(device)
    ref_model.load_state_dict(sft_state)
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
        weight_decay=weight_decay,
    )

    total_steps = math.ceil(len(dataloader) / accumulate_grad) * epochs
    scaler = create_scaler()

    print("\n--- DPO 前生成基线 ---")
    eval_prompts = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？",
        "请用一句话描述北京的秋天。",
        "写一首关于春天的五言诗。",
        "请解释一下什么是光合作用。",
    ]
    evaluate_generations(policy_model, tokenizer, eval_prompts, "DPO 生成评估")
    policy_model.train()

    # 模型结构快照: 供下游 (opd/serve) 经 extract_checkpoint_config 精确重建。
    # 字段与 GleamLMModel 构建参数一一对应，纯 dict（weights_only 安全）。
    _ckpt_cfg = {
        "vocab_size": model_kwargs["vocab_size"],
        "d_model": model_kwargs["d_model"],
        "num_layers": model_kwargs["num_layers"],
        "num_heads": model_kwargs["num_heads"],
        "num_kv_heads": model_kwargs["num_kv_heads"],
        "d_ff": model_kwargs["d_ff"],
        "dropout": model_kwargs["dropout"],
        "max_seq_len": model_kwargs["max_seq_len"],
        "pad_token_id": model_kwargs["pad_token_id"],
        "use_flash_attn": flash_attn,
    }

    global_step = 0
    log_interval = 50

    for epoch in range(epochs):
        policy_model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch}", mininterval=3)

        for batch_idx, batch in enumerate(pbar):
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            ref_cho, ref_rej = get_reference_logps(
                ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask
            )

            with safe_autocast():
                c_logits, _, _, _ = policy_model(chosen_ids)
                r_logits, _, _, _ = policy_model(rejected_ids)
                policy_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
                policy_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)
                loss = dpo_loss(policy_cho, policy_rej, ref_cho, ref_rej, beta)

            is_accum_step = (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(dataloader)
            # 残差批 (末尾不足 accumulate) 按实际批数除，避免归一化过头导致梯度偏小
            denom = ((batch_idx % accumulate_grad) + 1) if (batch_idx + 1) == len(dataloader) else accumulate_grad
            loss = loss / denom
            scaler.scale(loss).backward()
            if is_accum_step:
                if lr_scheduler == "wsd":
                    lr_mult = get_lr_wsd(global_step, total_steps, warmup_ratio, stable_ratio, min_lr_ratio)
                else:
                    lr_mult = get_lr_cosine(global_step, total_steps, warmup_ratio, min_lr_ratio)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * lr_mult
                optimizer_step(optimizer, scaler, parameters=policy_model.parameters(), clip_grad=clip_grad)
                global_step += 1

            epoch_loss += loss.item() * denom
            n_batches += 1

            if batch_idx % log_interval == 0:
                if lr_scheduler == "wsd":
                    lr_mult = get_lr_wsd(global_step, total_steps, warmup_ratio, stable_ratio, min_lr_ratio)
                else:
                    lr_mult = get_lr_cosine(global_step, total_steps, warmup_ratio, min_lr_ratio)
                cur_lr = lr * lr_mult
                pbar.set_postfix({"loss": f"{loss.item() * denom:.4f}", "lr": f"{cur_lr:.2e}"})

        epoch_loss /= max(n_batches, 1)
        avg_loss = epoch_loss

        print(f"\n--- DPO Epoch {epoch} 生成评估 ---")
        evaluate_generations(policy_model, tokenizer, eval_prompts, "DPO 生成评估")
        policy_model.train()

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}: dpo_loss={avg_loss:.4f}, lr={cur_lr:.2e}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "dpo_best.pt")
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),
            "dpo_loss": avg_loss,
            "_config": _ckpt_cfg,
        },
        save_path,
    )
    print(f"Model saved: {save_path}")

    print("\n" + "=" * 60)
    print("DPO 训练完成，最终生成评估")
    print("=" * 60)
    evaluate_generations(policy_model, tokenizer, eval_prompts, "DPO 生成评估")

    print(f"\nFinal dpo_loss: {avg_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
