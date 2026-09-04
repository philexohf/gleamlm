
"""全量 SFT 微调脚本 — TRL SFTTrainer（无 LoRA）。

对比:
  - industrial/sft_lora.py: LoRA 部分参数微调（快、省显存）
  - industrial/sft.py:       全量参数微调（效果更好、可直接进 DPO 链）
  - manual/sft.py:           手写 GleamLMModel + CrossEntropyLoss

用法:
  python industrial/sft.py \
    --model_path checkpoints/lite/best_model.pt \
    --data_path data/sft_data.jsonl \
    --output_dir checkpoints/lite/sft_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

管道位置:
  manual/pretrain.py (.pt) → industrial/sft.py (HF dir) → industrial/dpo.py (HF dir)

数据格式 (JSONL，推荐 messages，TRL 用 tokenizer chat template 自动渲染):
  {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
  {"prompt": [{"role": "user", "content": "你好"}], "completion": [{"role": "assistant", "content": "你好！"}]}
兼容旧格式（纯文本续写，自动 packing）:
  {"text": "<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n你好！<|im_end|>\\n"}
messages 格式自动开 completion_only_loss（只算 assistant 回答，对齐 TRL 工业默认）
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from gleamlm.utils.config import extract_checkpoint_config
from hf.hf_config import GleamLMConfig, gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full SFT for GleamLM (TRL)")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, default=None, help="Directory with config.json")
    parser.add_argument("--data_path", type=str, required=True, help="SFT data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./sft_out")
    parser.add_argument("--lr", type=float, default=5e-6, help="全量 SFT 用小 lr（vs LoRA 的 5e-5）")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Directory with tokenizer.json (HF format)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 支持两种输入: HF 目录（转换产物 / from_pretrained）或 core 轨 .pt
    if os.path.isdir(args.model_path):
        model = GleamLMForCausalLM.from_pretrained(args.model_path)
    else:
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if args.config_path:
            config = GleamLMConfig.from_pretrained(args.config_path)
        else:
            config = gleamlm_config_from_core(extract_checkpoint_config(ckpt))

        model = GleamLMForCausalLM(config)
        missing, unexpected = load_from_checkpoint(model, ckpt)
        if missing or unexpected:
            print(f"[warn] sft load — missing={len(missing)} unexpected={len(unexpected)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Full SFT — model: {total / 1e6:.2f}M params, LR: {args.lr:.1e}")

    raw_data = load_jsonl(args.data_path)
    dataset = Dataset.from_list(raw_data)

    if not args.tokenizer_path:
        # HF 目录输入自带 tokenizer.json 时默认复用（与 dpo/from_pretrained 链路一致）
        if os.path.isdir(args.model_path) and os.path.exists(
            os.path.join(args.model_path, "tokenizer.json")
        ):
            args.tokenizer_path = args.model_path
    if not args.tokenizer_path:
        raise ValueError(
            "SFT needs an HF-format tokenizer (tokenizer.json). "
            "Provide --tokenizer_path or convert BBPE tokenizer first."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 数据格式: 推荐 messages（role/content 数组，TRL 用 tokenizer 的 chat
    # template 自动渲染）；兼容旧 {"text": 预渲染ChatML} 格式。
    # messages 时开 completion_only_loss（只算 assistant 回答，prompt 置 -100）。
    is_conversational = "messages" in (dataset.column_names or []) or (
        "prompt" in (dataset.column_names or []) and "completion" in (dataset.column_names or [])
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        max_length=args.max_seq_length,
        packing=False,
        loss_type="nll",
    )
    if not is_conversational:
        training_args.dataset_text_field = "text"
        training_args.packing = True  # 纯文本（预训练式续写）才 packing
    else:
        # messages/prompt-completion: 只算 completion（回答）的 loss
        training_args.completion_only_loss = True

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Full SFT model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
