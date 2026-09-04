
"""LoRA + SFT 微调脚本 — PEFT LoRA 注入 + TRL SFTTrainer。

用法:
  # 0.6B: Megatron 预训练 checkpoint → HF 格式 → SFT
  python industrial/sft_lora.py \
    --model_path checkpoints/0.6b/pretrain_final.pt \
    --data_path data/0.6b/sft_data.jsonl \
    --output_dir checkpoints/0.6b/sft_lora_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

数据格式 (JSONL，推荐 messages):
  {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
兼容旧格式（纯文本续写，自动 packing）:
  {"text": "<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n你好！<|im_end|>\\n"}

注: tokenizer_path 需指向 HF 格式 tokenizer（BBPE 用 export_to_hf_format 导出）
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
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
    parser = argparse.ArgumentParser(description="LoRA SFT for GleamLM")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, default=None, help="Directory with config.json")
    parser.add_argument("--data_path", type=str, required=True, help="SFT data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./sft_lora_out")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Directory with tokenizer.json (HF format)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if args.config_path:
        config = GleamLMConfig.from_pretrained(args.config_path)
    else:
        config = gleamlm_config_from_core(extract_checkpoint_config(ckpt))

    model = GleamLMForCausalLM(config)
    missing, unexpected = load_from_checkpoint(model, ckpt)
    if missing or unexpected:
        print(f"[warn] sft_lora load — missing={len(missing)} unexpected={len(unexpected)}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_data = load_jsonl(args.data_path)
    dataset = Dataset.from_list(raw_data)

    if not args.tokenizer_path:
        raise ValueError(
            "SFT needs an HF-format tokenizer (tokenizer.json). "
            "Provide --tokenizer_path or convert BBPE tokenizer first."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 数据格式: messages（role/content）→ completion_only_loss；旧 {"text"} 纯文本 → packing
    is_conversational = "messages" in (dataset.column_names or []) or (
        "prompt" in (dataset.column_names or []) and "completion" in (dataset.column_names or [])
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
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
        training_args.packing = True
    else:
        training_args.completion_only_loss = True

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
