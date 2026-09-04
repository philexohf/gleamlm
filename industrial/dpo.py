
"""DPO (Direct Preference Optimization) 对齐脚本 — TRL DPOTrainer。

用法:
  # 0.6B: SFT 产物 → DPO 对齐
  python industrial/dpo.py \
    --model_path checkpoints/0.6b/sft_lora_hf \
    --data_path data/0.6b/dpo_data.jsonl \
    --output_dir checkpoints/0.6b/dpo_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

数据格式 (JSONL，推荐 role/content 数组，TRL 用 tokenizer chat template 渲染):
  {"prompt": [{"role": "user", "content": "你好"}],
   "chosen": [{"role": "assistant", "content": "你好！有什么可以帮你的？"}],
   "rejected": [{"role": "assistant", "content": "你好。"}]}
兼容纯文本（chosen/rejected 只含回答部分）:
  {"prompt": "<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n",
   "chosen": "你好！有什么可以帮你的？<|im_end|>",
   "rejected": "你好。<|im_end|>"}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

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
    parser = argparse.ArgumentParser(description="DPO for GleamLM")
    parser.add_argument("--model_path", type=str, required=True, help="Base model (HF format or .pt)")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--config_path", type=str, default=None, help="Directory with config.json")
    parser.add_argument("--data_path", type=str, required=True, help="DPO data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./dpo_out")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Directory with tokenizer.json (HF format)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_path.endswith(".pt"):
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if args.config_path:
            config = GleamLMConfig.from_pretrained(args.config_path)
        else:
            config = gleamlm_config_from_core(extract_checkpoint_config(ckpt))
        model = GleamLMForCausalLM(config)
        missing, unexpected = load_from_checkpoint(model, ckpt)
        if missing or unexpected:
            print(f"[warn] dpo load — missing={len(missing)} unexpected={len(unexpected)}")
    else:
        model = GleamLMForCausalLM.from_pretrained(args.model_path)

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)

    # trl 的 ref_model=None 需要从 model.config._name_or_path 重建 ref 模型；
    # .pt 直接构造的模型没有 _name_or_path 且 gleam_lm 未注册 AutoModel，
    # 因此导出为本地 HF 目录作为 ref 来源（与 DPO 消费 SFT 产物的设计一致）
    if not getattr(model.config, "_name_or_path", None):
        model.save_pretrained(args.output_dir)
        model.config._name_or_path = args.output_dir

    model.enable_input_require_grads()

    raw_data = load_jsonl(args.data_path)
    dataset = Dataset.from_list(raw_data)

    if not args.tokenizer_path:
        raise ValueError(
            "DPO needs an HF-format tokenizer (tokenizer.json). "
            "Provide --tokenizer_path or convert BBPE tokenizer first."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = DPOConfig(
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
        beta=args.beta,
        max_length=args.max_seq_length,
        # DPO 用 LoRA + 两次前向（model/ref），core 的 gradient checkpoint
        # 与 PEFT 组合在重算时 metadata 校验失败；DPO 数据量小无需省显存
        gradient_checkpointing=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
