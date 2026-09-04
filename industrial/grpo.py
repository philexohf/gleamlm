"""
GRPO (Group Relative Policy Optimization) — TRL 工业版 RLHF。

GRPO 核心设计:
  - 无需 Value Network：用 group 内奖励均值做 baseline，比 PPO 简单
  - 优势 = (r_i - mean(r_group)) / std(r_group)：归一化消去绝对奖励尺度
  - KL penalty：防止 π_θ 偏离 π_ref 太远
  - group_size 越大方差越小（显存允许的情况下 4-8）

对比手动版 (manual/grpo.py)：
  - 手动版：手写 group 采样/归一化/rollout/vLLM 后端
  - 工业版：GRPOConfig + GRPOTrainer 一行 → 聚焦 reward 设计

用法:
  # 0.6B: SFT/DPO 产物 → GRPO 强化对齐
  python industrial/grpo.py \
    --model_path checkpoints/0.6b/sft_lora_hf \
    --data_path data/0.6b/rlhf.jsonl \
    --output_dir checkpoints/0.6b/grpo_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

  # 多卡
  accelerate launch industrial/grpo.py \
    --model_path checkpoints/0.6b/sft_lora_hf \
    --data_path data/0.6b/rlhf.jsonl \
    --output_dir checkpoints/0.6b/grpo_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

数据格式 (JSONL，推荐带 ground_truth 做规则 reward):
  {"prompt": "请解释质能方程"}
  {"prompt": "2+2=?", "ground_truth": "4"}
  {"prompt": [{"role": "user", "content": "2+2=?"}], "ground_truth": "4"}

注意:
  GRPOTrainer 自动为每个 prompt 采样 num_generations 个 response，
  用 reward function 打分后组内归一化计算优势，然后 PPO-clip 更新。
  数据含 ground_truth 列时 default_reward 做答案匹配（工业规则 reward）。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from hf.hf_config import GleamLMConfig, gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint
from gleamlm.utils.config import extract_checkpoint_config

from trl import GRPOConfig, GRPOTrainer


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def default_reward(prompts, completions, **kwargs):
    """规则 reward：数据含 ground_truth 列时做答案匹配，否则长度兜底。

    TRL GRPOTrainer 的 reward_funcs 签名: 接收 prompts/completions/completion_ids
    及数据集中除 prompt 外的所有列（如 ground_truth）。返回 list[float]。
    工业上（DeepSeek/OpenR1）用规则 reward（答案正确性），生产可替换为
    Reward Model 或更细的格式/重复检查。
    """
    ground_truth = kwargs.get("ground_truth")
    if ground_truth is not None:
        # 规则匹配: 精确命中 +1，否则 0；空回答 -1 惩罚
        rewards = []
        for c, gt in zip(completions, ground_truth):
            if not c:
                rewards.append(-1.0)
            elif gt and str(gt).strip() in c:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    # 无 ground_truth: 长度兜底（短回答 +1，空回答 -1）
    return [1.0 if len(c) > 0 else -1.0 for c in completions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO RLHF for GleamLM (TRL)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="GleamLM checkpoint (.pt) or HF model dir")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Directory with config.json (if non-.pt model)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="GRPO queries (JSONL: {prompt/instruction})")
    parser.add_argument("--output_dir", type=str, default="./grpo_out")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="HF-format tokenizer dir (tokenizer.json)")
    # GRPO 核心超参
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="GRPO learning rate (通常 1e-6)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每个 device 的 prompt batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每个 prompt 采样的 response 数量 (group_size)")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty 系数 (GRPO: 推荐 0.01-0.1)")
    parser.add_argument("--max_prompt_length", type=int, default=256,
                        help="Prompt 截断长度（GRPOConfig 无此参数，由 tokenizer truncation 控制）")
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 模型加载 ──
    if args.model_path.endswith(".pt"):
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if args.config_path:
            config = GleamLMConfig.from_pretrained(args.config_path)
        else:
            config = gleamlm_config_from_core(extract_checkpoint_config(ckpt))
        model = GleamLMForCausalLM(config)
        missing, unexpected = load_from_checkpoint(model, ckpt)
        if missing or unexpected:
            print(f"[warn] grpo load — missing={len(missing)} unexpected={len(unexpected)}")
        # TRL 需要从 config._name_or_path 重建 ref 模型；.pt 构造的模型没有该字段，
        # 导出为本地 HF 目录作为 ref 来源
        model.save_pretrained(args.output_dir)
        model.config._name_or_path = args.output_dir
    else:
        model = GleamLMForCausalLM.from_pretrained(args.model_path)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 数据集 ──
    raw_data = load_jsonl(args.data_path)
    for item in raw_data:
        if "prompt" not in item and "instruction" in item:
            item["prompt"] = item.pop("instruction")
    dataset = Dataset.from_list(raw_data)

    # GRPO 无需 reward model / value function / GAE λ，核心只有 group_size + beta
    grpo_config = GRPOConfig(
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        bf16=torch.cuda.is_available(),
        fp16=False,
        output_dir=args.output_dir,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # GRPOTrainer 内部: generate × num_generations → reward → group 内归一化
    # 优势 (advantage = (reward - mean) / std) → PPO-clip 更新 → KL penalty
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[default_reward],
    )

    # ── 训练 ──
    trainer.train()

    # ── 保存 ──
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"GRPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
