"""
PPO / RLOO — TRL 工业版 RLHF 对齐。

PPO 的五个核心概念（TRL 1.x 中 PPO 已被 RLOO 替代，原理不变）：
  1. Policy clip：r(θ) = π_θ/π_old 限制在 [1-ε, 1+ε]，防止一步崩坏
  2. Advantage Estimation：RLOO 用 leave-one-out baseline，比 PPO 的 GAE 更简单
  3. KL penalty：防止 π_θ 偏离 π_ref 太远（β 系数控制）
  4. Mini-batch updates：一个 rollout batch 分多个 mini-batch 更新
  5. Reward function：工业上 reward 来自 Reward Model

TRL 1.x 变化:
  - TRL 0.x 时代：PPOTrainer + ValueHead → 标准 PPO
  - TRL 1.0+：PPO 被移除，推荐 RLOO 和 GRPO（DeepSeek-R1 同款）
  - RLOO vs PPO：去掉 Value Network，用 leave-one-out baseline → 更简单、更稳定
  - TRL 1.x 不再支持 PPOTrainer：工业界 RLHF 演进方向是 GRPO 和 RLOO，
    两者都不需要 Value Network，训练更稳定

对比手动版 (manual/ppo.py)：
  - 手动版：手写 ValueHead、GAE、clip loss、entropy bonus、old policy sync
  - 工业版：RLOOConfig + RLOOTrainer 一行 → 聚焦 reward 设计
用法:
  # 0.6B: SFT/DPO 产物 → RLOO 强化对齐
  python industrial/ppo.py \
    --model_path checkpoints/0.6b/sft_lora_hf \
    --data_path data/0.6b/rlhf.jsonl \
    --output_dir checkpoints/0.6b/ppo_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

  # 多卡
  accelerate launch industrial/ppo.py \
    --model_path checkpoints/0.6b/sft_lora_hf \
    --data_path data/0.6b/rlhf.jsonl \
    --output_dir checkpoints/0.6b/ppo_hf \
    --tokenizer_path checkpoints/bbpe_24k/hf_export

数据格式 (JSONL，推荐带 ground_truth 做规则 reward):
  {"prompt": "请解释质能方程"}
  {"prompt": "2+2=?", "ground_truth": "4"}
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

from trl import RLOOConfig, RLOOTrainer


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

    RLOOTrainer 要求 reward_funcs 至少一个可调用函数；生产环境应替换为
    Reward Model 或规则 reward（如格式/答案正确性检查）。
    """
    ground_truth = kwargs.get("ground_truth")
    if ground_truth is not None:
        rewards = []
        for c, gt in zip(completions, ground_truth):
            if not c:
                rewards.append(-1.0)
            elif gt and str(gt).strip() in c:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    return [1.0 if len(c) > 0 else -1.0 for c in completions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO/RLOO RLHF for GleamLM (TRL 1.x)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="GleamLM checkpoint (.pt) or HF model dir")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Directory with config.json (if non-.pt model)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="RLHF queries (JSONL: {prompt/instruction})")
    parser.add_argument("--output_dir", type=str, default="./ppo_out")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="HF-format tokenizer dir (tokenizer.json)")
    # RLHF 核心超参
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="RLOO learning rate (通常 1e-6)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每个 prompt 采样数（RLOO 的 group_size）")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty 系数 (PPO/RLOO: 推荐 0.01-0.1)")
    parser.add_argument("--max_prompt_length", type=int, default=256)
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
            print(f"[warn] ppo load — missing={len(missing)} unexpected={len(unexpected)}")
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

    # RLOO 无 ValueHead（leave-one-out baseline），无需调 GAE λ，
    # 参数只有 num_generations + beta
    rloo_config = RLOOConfig(
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

    # RLOO 内部: generate → reward → leave-one-out baseline → clip update
    # （去掉了 ValueHead 和 GAE）
    trainer = RLOOTrainer(
        model=model,
        args=rloo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[default_reward],
    )

    # ── 训练 ──
    trainer.train()

    # ── 保存 ──
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"PPO/RLOO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
