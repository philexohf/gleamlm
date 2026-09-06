"""
GRPO (Group Relative Policy Optimization) — DeepSeek 风格的 RLHF 对齐。

核心公式:
  loss = -E[log π_θ(y|x) * A] + β * KL(π_θ || π_ref)
  其中 A = (r_i - mean(r_group)) / std(r_group)

GRPO vs PPO:
  PPO:     value network + clip + GAE + entropy — 3 个 loss 项
  GRPO:    无 value network，优势 = group 内归一化奖励 — 1 个 loss 项
          更简单、更稳定、收敛更快

用法:"""

# RLHF 流水线: SFT → RM → RL。PPO 需 4 个模型 (policy+ref+reward+value)，
# value network 与 policy 同尺寸显存翻倍；GRPO 砍掉 value network，
# 用 group 内奖励统计量做 MC baseline: A_i = (r_i - mean(r_group)) / std(r_group)。
# 代价是推理预算增加 group_size 倍 (DeepSeek-R1 用 group_size=64)。

import argparse
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gleamlm.data.rl_data import RLHFDataset, tokenize_prompts
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.rl_trainer import compute_reward, grpo_loss
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from gleamlm.utils.torch_utils import clean_state_dict, safe_autocast


def train(args):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # 当前未实现 DDP (无 init_process_group / 梯度 all-reduce)，多卡直接禁用，
    # 避免各 rank 用不同数据分片静默训练出分歧副本。
    if world_size > 1:
        raise SystemExit("GRPO 未实现 DDP，请单进程运行 (不要用 torchrun)")
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)

    dataset = RLHFDataset(args.data, max_seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
    )

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = extract_checkpoint_config(ckpt)

    policy_model = GleamLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0),
        max_seq_len=args.seq_len,
        pad_token_id=tokenizer.pad_id,
        use_flash_attn=cfg.get("use_flash_attn", False),
    ).to(device)
    policy_model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=False)
    policy_model.train()

    # π_ref 必须是训练开始时的快照: 用自己当前参数算 KL 等于自己约束自己；
    # ref 只推理 (autocast + no_grad)，显存开销约 1 份模型。
    ref_model = deepcopy(policy_model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if rank == 0:
        total = sum(p.numel() for p in policy_model.parameters())
        print(f"GRPO — model: {total / 1e6:.2f}M, group_size={args.group_size}, beta={args.beta}")
        os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for _ in range(args.epochs):
        for batch_items in loader:
            # batch_items: list[{"prompt", "ground_truth"}]（RLHFDataset 返回 dict）
            batch_prompts = [it["prompt"] for it in batch_items]
            batch_ground_truth = [it.get("ground_truth") for it in batch_items]
            # Rollout: 每个 prompt 采样 group_size 个回答 (grouped sampling)；
            # GRPO 比 SFT 慢 group_size 倍 (每次迭代都先推理)。
            prompt_ids = tokenize_prompts(batch_prompts, tokenizer, args.seq_len).to(device)
            prompt_len = prompt_ids.size(1)

            gen_seqs = []
            with torch.no_grad():
                # rollout 必须 eval 模式: dropout 会往采样解码注入噪声，
                # 且会让 ref 的 log-prob 分布与推理时不一致
                policy_model.eval()
                for _ in range(args.group_size):
                    ids = prompt_ids.clone()
                    for _ in range(args.max_new_tokens):
                        logits, _, _, _ = policy_model(ids)
                        if args.temperature <= 0:
                            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        else:
                            # 组内多样性来源: 采样而非贪心——greedy 会让 group_size
                            # 个轨迹完全相同 → 组内 std=0 → 优势全 0，只剩 KL 在学
                            probs = F.softmax(logits[:, -1, :] / args.temperature, dim=-1)
                            nxt = torch.multinomial(probs, 1)
                        ids = torch.cat([ids, nxt], dim=-1)
                        if (nxt == tokenizer.eos_id).all():
                            break
                        if ids.size(1) >= args.seq_len:
                            break
                    gen_seqs.append(ids)
                policy_model.train()

            # 优势必须按 prompt 分组归一化: 不同 prompt 的奖励分布不同，
            # 跨 prompt 归一化会引入噪声；loss 按 group_size 平均累加。
            rewards = torch.zeros(len(batch_prompts), args.group_size, device=device)
            for g_idx in range(args.group_size):
                gen_ids = gen_seqs[g_idx]
                # reward 只针对回答部分打分 (prompt 是固定条件，不计入)。
                resp_ids = gen_ids[:, prompt_len:]
                decoded = [
                    tokenizer.decode(resp_ids[i].tolist(), skip_special=True)
                    for i in range(len(batch_prompts))
                ]
                rewards[:, g_idx] = torch.tensor(
                    [
                        compute_reward(d, gt)
                        for d, gt in zip(decoded, batch_ground_truth, strict=True)
                    ],
                    device=device,
                    dtype=torch.float,
                )
            adv = (rewards - rewards.mean(dim=-1, keepdim=True)) / (
                rewards.std(dim=-1, keepdim=True) + 1e-8
            )

            all_losses = []
            for g_idx in range(args.group_size):
                gen_ids = gen_seqs[g_idx]

                with safe_autocast():
                    # loss 前向与 rollout 同一分布: eval 模式算 log-prob，
                    # 否则 dropout 会让重要性比在随机函数上计算
                    # (当前 config dropout=0.0，此改动为理论一致性)
                    policy_model.eval()
                    p_logits, _, _, _ = policy_model(gen_ids)
                    policy_model.train()
                    with torch.no_grad():
                        r_logits, _, _, _ = ref_model(gen_ids)

                loss = grpo_loss(
                    p_logits,
                    r_logits,
                    gen_ids,
                    prompt_len,
                    adv[:, g_idx],
                    beta=args.beta,
                )
                all_losses.append(loss / args.group_size)

            total_loss = torch.stack(all_losses).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0 and global_step % args.log_interval == 0:
                print(f"step {global_step}  loss={total_loss.item():.4f}")
            global_step += 1

    if rank == 0:
        torch.save(
            {
                "model_state_dict": policy_model.state_dict(),
                "_config": extract_checkpoint_config(ckpt),
                "step": global_step,
            },
            os.path.join(args.output_dir, "grpo_final.pt"),
        )


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM GRPO alignment")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./checkpoints/grpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="rollout 采样温度 (0 = 贪心, 组内轨迹将完全相同)",
    )
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer_path", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
