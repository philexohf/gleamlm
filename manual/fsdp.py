"""
FSDP 分布式训练示例 — ZeRO-2 / ZeRO-3 分片策略

单机 8 卡 (0.6B):
  torchrun --nproc_per_node=8 manual/fsdp.py --data ./data --variant 0.6b

双机 16 卡 (1B+):
  torchrun --nnodes=2 --nproc_per_node=8 manual/fsdp.py --data ./data --variant 1b
"""

# 选型按单卡显存容量: DDP (每卡全量，仅 all-reduce 梯度，通信最小) →
# FSDP ZeRO-2 (分片优化器状态，省 ~12B/参数) → ZeRO-3 (参数+梯度+优化器全分片，
# 每层 forward/backward 各一次 all-gather + reduce-scatter，通信约 1.5×) →
# TP (层内矩阵按列/行切分，需 NVLink) / PP (按层分段，有气泡)；
# 工业界标准是 TP+PP+DP 三维混合 (Megatron-LM)。

import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gleamlm.data.dataset import tokenize_and_group
from gleamlm.models.model import GleamLMModel, DecoderLayer
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, ModelConfig
from gleamlm.trainer.schedulers import get_lr_cosine


def train(args, model_cfg: ModelConfig):
    dist.init_process_group(backend="nccl")
    local_rank = int(dist.get_rank())
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        print(f"World size: {dist.get_world_size()}, FSDP stage: ZeRO-{3 if args.full_shard else 2}")

    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)
    model = GleamLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=0.0,
        pad_token_id=tokenizer.pad_id,
        tie_weights=model_cfg.tie_weights,
        use_flash_attn=model_cfg.use_flash_attn,
        use_gradient_checkpointing=model_cfg.use_gradient_checkpointing,
        rope_scale=model_cfg.rope_scale,
        rope_factor=model_cfg.rope_factor,
        rope_theta=model_cfg.rope_theta,
    ).to(device)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DecoderLayer},
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD if args.full_shard else ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
    )

    dataset = tokenize_and_group(args.data, tokenizer, model_cfg.max_seq_len)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(loader)
    step = 0
    model.train()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)

            lr_mult = get_lr_cosine(step, total_steps, args.warmup_ratio, 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_mult

            logits, _, aux_loss, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss + 0.01 * aux_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if dist.get_rank() == 0 and step % args.log_interval == 0:
                print(f"step {step}/{total_steps}  loss={loss.item():.4f}")
            step += 1

    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM FSDP training")
    p.add_argument("--model", type=str, required=True,
                   help="模型架构 YAML (configs/models/*.yaml)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer_path", type=str, default="")
    p.add_argument("--full_shard", action="store_true", help="ZeRO-3 (default ZeRO-2)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_cfg = ModelConfig.from_yaml(args.model)
    print(f"Model: {args.model}")
    print(f"  d_model={model_cfg.d_model}  layers={model_cfg.num_layers}  "
          f"heads={model_cfg.num_heads}/{model_cfg.num_kv_heads}")
    train(args, model_cfg)
