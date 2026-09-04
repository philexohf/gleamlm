"""
DeepSpeed 分布式训练示例 — 对比 ZeRO-2 / ZeRO-3 + CPU offload

单机 8 卡 (0.6B+):
  deepspeed manual/deepspeed.py --data ./data --deepspeed_config configs/deepspeed_zero2.json

双机 16 卡 (1B+, CPU offload):
  deepspeed manual/deepspeed.py --data ./data \
    --deepspeed_config configs/deepspeed_config.json

选型: DDP + activation checkpointing 先 → 不够上 ZeRO-2 (分片 optimizer states ~70% 显存)
→ 还不够才 ZeRO-3 (参数也分片，通信 ×3 但显存最省)。
CPU offload 省显存但 PCIe 带宽瓶颈，forward/backward 慢 2-3×。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 本脚本文件名 manual/deepspeed.py 会 shadow 真实 deepspeed 包（script 目录在 sys.path[0]）。
# import 前先把脚本目录从 sys.path 剔除，加载真实包后再恢复。
_here = os.path.dirname(os.path.abspath(__file__))
_saved_path = [p for p in sys.path if os.path.abspath(p) != _here]
sys.path[:] = _saved_path
import deepspeed  # noqa: E402

sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, ".."))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402

from gleamlm.data.dataset import tokenize_and_group
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, ModelConfig
from gleamlm.trainer.schedulers import get_lr_cosine


def train(args, model_cfg: ModelConfig):
    if args.deepspeed_config and not os.path.exists(args.deepspeed_config):
        raise SystemExit(f"--deepspeed_config not found: {args.deepspeed_config}")
    # 新版 deepspeed(>=0.15) 已移除 deepspeed.init_distributed：用 torch dist 显式初始化
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        stage = int(os.environ.get("ZERO_STAGE", 3)) if args.deepspeed_config else 0
        print(f"DeepSpeed ZeRO-{stage}, world_size={dist.get_world_size()}")

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
    )

    dataset = tokenize_and_group(args.data, tokenizer, model_cfg.max_seq_len)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # canonical 用法: launcher 提供 --deepspeed --deepspeed_config，DS 从 args 读配置，
    # initialize 不再重复传 config_params（同时给会触发 assert）
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    step = 0
    model_engine.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits, _, aux_loss, _ = model_engine(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss + 0.01 * aux_loss

            model_engine.backward(loss)
            model_engine.step()

            if local_rank == 0 and step % args.log_interval == 0:
                print(f"step {step}  loss={loss.item():.4f}")
            step += 1


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM DeepSpeed training")
    p.add_argument("--model", type=str, required=True,
                   help="模型架构 YAML (configs/models/*.yaml)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer_path", type=str, default="")
    p.add_argument("--local_rank", type=int, default=-1, help="provided by deepspeed launcher")
    p.add_argument("--deepspeed_config", type=str, default="")
    p.add_argument("--deepspeed", action="store_true", help="enable deepspeed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_cfg = ModelConfig.from_yaml(args.model)
    print(f"Model: {args.model}")
    print(f"  d_model={model_cfg.d_model}  layers={model_cfg.num_layers}  "
          f"heads={model_cfg.num_heads}/{model_cfg.num_kv_heads}")
    train(args, model_cfg)
