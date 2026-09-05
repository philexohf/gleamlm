"""
预训练脚本 — 展示完整训练管线：
   1. 数据加载 + tokenize + block 切分
   2. 模型初始化
   3. AMP + 梯度累积 + LR 调度 + 断点续训
   4. Checkpoint 保存
   5. DDP 分布式训练（torchrun 一行启动）

用法:
  # 0 数据（统一工业格式，两轨共用核心库管线 data_tools/pretrain/run_pipeline.py 的产物）:
  python data_tools/pretrain/run_pipeline.py \
      --sources wiki --tokenizer bbpe \
      --tokenizer-path checkpoints/bbpe_24k \
      --output-prefix data/processed/wiki_zh

  # 单卡（--data 传 .bin/.idx 前缀，自动识别 mmap 格式；
  #      也可以传 .txt 文本，小数据场景自动走 HF datasets）
  python manual/pretrain.py --model configs/nano.yaml --data data/processed/wiki_zh
  python manual/pretrain.py --model configs/nano.yaml --data ./data.txt --resume ./checkpoints/step_1000.pt

  # 训练超参默认值来自同一 YAML（training/lr/advanced 段），CLI 可覆盖
  python manual/pretrain.py --model configs/nano.yaml --data ./data.txt --lr 5e-4 --epochs 2

  # 观测模式: 默认 tqdm 进度条（仅主进程渲染）；--no-pbar 切回日志式
  # （每 log_interval 步一行，适合输出重定向/无人值守）
  python manual/pretrain.py --model configs/nano.yaml --data data/nano/pretrain/train

  # 多卡 DDP（单机 4 卡示例）
  torchrun --nproc_per_node=4 manual/pretrain.py \
      --model configs/lite.yaml --data ./data.txt --output_dir ./checkpoints

  # 多卡 + torch.compile（Ampere+ GPU 额外加速 ~30%）
  torchrun --nproc_per_node=4 manual/pretrain.py \
      --model configs/lite.yaml --data ./data --compile --num_workers 4

  # 注: 0.6B 不走手写轨，由工业轨训练 (industrial/: Megatron 预训练 + HF 微调/对齐)
  #     手写轨覆盖 Nano 40M / Lite 87M / Pro 126M
"""

# 标准训练循环 6 环节: 数据 (tokenize_and_group) → 模型 → 优化器 (AdamW) →
# LR 调度 → AMP + 梯度累积 + clip → checkpoint / wandb。

import argparse
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from itertools import islice
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

try:
    import wandb
except ImportError:
    wandb = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from gleamlm.data.dataset import tokenize_and_group
from gleamlm.models.model import GleamLMModel, GQA, MLP, MoE
from gleamlm.models.attention_variants import NoPEGQA, AliBiGQA, SlidingWindowGQA
from gleamlm.trainer.base_trainer import (
    build_optimizer_param_groups,
    ddp_cleanup,
    ddp_setup,
    evaluate,
    is_main_process,
    set_seed,
)
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, ModelConfig, load_config_v2
from gleamlm.trainer.schedulers import get_lr_cosine, get_lr_wsd
from gleamlm.utils.torch_utils import safe_autocast
from gleamlm.tokenizer.tokenizer import BBPETokenizer

    # 变体注册表: CLI string → class
ATTN_REGISTRY = {"gqa": GQA, "nope": NoPEGQA, "alibi": AliBiGQA, "sliding": SlidingWindowGQA}
FFN_REGISTRY = {"mlp": MLP, "moe": MoE}


# ddp_setup / ddp_cleanup / is_main_process 见 gleamlm/trainer/base_trainer.py。


def _gpu_stats(device, local_rank):
    """GPU 占用 (util%, 进程显存 GiB, 峰值 GiB, 总显存 GiB) — 对齐 nvidia-smi。

    Windows 上 torch.cuda.utilization 因 pynvml 缺失抛异常（实测），若直接
    fallback 0 会误导监控（GPU 满载却显示 0%）；此处回退 nvidia-smi 子进程
    查询（每 log_interval 步一次，毫秒级开销）。显存取 memory.used（进程
    占用）而非 PyTorch 缓存分配器视图 allocated（只含活跃张量，远小于真实）。
    显存统一 GiB 口径（bytes/2**30 或 MiB/1024），与 nvidia-smi 读数一致。
    """
    if device.type != "cuda":
        return 0.0, 0.0, 0.0, 0.0
    try:
        util = float(torch.cuda.utilization(device))
        mem = torch.cuda.memory_allocated(device) / 2**30
        mem_max = torch.cuda.max_memory_allocated(device) / 2**30
        mem_total = torch.cuda.get_device_properties(device).total_memory / 2**30
        return util, mem, mem_max, mem_total
    except Exception:
        pass
    try:
        kwargs = {}
        if os.name == "nt":
            # Ctrl+C 隔离: 子进程默认与主进程同 console 前台组，用户 Ctrl+C 会
            # 同时中断 nvidia-smi（其挂起不退出会拖住下方 check_output 等待，
            # 表现为退出卡住）；新进程组使其不受 SIGINT 影响，正常跑完即退。
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=3, **kwargs)
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        # 多卡按 local_rank 取对应 GPU 行
        u, mem_mib, total_mib = lines[min(local_rank, len(lines) - 1)].split(",")
        return float(u), float(mem_mib) / 1024, 0.0, float(total_mib) / 1024
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def train(args, model_cfg: ModelConfig):
    if dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    if is_main_process():
        print(f"Device: {device}  (world_size={dist.get_world_size() if dist.is_initialized() else 1})")

    # GPU 性能开关: cudnn.benchmark 自动选最优卷积算法；
    # TF32 matmul (19 位尾数) 精度略降但速度 ~2×，训练 loss 不受影响。
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if is_main_process():
            gpu_name = torch.cuda.get_device_name(local_rank)
            props = torch.cuda.get_device_properties(local_rank)
            mem_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
            print(f"GPU: {gpu_name}  Memory: {mem_gb:.1f}GB")

    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)
    if is_main_process():
        print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    model = GleamLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        d_ff=model_cfg.d_ff or int(8 / 3 * model_cfg.d_model),
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        pad_token_id=tokenizer.pad_id,
        tie_weights=model_cfg.tie_weights,
        use_flash_attn=model_cfg.use_flash_attn,
        use_gradient_checkpointing=model_cfg.use_gradient_checkpointing,
        attn_variant=ATTN_REGISTRY[model_cfg.attn_type],
        ffn_variant=FFN_REGISTRY[model_cfg.ffn_type],
        num_experts=model_cfg.num_experts,
        top_k=model_cfg.top_k,
        rope_scale=model_cfg.rope_scale,
        rope_factor=model_cfg.rope_factor,
        rope_theta=model_cfg.rope_theta,
        layer_configs=model_cfg.layer_configs,
    ).to(device)

    # torch.compile: 把 Python forward 编译成 Triton 内核，融合算子减少 launch 开销；
    # 需要 Ampere+ (SM80+)，不支持的 GPU 静默退回原模式。
    # 注意: 不用 reduce-overhead (CUDA Graph) — tie_weights + 梯度累积会触发
    # "accessing tensor output of CUDAGraphs" 崩溃；mode=default 无此问题。
    if args.compile and device.type == "cuda":
        if is_main_process():
            print("Compiling model with torch.compile (mode=default)...")
        model = torch.compile(model, mode="default")

    # 顺序: compile 之后 DDP —— compile 对 parameters() 透明；
    # bucket_cap_mb 是梯度 all-reduce 的桶大小 (默认 25MB)；
    # find_unused_parameters=False 省通信 (模型无 unused 参数)。
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            bucket_cap_mb=args.ddp_bucket_mb,
            find_unused_parameters=False,
        )
        raw_model = model.module
    else:
        raw_model = model

    if is_main_process():
        total = sum(p.numel() for p in raw_model.parameters())
        print(f"Model: {total / 1e6:.2f}M params")
        if wandb is not None:
            wandb.init(
                project=args.wandb_project or "gleamlm",
                name=args.wandb_run_name or f"pretrain_{model_cfg.d_model}d_{model_cfg.num_layers}l",
                config=vars(args),
            )

    # DistributedSampler 按 rank 切分数据 (DDP 本身不处理数据分布!)；
    # set_epoch 每 epoch 重新 shuffle；num_workers 掩盖 CPU 预处理瓶颈。
    # 确定性采样（对齐 nanotron）: 单机/多卡统一用 DistributedSampler，
    # 固定 seed + set_epoch 保证重建采样序列确定 → 断点续训可精确回到断点。
    dataset = tokenize_and_group(args.data, tokenizer, model_cfg.max_seq_len)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    if is_main_process():
        print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    # 验证集（可选）
    val_loader = None
    val_interval = args.val_interval or args.save_interval
    if args.val_data:
        val_dataset = tokenize_and_group(args.val_data, tokenizer, model_cfg.max_seq_len)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        if is_main_process():
            print(f"Val dataset: {len(val_dataset)} samples, {len(val_loader)} batches")

    # TensorBoard（可选）
    writer = None
    if is_main_process() and args.tensorboard and SummaryWriter is not None:
        log_dir = os.path.join(args.output_dir, "runs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: tensorboard --logdir {log_dir}")

    # AdamW: weight decay 从梯度里拿出来单独做 (w *= 1-lr·λ)，不污染动量；
    # betas=(0.9,0.95): β2 更小让二阶动量更快适应大模型的高梯度方差；
    # weight_decay=0.1 是 1B 以下模型的经验区间。
    # wd 分组（SmolLM3/Megatron 一致）: embedding + norm 不加 wd，
    # 其余矩阵权重正常衰减 —— 提升训练稳定性且对齐工业实践。
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(raw_model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,  # 作为分组默认值，组内已覆写
    )
    # GradScaler: 解决 FP16 梯度 underflow (BF16 训练时 disabled)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # checkpoint 必须含 model + optimizer + scaler + step:
    # 只存权重会重置 LR 调度 / Adam 动量 / scaler 状态，续训 loss 会跳。
    start_step = 0
    start_epoch = 0
    start_batch = 0
    start_acc = 0
    start_consumed = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        raw_model.load_state_dict(sd, strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        start_batch = ckpt.get("batch", 0)
        # 断点续训必须恢复累积边界: checkpoint 若保存于 accumulate 组中间，
        # acc 归零会让恢复后的组边界与原始轨迹错位
        start_acc = ckpt.get("acc", 0)
        # 全局样本计数（权威位置）: 恢复时用它精确定位数据；
        # 旧 checkpoint 无此字段时回退到 batch 序号
        start_consumed = ckpt.get("consumed_train_samples", 0)
        if is_main_process():
            print(f"Resumed from step {start_step} (epoch {start_epoch}, batch {start_batch})")

    # step 以 optimizer step 计 (micro-batch / accumulate)，与 sft/dpo 的 global_step 对齐
    total_steps = args.epochs * math.ceil(len(loader) / args.accumulate)
    step = start_step
    acc_steps = start_acc
    # 全局已消费样本数: 每处理一个 micro-batch += batch_size；
    # 是断点续训的权威数据位置（与 DP 规模解耦，对齐 nanotron consumed_train_samples）
    consumed_train_samples = start_consumed
    best_val_loss = float("inf")
    raw_model.train()

    # 观测模式: 默认 tqdm 进度条（optimizer step 粒度、跨 epoch 连续、
    # initial=断点步数，resume 后位置精确）；--no-pbar → 日志式输出。
    # 进度条仅主进程渲染（多卡从进程静默，与旧 base_trainer 同策略）；
    # mininterval 节流重绘频率，避免交互终端刷屏/重定向被 \r 污染。
    pbar = None
    # GPU 显存缓存: set_postfix 每步引用，但查询较重（NVML 缺失时起
    # nvidia-smi 子进程）→ 每 log_interval 步刷新一次；此处先查一次让
    # 第 1 步即有真实值
    gpu_util, gpu_mem, gpu_mem_max, gpu_mem_total = 0.0, 0.0, 0.0, 0.0
    if args.pbar and is_main_process():
        pbar = tqdm(total=total_steps, initial=step, desc="pretrain",
                    mininterval=5, unit="step")
        if device.type == "cuda":
            gpu_util, gpu_mem, gpu_mem_max, gpu_mem_total = _gpu_stats(device, local_rank)

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        # 断点续训的数据衔接: set_epoch + 确定性采样固定了每个 epoch 的
        # shuffle 顺序（确定性），因此跳过已消费的 batch 即可精确续上，
        # 避免恢复后该 epoch 数据从头重放一遍（数据重复消费会导致样本
        # 被重复训练、混合比例偏移）。
        batch_iter = enumerate(loader)
        if epoch == start_epoch:
            # 权威位置 = consumed_train_samples（全局样本数，与 DP 解耦）；
            # 换算成该 rank 应跳过的 batch 序号 = 全局样本 / (batch_size × dp)
            dp_size = dist.get_world_size() if dist.is_initialized() else 1
            skip_batches = (consumed_train_samples // args.batch_size) // dp_size
            # 旧 checkpoint 无 consumed 字段时回退到 batch 序号
            if consumed_train_samples == 0 and start_batch > 0:
                skip_batches = start_batch
            if skip_batches > 0:
                batch_iter = islice(batch_iter, skip_batches, None)
        for batch_idx, batch in batch_iter:
            step_start = time.perf_counter()
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            # 每个 step 手动设置 lr（不用 scheduler.step()）:
            # 断点续训时 step 从 start_step 继续，手动乘 mult 更直观。
            if args.lr_scheduler == "wsd":
                lr_mult = get_lr_wsd(step, total_steps, args.warmup_ratio, args.stable_ratio,
                                       args.min_lr_ratio, args.wsd_decay_style)
            else:
                lr_mult = get_lr_cosine(step, total_steps, args.warmup_ratio, args.min_lr_ratio)
            lr = args.lr * lr_mult
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # 梯度累积 + no_sync: no_sync 内跳过每个 micro-batch 的
            # all-reduce，最后一步才同步一次 → 通信量 ÷ accumulate；
            # epoch 末尾不足 accumulate 的残差批也强制触发 step，避免梯度悬置。
            is_last_acc = (acc_steps + 1) % args.accumulate == 0 or batch_idx == len(loader) - 1
            sync_ctx = model.no_sync() if (dist.is_initialized() and not is_last_acc) else nullcontext()

            # 前向 + loss: safe_autocast BF16 自动混合精度 (GPU/CPU 自适应)
            with sync_ctx, safe_autocast(enabled=(device.type == "cuda")):
                logits, _, aux_loss, _ = model(x)
                # dataset 已做好 shift（y[i]=x[i+1]，无 pad 无需 ignore_index）。
                # 文档边界掩码（对齐 SmolLM `_use_doc_masking` / 工业轨 eod_mask_loss）:
                # x 为 EOD(=2, im_end) 的位置其目标是下一文档首 token，跨文档预测不参与 loss，
                # 只训练"生成 EOD 本身"；label_smoothing=0.1: 压 one-hot 过自信 (LLaMA 标配)。
                eod_id = tokenizer.eos_id  # 预训练文档边界 = <|im_end|> (2)
                doc_mask = (x != eod_id).float()
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1),
                    reduction="none",
                    label_smoothing=args.label_smoothing,
                )
                loss = (loss.view_as(x) * doc_mask).sum() / doc_mask.sum()
                # MoE aux loss: 驱动 router 均衡分配 token (0.01 是 Switch 标准值)
                loss = loss + 0.01 * aux_loss
                # z-loss: 正则化 logits 量级，防 softmax 饱和到 one-hot 梯度消失
                if args.z_loss > 0:
                    z = logits.float().logsumexp(dim=-1).pow(2).mean()
                    loss = loss + args.z_loss * z

            # loss 除本周期实际 micro-batch 数 (梯度平均而非求和，与 sft/dpo 一致)；
            # 残差批按实际批数除，避免归一化过头导致梯度偏小。
            denom = ((acc_steps % args.accumulate) + 1) if batch_idx == len(loader) - 1 else args.accumulate

            # AMP 协议: scale(loss).backward → unscale → clip → step → update；
            # clip 必须 unscale 后 (阈值作用于 ×scale 梯度会失准)；
            # 梯度 inf/nan 时 scaler 跳过 step 并减半 scale。
            scaler.scale(loss / denom).backward()
            acc_steps += 1
            consumed_train_samples += args.batch_size
            dt = time.perf_counter() - step_start

            if is_last_acc:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

            # 观测输出仅在本 accumulate 组末发生一次（重复守卫同 save/val）。
            # --pbar(默认): 每组末 update + set_postfix，loss/lr/tok/s 每步连续刷新
            # （对齐旧 base_trainer 节奏：从第 1 步就有值、平滑不跳跃）；
            # GPU 显存走缓存（查询较重——NVML 缺失时起 nvidia-smi 子进程，
            # 每步执行太贵——每 log_interval 步刷新一次，显存变化缓慢无感）。
            # --no-pbar 日志式: 每 log_interval 步打一行；wandb 两模式同节奏。
            if is_main_process() and is_last_acc:
                if args.pbar:
                    pbar.update(1)
                if step % args.log_interval == 0:
                    gpu_util, gpu_mem, gpu_mem_max, gpu_mem_total = _gpu_stats(device, local_rank)
                    tok_per_sec = args.batch_size * model_cfg.max_seq_len / dt
                    progress_pct = 100.0 * step / max(1, total_steps)
                    if not args.pbar:
                        print(f"step {step}/{total_steps} ({progress_pct:.1f}%)  loss={loss.item():.4f}  lr={lr:.6f}  {tok_per_sec/1e3:.1f}k tok/s  GPU:{gpu_mem:.1f}/{gpu_mem_total:.1f}G")
                    if wandb is not None:
                        wandb.log({
                            "loss": loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "step": step,
                            "epoch": epoch,
                            "progress_pct": progress_pct,
                            "tok_per_sec": tok_per_sec,
                            "time_per_step_ms": dt * 1000,
                            "gpu_util": gpu_util,
                            "gpu_mem_gb": gpu_mem,
                            "gpu_mem_peak_gb": gpu_mem_max,
                        }, step=step)
                if args.pbar:
                    # 每步刷新（旧 base_trainer 组末 set_postfix 同款），
                    # 数值连续变化；重绘频率由 mininterval=5 节流
                    tok_per_sec = args.batch_size * model_cfg.max_seq_len / dt
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr:.6f}",
                        "tok/s": f"{tok_per_sec/1e3:.1f}k",
                        "GPU": f"{gpu_mem:.1f}/{gpu_mem_total:.1f}G",
                    })

            # 周期保存/验证与日志同规则：只在组末判定一次，否则命中步会
            # 被组内每个 micro-batch 重复执行（save 重复全量写盘、val 重复跑整集）。
            if is_main_process() and is_last_acc and step > 0 and step % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
                torch.save({
                    "step": step, "epoch": epoch, "batch": batch_idx + 1,
                    "acc": acc_steps % args.accumulate,
                    "consumed_train_samples": consumed_train_samples,
                    "_config": _ckpt_model_config(model_cfg, tokenizer.get_vocab_size()),
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "loss": loss.item(),
                }, ckpt_path)
                print(f"Saved: {ckpt_path}")

            # 周期验证 + best model
            if is_main_process() and is_last_acc and val_loader is not None and step > 0 and step % val_interval == 0:
                raw_model.eval()
                # 仅 rank 0 跑完整 val 集 (val_loader 只在 rank 0 构建)；
                # world_size=1 跳过 all_reduce，避免其他 rank 不参与验证导致的死锁。
                val_loss, val_ppl = evaluate(raw_model, val_loader, device,
                                             pad_token_id=tokenizer.pad_id,
                                             world_size=1)
                raw_model.train()
                print(f"  Val step {step}: loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                if writer is not None:
                    writer.add_scalar("Eval/Loss", val_loss, step)
                    writer.add_scalar("Eval/Perplexity", val_ppl, step)
                if wandb is not None:
                    wandb.log({"val_loss": val_loss, "val_ppl": val_ppl, "step": step}, step=step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        "step": step, "epoch": epoch,
                        "_config": _ckpt_model_config(model_cfg, tokenizer.get_vocab_size()),
                        "model_state_dict": raw_model.state_dict(),
                        "val_loss": val_loss, "val_ppl": val_ppl,
                    }, best_path)
                    print(f"  Best model saved (val_loss={val_loss:.4f}) -> {best_path}")

        epoch_duration = time.perf_counter() - epoch_start
        if is_main_process():
            print(f"Epoch {epoch} done: {epoch_duration:.0f}s = {epoch_duration/60:.1f} min")
            if wandb is not None:
                wandb.log({"epoch": epoch, "epoch_duration": epoch_duration}, step=step)

    if pbar is not None:
        pbar.close()

    if is_main_process():
        ckpt_path = os.path.join(args.output_dir, "final.pt")
        torch.save({
            "step": step, "epoch": args.epochs,
            "_config": _ckpt_model_config(model_cfg, tokenizer.get_vocab_size()),
            "model_state_dict": raw_model.state_dict(),
        }, ckpt_path)
        print(f"Done. Final checkpoint: {ckpt_path}")
        if writer is not None:
            writer.close()
        if wandb is not None:
            wandb.finish()


def _ckpt_model_config(model_cfg: ModelConfig, vocab_size: int) -> dict:
    """记录模型真实结构，供下游 (grpo/ppo/distill/serve) 精确重建。"""
    result = model_cfg.model_dump()
    result["vocab_size"] = vocab_size
    return result


def _load_training_defaults(training_path: str | None) -> dict[str, Any]:
    """从训练 YAML 加载超参默认值，CLI 可覆盖。"""
    if training_path is None:
        return {}
    cfg = load_config_v2(training_path)
    t = cfg.training
    lr = cfg.lr
    return {
        "epochs": t.epochs, "batch_size": t.batch_size,
        "accumulate": t.accumulate_grad, "weight_decay": t.weight_decay,
        "clip": t.clip_grad, "log_interval": t.log_interval,
        "save_interval": t.save_interval,
        "seed": t.seed,
        "lr_scheduler": lr.type, "lr": lr.lr,
        "warmup_ratio": lr.warmup_ratio, "stable_ratio": lr.stable_ratio,
        "min_lr_ratio": lr.min_lr_ratio, "z_loss": cfg.advanced.z_loss_weight,
        "label_smoothing": t.label_smoothing,
        "num_workers": cfg.advanced.num_workers or 0,
        "tokenizer_path": cfg.data.tokenizer_path or "",
        "data_dir": cfg.data.data_dir or "",
        "checkpoint_dir": cfg.data.checkpoint_dir or "",
    }


def _show_full_help():
    """不带 --model 时也能显示完整帮助。"""
    p = argparse.ArgumentParser(description="GleamLM pretraining (DDP-ready)",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="""Examples:
  python pretrain.py --model configs/nano.yaml --data ./data.txt
  python pretrain.py --model configs/nano.yaml --data ./data.txt --lr 5e-4
  torchrun --nproc_per_node=4 pretrain.py --model configs/lite.yaml --data ./data.txt""")
    p.add_argument("--model", type=str, required=True,
                   help="模型架构 YAML (configs/*.yaml，完整 GleamLMConfig 结构时自动取 model 段) — 必传")
    p.add_argument("--training", type=str, default=None,
                   help="训练默认超参 YAML (默认复用 --model 的同一文件)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--stable_ratio", type=float, default=0.80)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--wsd_decay_style", type=str, default="linear",
                   choices=["cosine", "linear"],
                   help="WSD decay 段衰减方式: linear 对齐 nano 实际 (nano_wsd_linear_v2/SmolLM3)，lr 线性降到 min_lr_ratio")
    p.add_argument("--z_loss", type=float, default=1e-5,
                   help="Z-Loss 系数 (SmolLM3 用 1e-5，防 logits 爆炸；0 禁用)")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tokenizer_path", type=str, default="")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--ddp_bucket_mb", type=int, default=25)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--label_smoothing", type=float, default=0.0, help="CrossEntropy label smoothing (LLaMA 用 0.1)")
    p.add_argument("--pbar", action=argparse.BooleanOptionalAction, default=True,
                   help="用 tqdm 进度条输出（默认开；--no-pbar 切日志式；仅主进程渲染）")
    p.print_help()


# 有 LOCAL_RANK → torchrun 启动走 DDP 初始化；没有 → 单卡直接训练。
# 一个脚本两种用法，无需用户判断。

def main():
    # 1. --help 不带 --model 时直接显示
    if "--help" in sys.argv or "-h" in sys.argv:
        _show_full_help()
        return

    # 2. 预解析 model/training
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model", type=str, required=True)
    pre_parser.add_argument("--training", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    # 3. 加载模型架构（必须）
    model_cfg = ModelConfig.from_yaml(pre_args.model)

    # 3. 加载训练默认值 + 全量 parser
    # 未单独指定 --training 时复用 --model 的同一 YAML（其 training/lr/advanced 段
    # 即为训练默认值；纯模型架构文件则全部回落 argparse 内置默认）
    training_defaults = _load_training_defaults(pre_args.training or pre_args.model)
    parser = argparse.ArgumentParser(description="GleamLM pretraining (DDP-ready)")
    parser.add_argument("--model", type=str, default=pre_args.model,
                        help="模型架构 YAML (configs/*.yaml，自动取 model 段)")
    parser.add_argument("--training", type=str, default=pre_args.training,
                        help="训练默认超参 YAML (默认复用 --model 的同一文件)")
    # 训练超参
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=training_defaults.get("epochs", 3))
    parser.add_argument("--batch_size", type=int, default=training_defaults.get("batch_size", 4))
    parser.add_argument("--accumulate", type=int, default=training_defaults.get("accumulate", 8))
    parser.add_argument("--lr", type=float, default=training_defaults.get("lr", 3e-4))
    parser.add_argument("--seed", type=int, default=training_defaults.get("seed", 42),
                        help="随机种子（数据采样/参数初始化的确定性复现基准）")
    parser.add_argument("--lr_scheduler", type=str, default=training_defaults.get("lr_scheduler", "cosine"),
                        choices=["cosine", "wsd"])
    parser.add_argument("--warmup_ratio", type=float, default=training_defaults.get("warmup_ratio", 0.01))
    parser.add_argument("--stable_ratio", type=float, default=training_defaults.get("stable_ratio", 0.80))
    parser.add_argument("--min_lr_ratio", type=float, default=training_defaults.get("min_lr_ratio", 0.05))
    parser.add_argument("--wsd_decay_style", type=str, default="linear",
                        choices=["cosine", "linear"],
                        help="WSD decay 段衰减方式: linear 对齐 nano 实际 (nano_wsd_linear_v2/SmolLM3)，lr 线性降到 min_lr_ratio")
    parser.add_argument("--z_loss", type=float, default=training_defaults.get("z_loss", 1e-5),
                        help="Z-Loss 系数 (SmolLM3 用 1e-5，防 logits 爆炸；0 禁用)")
    parser.add_argument("--weight_decay", type=float, default=training_defaults.get("weight_decay", 0.1))
    parser.add_argument("--clip", type=float, default=training_defaults.get("clip", 1.0))
    parser.add_argument("--log_interval", type=int, default=training_defaults.get("log_interval", 10))
    parser.add_argument("--save_interval", type=int, default=training_defaults.get("save_interval", 500))
    # 执行选项
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=training_defaults.get("num_workers", 0))
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--ddp_bucket_mb", type=int, default=25)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None, help="验证数据路径（可选，用于周期性验证）")
    parser.add_argument("--val_interval", type=int, default=None, help="验证间隔步数（默认=save_interval）")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="CrossEntropy label smoothing (LLaMA 用 0.1)")
    parser.add_argument("--tensorboard", action="store_true", help="启用 TensorBoard 日志")
    parser.add_argument("--pbar", action=argparse.BooleanOptionalAction, default=True,
                        help="用 tqdm 进度条输出（默认开；--no-pbar 切日志式；仅主进程渲染）")
    args = parser.parse_args(remaining)
    args.model = None  # 清掉，不参与训练逻辑

    # YAML data 路径覆盖（CLI 没传时才从 YAML 取）
    if training_defaults.get("tokenizer_path") and args.tokenizer_path == "":
        args.tokenizer_path = training_defaults["tokenizer_path"]
    if training_defaults.get("data_dir") and args.data == parser.get_default("data"):
        pass  # --data is required, must be from CLI
    if training_defaults.get("checkpoint_dir") and args.output_dir == "./checkpoints":
        args.output_dir = training_defaults["checkpoint_dir"]

    # 4. 启动训练
    # 固定 seed（含模型初始化 + 确定性数据采样），保证实验可复现；
    # 与断点续训的确定性恢复配合：同一 seed 下重建的采样序列可精确回到断点。
    set_seed(args.seed)
    if is_main_process():
        print(f"Model: {pre_args.model}")
        print(f"  d_model={model_cfg.d_model}  layers={model_cfg.num_layers}  "
              f"heads={model_cfg.num_heads}/{model_cfg.num_kv_heads}  "
              f"d_ff={model_cfg.d_ff}  seq_len={model_cfg.max_seq_len}")
    if "LOCAL_RANK" in os.environ:
        ddp_setup()
    try:
        train(args, model_cfg)
    except KeyboardInterrupt:
        # Ctrl+C 优雅退出：pbar 随栈展开 GC 自动 close（tqdm __del__），
        # 此处只做提示；step_*.pt 每 save_interval 步已落盘，可断点续训。
        if is_main_process():
            print("\n训练被中断 (Ctrl+C)。")
            print(f"续训: python manual/pretrain.py --model {pre_args.model} "
                  f"--data {args.data} --resume {args.output_dir}/step_*.pt")
        raise SystemExit(130)
    finally:
        if dist.is_initialized():
            ddp_cleanup()


if __name__ == "__main__":
    main()
