"""
Megatron-Core 预训练 — 最小训练循环（学习工业级预训练框架）。

学习要点（全程与手写轨 `manual/pretrain.py` 对比）:
  1. 并行方式:      手写轨把 DDP 写在代码里；Megatron 把 TP/PP/DP 做成配置，
                    `parallel_state.initialize_model_parallel` 一行完成并行初始化
  2. 模型:          手写轨用 GleamLMModel；Megatron 用自家 GPTModel 族
                    （含 tensor parallel 的列/行切分 Linear，与手写模型不通用）
  3. 数据:          手写轨内存 Dataset；Megatron 用 .bin/.idx mmap（见 data_tools/pretrain/run_pipeline.py）
  4. 优化器:        生产用 get_megatron_optimizer（distributed optimizer，
                    与 ZeRO-2 同思路的优化器状态分片）；本脚本用 torch 原生便于教学
  5. checkpoint:    手写轨单文件；Megatron 生产按 rank 分片多文件

用法（Linux / WSL2，megatron-core 已安装）:
  # 0 数据（复用核心库管线，生成 megatron 0.16 标准 .bin/.idx）:
  python data_tools/pretrain/run_pipeline.py \
    --sources wiki --tokenizer bbpe \
    --tokenizer-path checkpoints/bbpe_24k \
    --output-prefix data/processed/wiki_zh

  # 单卡（对应手写轨 manual/pretrain.py 单卡）:
  python industrial/pretrain.py \
    --config industrial/configs/0.6b.yaml \
    --data data/processed/wiki_zh

  # 多卡数据并行（对应手写轨 torchrun 版本）:
  torchrun --nproc_per_node=4 industrial/pretrain.py \
    --config industrial/configs/0.6b.yaml \
    --data data/processed/wiki_zh

  # 张量并行 2 + 流水线并行 2（Megatron 招牌 3D 并行，8 卡）:
  torchrun --nproc_per_node=8 industrial/pretrain.py \
    --config industrial/configs/0.6b.yaml \
    --data data/processed/wiki_zh

生产级用法（Megatron-LM 官方入口，功能最全，直接可跑）:
  torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 37 --hidden-size 1024 --num-attention-heads 16 \
    --num-query-groups 8 \
    --seq-length 4096 --max-position-embeddings 4096 \
    --micro-batch-size 2 --global-batch-size 32 \
    --train-iters 10000 --lr 2.0e-4 --min-lr 1.0e-5 \
    --lr-decay-style cosine --lr-warmup-iters 500 \
    --weight-decay 0.1 --clip-grad 1.0 --bf16 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model checkpoints/bbpe_24k/hf_export \
    --data-path data/processed/wiki_zh --split 949,50,1 \
    --save checkpoints/megatron --load checkpoints/megatron \
    --log-interval 10 --save-interval 1000 --eval-interval 500

API 说明: 本脚本以 megatron-core 0.9+ 为例；若版本不同，
参考官方 examples/run_simple_mcore_train_loop.py 微调。
"""

import argparse
import bisect
import itertools
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import yaml

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader, Dataset

from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig

# 以下为学习脚本的最小依赖集；生产预训练请使用官方 pretrain_gpt.py
# from megatron.core.models.gpt.gpt_model import GPTModel  ← 延迟导入（见 build_model）


class IndexedDatasetWrapper(Dataset):
    """把 .bin/.idx 包装成 PyTorch Dataset + 定长 block 切分。

    对比手写轨 `tokenize_and_group`: 那里是内存里切 block，
    这里通过 mmap 随机访问，任意大语料不占内存。
    """

    def __init__(self, prefix: str, seq_len: int):
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        self.ds = IndexedDataset(prefix)
        self.seq_len = seq_len
        self.lengths = self.ds.sequence_lengths.tolist() or [0]
        # 每个文档的起始 token 位置（累计和），用于二分定位 start_tok 所在文档
        self.cumsum = list(itertools.accumulate(self.lengths))
        total = self.cumsum[-1]
        self.num_blocks = max(total // seq_len, 1)

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int):
        # 跨文档连续采样: 从 start_tok 所在文档起连续拼接，直到凑满 seq_len；
        # 文档末尾不 padding（生产 Megatron 的 BlendedMegatronDatasetBuilder 同思路）
        start_tok = idx * self.seq_len
        tokens: list[int] = []
        doc_i = bisect.bisect_right(self.cumsum, start_tok)
        pos = start_tok - (self.cumsum[doc_i - 1] if doc_i > 0 else 0)
        while len(tokens) < self.seq_len:
            doc = self.ds[doc_i].tolist()
            tokens.extend(doc[pos:])
            pos = 0
            doc_i = (doc_i + 1) % len(self.lengths)
        input_ids = torch.tensor(tokens[: self.seq_len], dtype=torch.long)
        labels = input_ids.clone()
        return input_ids, labels


def build_train_dataset(config, prefix: str, tokenizer, seq_length: int) -> Dataset:
    """官方 GPTDataset + BlendedMegatronDatasetBuilder 构建训练数据集。

    megatron 官方 pretrain_gpt.py 的数据路径:
      BlendedMegatronDatasetBuilder(GPTDataset, sizes, is_built_on_rank, config).build()
    GPTDataset 内部完成跨文档滑窗切块（document/sample/shuffle index），
    返回 {tokens, labels, attention_mask, loss_mask, position_ids} 样本。
    这里替换手写 IndexedDatasetWrapper，对齐官方数据类语义。
    """
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.utils import Split

    dataset_config = GPTDatasetConfig(
        random_seed=42,
        sequence_length=seq_length,
        blend=([prefix], None),
        split="100,0,0",
        path_to_cache=None,
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        create_attention_mask=False,
        allow_ambiguous_pad_tokens=True,
    )

    def is_built_on_rank() -> bool:
        # 每个 rank 独立构建（单机小规模）；官方按 data parallel group 首 rank 建 + 广播
        return True

    builder = BlendedMegatronDatasetBuilder(
        GPTDataset, [None, None, None], is_built_on_rank, dataset_config
    )
    datasets = builder.build()  # [train, valid, test]
    train_ds = datasets[Split.train.value]
    if train_ds is None:
        raise RuntimeError("train split 未构建")
    return train_ds


def build_model(config: TransformerConfig, vocab_size: int, max_sequence_length: int) -> torch.nn.Module:
    """构建 Megatron GPT 模型（延迟导入，未装 megatron 时错误信息更友好）。

    megatron-core 0.16: vocab_size / max_sequence_length / position_embedding_type
    不在 TransformerConfig，由 GPTModel 参数显式传入。
    架构对齐手写轨（GQA/SwiGLU/RMSNorm/RoPE/QK-Norm）:
      - layer spec normalization='RMSNorm' + qk_layernorm=True
      - GPTModel position_embedding_type='rope'（对齐手写轨 RoPE，而非
        megatron 默认 learned_absolute）
    """
    try:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.models.gpt.gpt_model import GPTModel
    except ImportError as e:
        raise SystemExit(
            "未安装 megatron-core。请在 Linux/WSL2 环境执行: pip install megatron-core"
        ) from e

    # 纯 PyTorch layer spec；RMSNorm + QK-Norm 与手写轨架构对齐
    layer_spec = get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")

    return GPTModel(
        config,
        transformer_layer_spec=layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=True,
        post_process=True,          # 含 lm_head
        parallel_output=False,      # TP>1 时为 True（logits 按 rank 切分）
        share_embeddings_and_output_weights=True,  # 与手写轨 weight tying 对齐
        position_embedding_type="rope",   # RoPE（手写轨一致），非 learned_absolute
        rotary_base=config.rotary_base if hasattr(config, "rotary_base") else 10000.0,
    )


def build_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """position_ids: 每行从 0 递增（手写轨在 RoPE 里隐式处理，Megatron 显式传入）。"""
    B, S = input_ids.shape
    return torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)


def forward_backward(model, batch, z_loss_weight: float = 0.0):
    """单个 micro-batch 前向 + 反向，返回 loss（不负责 zero_grad/step/reduce）。

    GPTDataset 返回的 labels 已 shift（roll -1），loss_mask 与 labels 位置对齐；
    因此 CE 直接用 logits[i] 预测 labels[i]（标准 LM 对齐），不可再 shift。
    """
    if isinstance(batch, dict):
        input_ids = batch["tokens"].cuda()
        labels = batch["labels"].cuda()
        loss_mask = batch.get("loss_mask")
    else:  # 兼容 tuple (input_ids, labels)
        input_ids, labels = batch
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        loss_mask = None
    # megatron-core 0.16 需要显式因果 attention_mask（bool，True=屏蔽）。
    # GPTDataset create_attention_mask=False，必须自己构造: 上三角(含未来)=True。
    s = input_ids.size(1)
    causal = torch.triu(
        torch.ones(s, s, dtype=torch.bool, device=input_ids.device), diagonal=1
    )
    attention_mask = causal.unsqueeze(0).expand(input_ids.size(0), 1, s, s)

    # 前向: 不传 labels，模型返回 logits [b, s, h]，
    # 用标准 CE 算 loss —— 规避 megatron 0.16 TP=1 下
    # _VocabParallelCrossEntropy 反向 view+inplace 冲突（第三方 bug）
    logits = model(
        input_ids=input_ids,
        position_ids=build_position_ids(input_ids),
        attention_mask=attention_mask,
    )
    b, s, v = logits.shape
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, v), labels.reshape(-1), reduction="none"
    ).reshape(b, s)
    if loss_mask is None:
        loss = ce.mean()
    else:
        lm = loss_mask.to(device=input_ids.device, dtype=ce.dtype)
        loss = (ce * lm).sum() / max(lm.sum(), 1.0)
    # Z-Loss: 正则化 logits 量级，防 softmax 饱和（对齐手写轨 advanced.z_loss_weight）
    if z_loss_weight > 0:
        z = logits.float().logsumexp(dim=-1).pow(2).mean()
        loss = loss + z_loss_weight * z
    loss.backward()
    return loss.detach()


def reduce_dp_grads(model):
    """DP>1 时对模型梯度做数据并行组内 AVG all-reduce（megatron 旧语义）。

    get_megatron_optimizer 不负责 DP 梯度归约（归约在 mcore 训练栈的
    DistributedDataParallel 里）；这里手动补齐，DP=1 时 no-op。
    平均除数必须用 DP 组大小而非全局 world_size（TP×PP>1 时二者不同）。
    """
    dp_size = parallel_state.get_data_parallel_world_size()
    if dp_size <= 1:
        return
    group = parallel_state.get_data_parallel_group()
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
        p.grad.div_(dp_size)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_eval(model, eval_prefix, tokenizer, seq_length, transformer_config,
              max_batches: int = 200, autocast_ctx=None):
    """训练结束后在验证集上评估 loss/ppl（rank0，只读 mmap）。

    与 forward_backward 相同 CE 口径（labels 已 shift + loss_mask 加权）。
    """
    import torch.nn.functional as F

    dataset = build_train_dataset(transformer_config, eval_prefix, tokenizer, seq_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    total, n = 0.0, 0
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            b = batch if isinstance(batch, dict) else {}
            input_ids = b["tokens"].cuda()
            labels = b["labels"].cuda()
            lm = b["loss_mask"].cuda()
            s = input_ids.size(1)
            causal = torch.triu(
                torch.ones(s, s, dtype=torch.bool, device=input_ids.device), diagonal=1
            )
            attn = causal.unsqueeze(0).expand(input_ids.size(0), 1, s, s)
            with ctx:
                logits = model(
                    input_ids=input_ids,
                    position_ids=build_position_ids(input_ids),
                    attention_mask=attn,
                )
            b_, s_, v = logits.shape
            ce = F.cross_entropy(
                logits.reshape(-1, v), labels.reshape(-1), reduction="none"
            ).reshape(b_, s_)
            loss = (ce * lm).sum() / max(lm.sum(), 1.0)
            total += loss.item()
            n += 1
    model.train()
    avg = total / max(n, 1)
    ppl = 2.718281828 ** avg
    print(f"[eval] {n} batches | val_loss {avg:.4f} | val_ppl {ppl:.2f}")
    if wandb is not None:
        wandb.log({"val_loss": avg, "val_ppl": ppl})
    return avg, n


def main():
    parser = argparse.ArgumentParser(description="Megatron-Core 最小预训练循环")
    parser.add_argument("--config", required=True, help="模型配置 YAML")
    parser.add_argument("--data", required=True, help=".bin/.idx 前缀 (data_tools/pretrain/run_pipeline.py 输出)")
    parser.add_argument("--tokenizer-path", default="gleamlm/tokenizer/checkpoints/bbpe_12k",
                        help="BBPE tokenizer 目录 (GPTDataset 需 vocab_size/eod)")
    parser.add_argument("--out", default="checkpoints/megatron", help="checkpoint 目录")
    parser.add_argument("--load", default=None,
                        help="恢复 checkpoint (out 目录下 iter_<N>.pt 的完整路径)")
    parser.add_argument("--eval-data", default=None,
                        help="验证数据 .bin/.idx 前缀（可选，训练结束后评估 val loss/ppl）")
    parser.add_argument("--wandb_project", default=None,
                        help="wandb project（默认 gleamlm；装即启用，未装自动禁用）")
    parser.add_argument("--wandb_run_name", default=None, help="wandb run 名")
    args = parser.parse_args()

    cfg = load_config(args.config)
    m, p, t = cfg["model"], cfg["parallel"], cfg["training"]

    # ── 1. 初始化分布式（手写轨: ddp_setup() 手动 init_process_group + DDP wrap）──
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=p["tensor_model_parallel_size"],
        pipeline_model_parallel_size=p["pipeline_model_parallel_size"],
        context_parallel_size=p.get("context_parallel_size", 1),
    )
    torch.cuda.set_device(dist.get_rank())
    # megatron-core 0.16: TP 权重初始化需要 model-parallel cuda seed（含 rng tracker）
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(seed=42)

    # 2. TransformerConfig: 手写轨 ModelConfig 转框架配置对象
    #    (megatron-core 0.16: micro/global_batch、clip_grad 由训练循环管理，
    #    use_flash_attention/normalize_rmsnorm 并入 layer spec / normalization)
    config = TransformerConfig(
        num_layers=m["num_layers"],
        hidden_size=m["hidden_size"],
        num_attention_heads=m["num_attention_heads"],
        num_query_groups=m.get("num_query_groups", m["num_attention_heads"]),
        ffn_hidden_size=m["ffn_hidden_size"],
        kv_channels=max(m["hidden_size"] // m["num_attention_heads"], 1),
        tensor_model_parallel_size=p["tensor_model_parallel_size"],
        pipeline_model_parallel_size=p["pipeline_model_parallel_size"],
        # bf16 优先；未显式指定 fp16 时，bf16=False 才默认 fp16（保留 0.6b.yaml 语义）
        fp16=bool(t.get("fp16") if t.get("fp16") is not None else (not t.get("bf16", True))),
        bf16=bool(t.get("bf16", True)),
        layernorm_epsilon=1e-6,  # 对齐手写轨 RMSNorm 默认 eps(1e-6)，与 verify/转换侧一致
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # 对齐手写轨架构: RMSNorm + SwiGLU + 无 bias（Qwen3 标准），RoPE 由 GPTModel 参数指定
        normalization="RMSNorm",
        gated_linear_unit=True,      # SwiGLU（手写轨 MLP 三权重），megatron 默认 False
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        # bf16 下 softmax 默认走 fp32，attention_probs(float) 与 value(bf16) 在
        # bmm 冲突（megatron 0.16 bug）；直接在 bf16 下算 softmax
        attention_softmax_in_fp32=False,
        # Activation recompute（配置驱动；显存换速度，长序列/大批量时开启）
        recompute_granularity=m.get("recompute_granularity"),
        recompute_method=m.get("recompute_method"),
        recompute_num_layers=m.get("recompute_num_layers", 0),
    )

    # 3. 模型（GleamLMModel + DDP wrap 对应 Megatron build_model）
    model = build_model(config, m["vocab_size"], m["max_position_embeddings"]).cuda()

    # 4. 数据（官方 GPTDataset + BlendedMegatronDatasetBuilder，对齐 pretrain_gpt.py）
    from hf.hf_megatron_tokenizer import MegatronBBPETokenizer
    from gleamlm.tokenizer.tokenizer import BBPETokenizer

    tok = BBPETokenizer.load(args.tokenizer_path)
    megatron_tok = MegatronBBPETokenizer(tok)
    dataset = build_train_dataset(config, args.data, megatron_tok, m["seq_length"])
    # 分布式采样: 每个 rank 拿不同 batch（GPTDataset 按 idx 返回样本，
    # DistributedSampler 保证各 rank 数据不重叠）
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=t["micro_batch_size"], sampler=sampler)

    # 5. 优化器 + LR 调度
    # LR 调度用官方 OptimizerParamScheduler（WSD/cosine 原生支持，state_dict 可持久化）。
    # 优化器不用 get_megatron_optimizer：它要求 mcore 内部 Float16Module/ModuleChunk
    # 包装（参数须带 main_grad），裸 GPTModel 不满足；这里用 torch AdamW + fp32 master
    # （与手写轨同语义）。生产多卡请切官方 pretrain_gpt.py 训练栈。
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    model.float()  # fp32 master 参数；计算走 autocast bf16
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["lr"], betas=(0.9, 0.95), weight_decay=t.get("weight_decay", 0.1)
    )

    train_iters = t["train_iters"]
    bf16 = bool(t.get("bf16", True))
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    )
    # ratio → step 换算（对齐 configs/{variant}.yaml 的 lr 段；显式 iters 优先）
    warmup_steps = t.get("lr_warmup_iters") or int(
        round(t.get("lr_warmup_ratio", 0.02) * train_iters)
    )
    decay_style = str(t.get("lr_decay_style", "cosine")).upper()
    max_lr = t["lr"]
    min_lr = t.get("min_lr", max_lr * t.get("min_lr_ratio", 0.1))
    wd = t.get("weight_decay", 0.1)
    sched_kwargs = dict(
        init_lr=min_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=train_iters,
        lr_decay_style=decay_style,
        start_wd=wd,
        end_wd=wd,
        wd_incr_steps=train_iters,
        wd_incr_style="constant",
    )
    if decay_style == "WSD":
        sched_kwargs.update(
            wsd_decay_steps=t.get("wsd_decay_steps")
            or int(round((1.0 - t.get("stable_ratio", 0.8)) * train_iters)),
            lr_wsd_decay_style=str(t.get("lr_wsd_decay_style", "cosine")),
        )
    scheduler = OptimizerParamScheduler(optimizer, **sched_kwargs)

    # wandb（与手写轨一致: 装了就默认记录；--wandb_project/--wandb_run_name 可覆盖）
    run_name = args.wandb_run_name or (
        f"pretrain_{m['num_layers']}l_{m['hidden_size']}d_seq{m['seq_length']}"
    )
    if wandb is not None and dist.get_rank() == 0:
        wandb.init(project=args.wandb_project or "gleamlm", name=run_name, config=cfg)

    # ── 6. 训练循环：梯度累积 + 断点续训 ─────────────────────────────
    # step 以 optimizer step 计；consume 以 micro-batch 计。
    # 恢复时按 consumed_micro 定位 epoch/skip，set_epoch 确定性重放数据。
    accumulate_grad = max(1, int(t.get("accumulate_grad", 1)))
    micros_per_epoch = max(len(dataloader), 1)
    clip_grad = t.get("clip_grad", 1.0)
    save_interval = t.get("save_interval", 0)
    z_loss_weight = float(t.get("z_loss_weight", 0.0))

    # MFU 估算（近似；4070 Ti bf16 稠密算力 ~165 TFLOPS）
    num_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = int(t["micro_batch_size"]) * int(m["seq_length"]) * accumulate_grad
    peak_flops = 165e12

    step, start_epoch = 0, 0
    consumed_micro = 0
    total_loss = 0.0
    # resume/保存仅支持 TP=PP=1（数据并行规模）: 该脚本为单机学习轨；
    # TP/PP 切分下模型分片、须用官方 dist_checkpointing
    if args.load:
        if not os.path.exists(args.load):
            raise SystemExit(f"ERROR: --load 路径不存在: {args.load}")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        ck = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])  # state 已与参数同设备
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        step = ck["step"]
        consumed_micro = ck.get("consumed_micro", step * accumulate_grad)
        total_loss = ck.get("total_loss", 0.0)  # 跨会话累计 loss 和，日志平均才连续
        start_epoch = consumed_micro // micros_per_epoch
        skip_first = consumed_micro % micros_per_epoch
        if dist.get_rank() == 0:
            print(
                f"Resumed step {step} (consumed_micro={consumed_micro}, "
                f"epoch={start_epoch}, skip={skip_first})"
            )
    else:
        skip_first = 0

    os.makedirs(args.out, exist_ok=True)
    epoch = start_epoch
    is_first_epoch = True
    _t_last = [time.monotonic()]  # 容器便于 _finish_step 闭包修改

    def _save(path: str):
        if dist.get_rank() != 0:
            return
        sd = model.state_dict()  # 同一对象挂双键，pickle memo 只落盘一份
        torch.save(
            {
                "model_state_dict": sd,
                "model": sd,  # 兼容既有 deploy/megatron_to_hf 旧键
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "consumed_micro": consumed_micro,
                "total_loss": total_loss,
                "config": cfg,
            },
            path,
        )
        print(f"Saved: {path}")

    def _finish_step(acc_n: int) -> None:
        """收敛一个累积组的梯度并执行一次 optimizer step。"""
        nonlocal step, total_loss

        # NaN/Inf 防护：任一 rank loss 异常则全员退出（避免挂死/污染 ckpt）。
        # 用 SUM 标志传播（NCCL 对 NaN 走 MAX 无标准保证），bad=1/0 求和>0 即中止
        bad = acc_loss != acc_loss or abs(acc_loss) == float("inf")
        if dist.get_world_size() > 1:
            bad_t = torch.tensor(1.0 if bad else 0.0, device="cuda")
            dist.all_reduce(bad_t, op=dist.ReduceOp.SUM)
            bad = bad_t.item() > 0.5
        if bad:
            if dist.get_rank() == 0:
                print(f"[ABORT] NaN/Inf loss at step {step}, acc_loss={acc_loss}")
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
            raise SystemExit(1)
        if acc_n > 1:
            # 梯度按实际 micro 数平均（等价手写轨 loss/denom），残差批同规则
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(acc_n)
        reduce_dp_grads(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step(1)  # 官方顺序: step 后再推进 lr（下一 step 生效）
        step += 1
        total_loss += acc_loss / max(acc_n, 1)
        if step % t["log_interval"] == 0 and dist.get_rank() == 0:
            now = time.perf_counter()
            dt = now - _t_last[0]
            steps_in_window = max(t["log_interval"], 1)
            window_tokens = steps_in_window * tokens_per_step  # dt 跨多个 step
            tok_s = window_tokens / max(dt, 1e-9)
            mfu = 6.0 * num_params * window_tokens / max(dt, 1e-9) / peak_flops * 100.0
            _t_last[0] = now
            mem_gb = torch.cuda.memory_allocated() / 1e9
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:6d} | loss {total_loss / max(step, 1):.4f} "
                f"| lr {lr_now:.2e} "
                f"| {tok_s/1e3:.0f}k tok/s | MFU {mfu:.1f}% | mem {mem_gb:.1f}G"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "loss": total_loss / max(step, 1),
                        "lr": lr_now,
                        "tok_per_sec": tok_s,
                        "mfu": mfu,
                        "mem_gb": mem_gb,
                        "step": step,
                    },
                    step=step,
                )
        if save_interval > 0 and step % save_interval == 0:
            _save(os.path.join(args.out, f"iter_{step:07d}.pt"))

    # 第一次优化器 step 前恢复 epoch 内已消耗的 micro；随后逐 epoch 续跑
    while step < train_iters:
        sampler.set_epoch(epoch)
        batch_iter = enumerate(iter(dataloader))
        if is_first_epoch and skip_first > 0:
            batch_iter = itertools.islice(batch_iter, skip_first, None)
        is_first_epoch = False

        optimizer.zero_grad()
        acc_loss, acc_n = 0.0, 0
        for micro_i, batch in batch_iter:
            with autocast_ctx:
                loss = forward_backward(model, batch, z_loss_weight=z_loss_weight)
            acc_loss += loss.item()
            acc_n += 1
            consumed_micro += 1
            if acc_n < accumulate_grad:
                # 未凑满一组：只累积，不 step（残差批由 epoch 末尾收尾）
                continue
            _finish_step(acc_n)
            acc_loss, acc_n = 0.0, 0
            if step >= train_iters:
                break
            optimizer.zero_grad()

        if step >= train_iters:
            break
        # 一个 epoch 跑完但剩余不足一组：强制收尾一次 step
        if acc_n > 0:
            _finish_step(acc_n)
            optimizer.zero_grad()
            acc_loss, acc_n = 0.0, 0
        epoch += 1

    # 7. 可选验证: --eval-data (.bin/.idx 前缀) 训练结束后评估 val loss/ppl
    if args.eval_data and dist.get_rank() == 0:
        _run_eval(model, args.eval_data, megatron_tok, m["seq_length"], config,
                  max_batches=t.get("eval_max_batches", 200), autocast_ctx=autocast_ctx)

    # 8. final checkpoint
    _save(os.path.join(args.out, "megatron_final.pt"))
    if wandb is not None and dist.get_rank() == 0:
        wandb.finish()

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
