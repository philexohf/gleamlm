"""工业轨 pretrain 配置校验 — Megatron 键名体系的轻量必读护栏。

与手动轨 gleamlm/utils/config.py 的关系: 两套体系键名不同。手动轨是
Pydantic 单轨 + scope 必读校验; 工业轨是 Megatron 语义自包含 YAML
(model/parallel/training 三段), 这里只做消费端必读键校验, 不做建模。

背景: industrial/pretrain.py 消费端曾用 .get(key, 默认值) 静默兜底
(lr_decay_style 默认 cosine / z_loss_weight 默认 0 / save_interval 默认 0),
yaml 忘写会静默跑错调度或丢 z-loss/checkpoint —— 与手动轨历史漂移
(dpo.lr 1e-7、stable_ratio 0.0) 同型。现改为 load 即校验: 缺失列清单
报错, 不再静默回退。注意本模块为纯 stdlib 轻量模块 (无 megatron/torch
依赖), 供 pretrain.py 与 tests 双侧复用。
"""

import yaml

# 消费端必读键: 现网两份配置 (industrial/configs/{nano,0.6b}.yaml) 均显式给出;
# 缺失会导致静默错值/断对齐, 一律要求显式, 不提供代码默认
_IND_REQUIRED = {
    "model": (
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "ffn_hidden_size",
        "vocab_size",
        "max_position_embeddings",
        "seq_length",
    ),
    "parallel": (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    ),
    "training": (
        "micro_batch_size",
        "accumulate_grad",
        "train_iters",
        "lr",
        "log_interval",
        "weight_decay",
        "clip_grad",
        "save_interval",
        "z_loss_weight",
        "bf16",
        "lr_decay_style",
    ),
}

# 推导二选一: (a, b) 至少给一个 (a 缺省时按 b 推导, 两者全缺才报错)
_IND_ALTERNATIVES = {
    "lr_warmup_iters|lr_warmup_ratio": ("lr_warmup_iters", "lr_warmup_ratio"),
    "min_lr|min_lr_ratio": ("min_lr", "min_lr_ratio"),
}


def validate_industrial_config(data: dict) -> None:
    """校验工业轨配置的消费端必读键; 缺失即报错, 不静默回退推导默认。"""
    missing: list[str] = []
    for section, keys in _IND_REQUIRED.items():
        sec = data.get(section)
        if not isinstance(sec, dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in sec:
                missing.append(f"{section}.{key}")
    training = data.get("training")
    if not isinstance(training, dict):
        return  # training 段缺失已在上面计入 missing
    for label, (a, b) in _IND_ALTERNATIVES.items():
        if a not in training and b not in training:
            missing.append(f"training.{label} (至少其一)")
    if str(training.get("lr_decay_style", "")).lower() == "wsd":
        # WSD: decay 步数换算依赖 stable_ratio; 衰减形状是方案属性 (nano 实证 linear)
        if "wsd_decay_steps" not in training and "stable_ratio" not in training:
            missing.append("training.wsd_decay_steps|training.stable_ratio (wsd 至少其一)")
        if "lr_wsd_decay_style" not in training:
            missing.append("training.lr_wsd_decay_style (wsd 必配)")
    if missing:
        raise ValueError(
            "工业轨配置缺少消费端必读字段 (参考 industrial/configs/nano.yaml 或 "
            f"industrial/configs/0.6b.yaml 补全): {', '.join(missing)}"
        )


def load_config(path: str) -> dict:
    """读取工业轨 YAML 并做必读校验 (load 即校验, 防静默回退)。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    validate_industrial_config(data)
    return data
