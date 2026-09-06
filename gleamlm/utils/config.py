"""YAML config loader with inheritance and validation.

两套配置体系并存（对应不同消费方）:
  - load_config → _DictWrapper（属性链访问，manual/train.py, sft.py, dpo.py 用）
  - load_config_v2 → GleamLMConfig（Pydantic v2，manual/pretrain.py, deepspeed.py, fsdp.py 用）

Pydantic 体系移植自历史版本 core/config.py（旧版实现，源目录已删除），
提供字段校验、默认值、model_dump()（checkpoint 配置快照）等能力。
"""

from __future__ import annotations

import argparse
import os
from importlib.resources import files
from typing import Any, Literal

import yaml

from gleamlm.types import ConfigValidationError

try:
    from pydantic import BaseModel, Field, field_validator

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

DEFAULT_TOKENIZER_PATH = str(files("gleamlm") / "tokenizer" / "checkpoints" / "bbpe_12k")


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class _DictWrapper:
    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        if name not in self._data:
            raise AttributeError(f"'{type(self).__name__}' has no key '{name}'")
        v = self._data[name]
        if isinstance(v, dict):
            return _DictWrapper(v)
        return v

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        v = self._data.get(key, default)
        if isinstance(v, dict):
            return _DictWrapper(v)
        return v

    def to_dict(self) -> dict[str, Any]:
        return self._data.copy()  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_yaml(path: str) -> dict[str, Any]:
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if "extends" in data:
        parent_path = os.path.join(base_dir, data.pop("extends"))
        parent = load_yaml(parent_path)
        data = _deep_merge(parent, data)

    return data


def resolve_relative_path(base_root: str, path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_root, path))


_CONFIG_VALIDATORS: dict[str, dict[str, tuple[type, Any]]] = {
    "model": {
        "d_model": (int, lambda v: v >= 64),
        "num_layers": (int, lambda v: 1 <= v <= 256),
        "num_heads": (int, lambda v: v >= 1),
        "num_kv_heads": (int, lambda v: v >= 1),
        "d_ff": (int, lambda v: v >= 64),
        "max_seq_len": (int, lambda v: v >= 32),
        "vocab_size": (int, lambda v: v >= 256),
        "dropout": (float, lambda v: 0.0 <= v <= 1.0),
        "tie_weights": (bool, None),
        "use_flash_attn": (bool, None),
        "use_gradient_checkpointing": (bool, None),
    },
    "training": {
        "epochs": (int, lambda v: 1 <= v <= 1000),
        "batch_size": (int, lambda v: v >= 1),
        "accumulate_grad": (int, lambda v: v >= 1),
        "clip_grad": (float, lambda v: v >= 0),
        "weight_decay": (float, None),
        "seed": (int, None),
        "max_train_chars": (int, lambda v: v >= 0),
    },
    "lr": {
        "lr": (float, lambda v: v > 0),
        "warmup_ratio": (float, lambda v: 0.0 <= v <= 1.0),
        "min_lr_ratio": (float, lambda v: 0.0 <= v <= 1.0),
    },
    "advanced": {
        "z_loss_weight": (float, lambda v: v >= 0),
        "bf16": (bool, None),
    },
    "data": {
        "data_dir": (str, lambda v: len(v) > 0),
        "tokenizer_path": (str, lambda v: len(v) > 0),
        "checkpoint_dir": (str, lambda v: len(v) > 0),
    },
}


def _validate_config(cfg_dict: dict[str, Any]) -> None:
    errors: list[str] = []
    for section, fields in _CONFIG_VALIDATORS.items():
        sec = cfg_dict.get(section, {})
        if not isinstance(sec, dict):
            continue
        for key, (expected_type, validator) in fields.items():
            if key not in sec:
                continue
            val = sec[key]
            if not isinstance(val, expected_type):
                errors.append(
                    f"{section}.{key}: 期望 {expected_type.__name__}, 实际 {type(val).__name__}({val!r})"
                )
                continue
            if validator is not None and not validator(val):
                errors.append(f"{section}.{key}: 值 {val!r} 超出有效范围")
    if errors:
        raise ConfigValidationError("配置校验失败:\n" + "\n".join(f"  {e}" for e in errors))


def extract_checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """从 checkpoint 提取模型结构配置（唯一格式: `_config` 纯 dict 快照）。

    保存端 (pretrain/opd/ppo/grpo/sft/dpo/distill/sft_lora) 统一写 `_config`，
    其字段由 ModelConfig dump（pretrain）或构建参数（sft/dpo）直接生成，
    无手写默认值可猜错。历史格式（args Namespace / config dict）已在格式
    统一时移除，不再兼容。
    """
    raw = checkpoint.get("_config")
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            "checkpoint 缺少 '_config' 字段，无法重建模型结构。"
            "若来自 mcore 轨 (industrial/pretrain)，请先运行 "
            "deploy/megatron_to_hf.py 转换为 HF 目录后再加载"
        )
    return raw


def load_config(config_file: str) -> _DictWrapper:
    cfg_dict = load_yaml(config_file)
    _validate_config(cfg_dict)
    return _DictWrapper(cfg_dict)


def cfg_to_namespace(cfg: _DictWrapper, root_dir: str) -> argparse.Namespace:
    c = cfg
    return argparse.Namespace(
        d_model=c.model.d_model,
        num_layers=c.model.num_layers,
        num_heads=c.model.num_heads,
        num_kv_heads=c.model.num_kv_heads,
        d_ff=c.model.d_ff,
        max_seq_len=c.model.max_seq_len,
        vocab_size=c.model.vocab_size,
        dropout=c.model.dropout,
        tie_weights=c.model.tie_weights,
        use_flash_attn=getattr(c.model, "use_flash_attn", False),
        use_gradient_checkpointing=getattr(c.model, "use_gradient_checkpointing", False),
        seed=c.training.seed,
        epochs=c.training.epochs,
        batch_size=c.training.batch_size,
        accumulate_grad=c.training.accumulate_grad,
        clip_grad=c.training.clip_grad,
        weight_decay=c.training.weight_decay,
        label_smoothing=c.training.label_smoothing,
        log_interval=c.training.log_interval,
        eval_interval=c.training.eval_interval,
        save_interval=c.training.save_interval,
        max_train_chars=c.training.max_train_chars,
        lr=c.lr.lr,
        type=c.lr.type,
        warmup_ratio=c.lr.warmup_ratio,
        min_lr_ratio=c.lr.min_lr_ratio,
        stable_ratio=getattr(c.lr, "stable_ratio", 0.0),
        data_dir=resolve_relative_path(root_dir, c.data.data_dir),
        tokenizer_path=resolve_relative_path(root_dir, c.data.tokenizer_path),
        checkpoint_dir=resolve_relative_path(root_dir, c.data.checkpoint_dir),
        ids_prefix=c.data.ids_prefix,
        load_checkpoint=getattr(c.data, "load_checkpoint", None),
        z_loss_weight=c.advanced.z_loss_weight,
        bf16=c.advanced.bf16,
        pin_memory=c.advanced.pin_memory,
        num_workers=c.advanced.num_workers,
        optimizer_type=c.optimizer.type,
        optimizer_betas=c.optimizer.betas,
        optimizer_eps=c.optimizer.eps,
        sft_epochs=c.sft.epochs,
        sft_batch_size=c.sft.batch_size,
        sft_accumulate_grad=c.sft.accumulate_grad,
        sft_lr=c.sft.lr,
        sft_warmup_ratio=c.sft.warmup_ratio,
        sft_weight_decay=c.sft.weight_decay,
        sft_max_seq_len=c.sft.max_seq_len,
        sft_data_path=c.sft.data_path,
        sft_inject_system_ratio=getattr(c.sft, "inject_system_ratio", 0.2),
        dpo_epochs=c.dpo.epochs,
        dpo_batch_size=c.dpo.batch_size,
        dpo_accumulate_grad=c.dpo.accumulate_grad,
        dpo_lr=c.dpo.lr,
        dpo_beta=c.dpo.beta,
        dpo_max_seq_len=c.dpo.max_seq_len,
        dpo_warmup_ratio=getattr(c.dpo, "warmup_ratio", 0.02),
        dpo_min_lr_ratio=getattr(c.dpo, "min_lr_ratio", 0.05),
        dpo_data_path=c.dpo.data_path,
        distributed_backend=c.distributed.backend,
    )


# ── load_config_v2 消费方必读字段 ─────────────────────────────────────
# manual/pretrain.py (_load_training_defaults) 与 manual/train_tokenizer.py
# 消费的 YAML 键全集。YAML 缺键时直接报错, 禁止静默回退 Pydantic 默认 ——
# 双轨默认值曾多次漂移 (dpo.lr 1e-7 / sft.lr 5e-6 / stable_ratio 0.0 等
# 历史坑), 一致性由 tests/test_config_defaults.py 锁死。
# data_sources 允许空列表 (键存在即可, train_tokenizer.py 自带防空)。
_REQUIRED_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "training": (
        "epochs", "batch_size", "accumulate_grad", "weight_decay",
        "clip_grad", "log_interval", "save_interval", "seed",
        "label_smoothing",
    ),
    "lr": ("type", "lr", "warmup_ratio", "stable_ratio", "min_lr_ratio"),
    "advanced": ("z_loss_weight", "num_workers"),
    "data": ("tokenizer_path", "data_dir", "checkpoint_dir"),
}


def validate_required_config_fields(data: dict[str, Any]) -> None:
    """校验训练消费方必读字段齐全; 缺失即报错, 不静默回退默认值。"""
    missing: list[str] = []
    for section, keys in _REQUIRED_CONFIG_SECTIONS.items():
        sec = data.get(section)
        if not isinstance(sec, dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in sec:
                missing.append(f"{section}.{key}")
    if "data_sources" not in data:
        missing.append("data_sources")
    if missing:
        raise ConfigValidationError(
            "配置缺少训练消费方必读字段 (参考 configs/base.yaml 补全): "
            + ", ".join(missing)
        )


# Pydantic 配置体系（v2），手写轨 pretrain/deepspeed/fsdp 使用；
# 移植自历史版本 core/config.py（旧版已删除），load_config_v2 返回 GleamLMConfig。

if _HAS_PYDANTIC:

    class LrConfig(BaseModel):
        # 默认值 = base.yaml 实证 (WSD 4e-4 → 4e-5, stable 80%)。
        # 历史坑: 曾回落 cosine/3e-4/stable 0.0 与实证漂移; 一致性由
        # tests/test_config_defaults.py 锁死
        type: Literal["wsd", "cosine"] = "wsd"
        lr: float = 0.0004
        warmup_ratio: float = 0.02
        min_lr_ratio: float = 0.1
        stable_ratio: float = 0.8

        @field_validator("lr")
        @classmethod
        def lr_positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError(f"lr must be positive, got {v}")
            return v

    class OptimizerConfig(BaseModel):
        type: str = "adamw"
        betas: tuple[float, float] = (0.9, 0.95)
        eps: float = 1e-8

    class ModelConfig(BaseModel):
        """模型架构配置 — 手写轨 --model YAML 入口（pretrain/deepspeed/fsdp）。"""

        vocab_size: int = 12002
        d_model: int = 512
        num_layers: int = 12
        num_heads: int = 8
        num_kv_heads: int = 4
        d_ff: int = 1365
        max_seq_len: int = 1024
        dropout: float = 0.1
        tie_weights: bool = True
        use_flash_attn: bool = False
        use_gradient_checkpointing: bool = False
        attn_type: str = "gqa"
        ffn_type: str = "mlp"
        num_experts: int = 8
        top_k: int = 2
        # YaRN
        rope_scale: float = 1.0
        rope_factor: float = 8.0
        rope_theta: float = 10000.0
        # 按层混合变体（可选，None 时所有层使用 attn_type/ffn_type 全局值）
        layer_configs: list[dict] | None = None

        @field_validator("d_model")
        @classmethod
        def d_model_valid(cls, v: int) -> int:
            if v < 64:
                raise ValueError(f"d_model must be >= 64, got {v}")
            return v

        @field_validator("num_layers")
        @classmethod
        def num_layers_valid(cls, v: int) -> int:
            if not 1 <= v <= 256:
                raise ValueError(f"num_layers must be 1-256, got {v}")
            return v

        @classmethod
        def from_yaml(cls, path: str) -> "ModelConfig":
            """从 YAML 加载（经 load_yaml 处理 extends 继承），兼容两种结构:

            - 纯 model 字段:  {d_model: 512, num_layers: 12, ...}
            - 完整 GleamLMConfig 结构: {training: ..., model: {...}, ...}（取 model 段）
            """
            data = load_yaml(path)
            if isinstance(data, dict) and isinstance(data.get("model"), dict):
                data = data["model"]
            return cls(**data)

    class TrainingConfig(BaseModel):
        # 默认值 = base.yaml training 段实证 (epochs=1, 6.13B chars 预算)
        seed: int = 42
        epochs: int = 1
        batch_size: int = 8
        accumulate_grad: int = 8
        clip_grad: float = 1.0
        weight_decay: float = 0.01
        label_smoothing: float = 0.1
        log_interval: int = 50
        eval_interval: int = 500
        save_interval: int = 2000
        max_train_chars: int = 6130000000

    class DataConfig(BaseModel):
        data_dir: str = ""
        tokenizer_path: str = ""
        checkpoint_dir: str = ""
        ids_prefix: str = ""
        load_checkpoint: str | None = None

    class AdvancedConfig(BaseModel):
        # nano 实证 z-loss 系数 1e-4（防 logits 爆炸）；0 禁用。
        # 历史坑: 曾沿用 SmolLM3 的 1e-5, 与 base.yaml 漂移 10×
        z_loss_weight: float = 1e-4
        bf16: bool = True
        pin_memory: bool = True
        num_workers: int = 0

    class SFTConfig(BaseModel):
        epochs: int = 3
        batch_size: int = 8
        accumulate_grad: int = 4
        # 历史坑: 曾误配 5e-6 (=1e-4×min_lr 0.05, 余弦终点值非起点), 与 base.yaml 实证 1e-4 对齐
        lr: float = 1e-4
        warmup_ratio: float = 0.02
        weight_decay: float = 0.01
        max_seq_len: int = 1024
        inject_system_ratio: float = 0.2
        data_path: str = ""

    class DPOConfig(BaseModel):
        epochs: int = 1
        batch_size: int = 2
        accumulate_grad: int = 2
        # 历史坑: 曾误配 1e-7 (比 base.yaml 实证低 10×)
        lr: float = 1e-6
        beta: float = 0.1
        max_seq_len: int = 1024
        warmup_ratio: float = 0.02
        min_lr_ratio: float = 0.05
        data_path: str = ""

    class DistributedConfig(BaseModel):
        backend: str = "auto"

    class DataSource(BaseModel):
        name: str
        type: str
        ratio: float
        file: str | None = None

    class GleamLMConfig(BaseModel):
        """完整训练配置 — load_config_v2 的返回类型。"""

        training: TrainingConfig = Field(default_factory=TrainingConfig)
        lr: LrConfig = Field(default_factory=LrConfig)
        optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
        model: ModelConfig = Field(default_factory=ModelConfig)
        data: DataConfig = Field(default_factory=DataConfig)
        data_sources: list[DataSource] = []
        advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
        sft: SFTConfig = Field(default_factory=SFTConfig)
        dpo: DPOConfig = Field(default_factory=DPOConfig)
        distributed: DistributedConfig = Field(default_factory=DistributedConfig)

        @classmethod
        def from_yaml(cls, path: str) -> "GleamLMConfig":
            data = load_yaml(path)
            validate_required_config_fields(data)
            return cls(**data)

        def resolve_paths(self, root_dir: str) -> "GleamLMConfig":
            for field_name in ("data_dir", "tokenizer_path", "checkpoint_dir"):
                raw = getattr(self.data, field_name)
                if raw and not os.path.isabs(raw):
                    setattr(
                        self.data, field_name, os.path.normpath(os.path.join(root_dir, raw))
                    )
            if self.data.load_checkpoint and not os.path.isabs(self.data.load_checkpoint):
                self.data.load_checkpoint = os.path.normpath(
                    os.path.join(root_dir, self.data.load_checkpoint)
                )
            if self.sft.data_path and not os.path.isabs(self.sft.data_path):
                self.sft.data_path = os.path.normpath(os.path.join(root_dir, self.sft.data_path))
            if self.dpo.data_path and not os.path.isabs(self.dpo.data_path):
                self.dpo.data_path = os.path.normpath(os.path.join(root_dir, self.dpo.data_path))
            return self


def load_config_v2(config_file: str, root_dir: str = "") -> "GleamLMConfig":
    """Pydantic 版配置加载 — extends 继承 + 可选相对路径解析（旧版同款）。"""
    if not _HAS_PYDANTIC:
        raise ImportError(
            "load_config_v2 需要 pydantic>=2.0。请执行: pip install pydantic"
        )
    cfg = GleamLMConfig.from_yaml(config_file)
    if root_dir:
        cfg.resolve_paths(root_dir)
    return cfg
