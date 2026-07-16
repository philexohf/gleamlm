"""GleamLM YAML 配置加载器"""

from __future__ import annotations

import argparse
import os
from importlib.resources import files
from typing import Any

import yaml

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

    def to_dict(self) -> dict:
        return self._data.copy()

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_yaml(path: str) -> dict:
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "extends" in data:
        parent_path = os.path.join(base_dir, data.pop("extends"))
        parent = load_yaml(parent_path)
        data = _deep_merge(parent, data)

    return data


def resolve_relative_path(base_root: str, path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_root, path))


_CONFIG_VALIDATORS = {
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
}


def _validate_config(cfg_dict: dict) -> None:
    errors = []
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
        raise ValueError("配置校验失败:\n" + "\n".join(f"  {e}" for e in errors))


def load_config(config_file: str) -> _DictWrapper:
    cfg_dict = load_yaml(config_file)
    _validate_config(cfg_dict)
    return _DictWrapper(cfg_dict)


def cfg_to_namespace(cfg: _DictWrapper, root_dir: str) -> argparse.Namespace:
    c = cfg
    return argparse.Namespace(
        # ── model ──
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
        use_qk_norm=getattr(c.model, "use_qk_norm", True),
        use_gradient_checkpointing=getattr(c.model, "use_gradient_checkpointing", False),
        # ── training ──
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
        # ── lr ──
        lr=c.lr.lr,
        type=c.lr.type,
        warmup_ratio=c.lr.warmup_ratio,
        min_lr_ratio=c.lr.min_lr_ratio,
        stable_ratio=getattr(c.lr, "stable_ratio", 0.0),
        # ── data (paths resolved) ──
        data_dir=resolve_relative_path(root_dir, c.data.data_dir),
        tokenizer_path=resolve_relative_path(root_dir, c.data.tokenizer_path),
        checkpoint_dir=resolve_relative_path(root_dir, c.data.checkpoint_dir),
        ids_prefix=c.data.ids_prefix,
        load_checkpoint=getattr(c.data, "load_checkpoint", None),
        # ── advanced ──
        z_loss_weight=c.advanced.z_loss_weight,
        bf16=c.advanced.bf16,
        pin_memory=c.advanced.pin_memory,
        num_workers=c.advanced.num_workers,
        # ── optimizer (prefixed) ──
        optimizer_type=c.optimizer.type,
        optimizer_betas=c.optimizer.betas,
        optimizer_eps=c.optimizer.eps,
        # ── sft (prefixed) ──
        sft_epochs=c.sft.epochs,
        sft_batch_size=c.sft.batch_size,
        sft_accumulate_grad=c.sft.accumulate_grad,
        sft_lr=c.sft.lr,
        sft_warmup_ratio=c.sft.warmup_ratio,
        sft_weight_decay=c.sft.weight_decay,
        sft_max_seq_len=c.sft.max_seq_len,
        sft_data_path=c.sft.data_path,
        sft_inject_system_ratio=getattr(c.sft, "inject_system_ratio", 0.2),
        # ── dpo (prefixed) ──
        dpo_epochs=c.dpo.epochs,
        dpo_batch_size=c.dpo.batch_size,
        dpo_accumulate_grad=c.dpo.accumulate_grad,
        dpo_lr=c.dpo.lr,
        dpo_beta=c.dpo.beta,
        dpo_max_seq_len=c.dpo.max_seq_len,
        dpo_warmup_ratio=getattr(c.dpo, "warmup_ratio", 0.02),
        dpo_min_lr_ratio=getattr(c.dpo, "min_lr_ratio", 0.05),
        dpo_data_path=c.dpo.data_path,
        # ── distributed (prefixed) ──
        distributed_backend=c.distributed.backend,
    )
