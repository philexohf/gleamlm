"""GleamLM YAML + CLI 配置加载器"""

from __future__ import annotations

import argparse
import contextlib
import os
from importlib.resources import files
from typing import Any

import yaml

DEFAULT_TOKENIZER_PATH = str(files("gleamlm") / "tokenizer" / "checkpoints" / "bbpe_12k")

_NO_PREFIX_SECTIONS: set[str] = {"model", "training", "data", "advanced", "lr"}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个 dict，override 优先。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class _DictWrapper:
    """将嵌套 dict 包装为属性访问风格。cfg.training.epochs"""

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
    """递归加载 YAML 文件，自动解析 extends 继承。"""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "extends" in data:
        parent_path = os.path.join(base_dir, data.pop("extends"))
        parent = load_yaml(parent_path)
        data = _deep_merge(parent, data)

    return data


def _resolve_paths(cfg_dict: dict, model_name: str) -> None:
    """自动补全 data 块下的空路径 (基于 model_name 变体)。"""
    d = cfg_dict.setdefault("data", {})
    if not d.get("data_dir"):
        d["data_dir"] = os.path.join(os.getcwd(), "data", model_name, "pretrain")
    if not d.get("tokenizer_path"):
        d["tokenizer_path"] = DEFAULT_TOKENIZER_PATH
    if not d.get("checkpoint_dir"):
        d["checkpoint_dir"] = os.path.join(os.getcwd(), "checkpoints", model_name)


def _parse_cli_overrides(cfg_dict: dict) -> dict:
    """将 sys.argv 解析为命令行覆盖。支持 --training.epochs 10 和 --epochs 10"""
    parser = argparse.ArgumentParser(add_help=False)

    NO_PREFIX: set[str] = _NO_PREFIX_SECTIONS

    def flatten(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, full_key)
            else:
                arg_type = str if isinstance(v, bool) else (type(v) if v is not None else str)
                parser.add_argument(f"--{full_key}", type=arg_type, default=None)
                if prefix in NO_PREFIX:
                    try:
                        parser.add_argument(f"--{k}", type=arg_type, default=None, dest=full_key)
                    except argparse.ArgumentError:
                        pass

    flatten(cfg_dict)

    known, _ = parser.parse_known_args()
    overrides = {k: v for k, v in vars(known).items() if v is not None}
    return overrides


def _apply_overrides(cfg_dict: dict, overrides: dict) -> None:
    """将扁平 override dict 写入嵌套 cfg_dict。"""
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg_dict
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        if isinstance(value, str):
            current = d.get(parts[-1])
            if isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                with contextlib.suppress(ValueError):
                    value = int(value)
            elif isinstance(current, float):
                with contextlib.suppress(ValueError):
                    value = float(value)
        d[parts[-1]] = value


# 配置校验规则
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
    """校验加载后的配置，类型和值范围检查，不符合则抛出 ValueError。"""
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


def load_config(
    config_file: str, model_name: str | None = None, cli_overrides: bool = False
) -> _DictWrapper:
    """加载配置文件，返回 _DictWrapper 属性访问对象。"""
    cfg_dict = load_yaml(config_file)

    if model_name is None:
        basename = os.path.splitext(os.path.basename(config_file))[0]
        model_name = basename

    _resolve_paths(cfg_dict, model_name)
    _validate_config(cfg_dict)

    if cli_overrides:
        overrides = _parse_cli_overrides(cfg_dict)
        _apply_overrides(cfg_dict, overrides)
        _validate_config(cfg_dict)

    return _DictWrapper(cfg_dict)


def to_namespace(cfg: _DictWrapper) -> argparse.Namespace:
    """将 DictWrapper 配置转为 argparse.Namespace。"""
    NO_PREFIX: set[str] = _NO_PREFIX_SECTIONS

    def _flatten_prefix(d: dict, prefix: str) -> dict:
        result: dict = {}
        for k, v in d.items():
            full = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten_prefix(v, full))
            else:
                result[full] = v
        return result

    result: dict = {}
    import warnings

    for section_name, section_data in cfg._data.items():
        if isinstance(section_data, dict):
            if section_name in NO_PREFIX:
                for k, v in section_data.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            result.setdefault(k2, v2)
                    else:
                        result.setdefault(k, v)
            else:
                prefixed = _flatten_prefix(section_data, section_name)
                for k in prefixed:
                    if k in result:
                        warnings.warn(
                            f"[to_namespace] 键冲突: '{k}' 从 prefixed section 覆盖已存在的键",
                            stacklevel=2,
                        )
                result.update(prefixed)
        else:
            result[section_name] = section_data

    return argparse.Namespace(**result)


def load_config_as_args(
    config_file: str, model_name: str | None = None, cli_overrides: bool = False
) -> argparse.Namespace:
    """一站式加载 YAML 配置并转为 argparse.Namespace"""
    cfg = load_config(config_file, model_name, cli_overrides)
    return to_namespace(cfg)
