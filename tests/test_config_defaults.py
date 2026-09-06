"""配置双轨一致性测试 — base.yaml 与 Pydantic 默认值必须一致 (防漂移)。

背景: gleamlm/utils/config.py 的 Pydantic 默认值与 configs/base.yaml 实证值
曾多次漂移 (dpo.lr 1e-7 vs 1e-6、sft.lr 5e-6 vs 1e-4、lr.stable_ratio 0.0
vs 0.8、epochs 4 vs 1 等), YAML 缺键时会静默回退错误的旧默认值。
本文件锁死两边一致 —— 任何一边被单独改动都会让用例变红, 必须显式决定
"哪边为真"再同步另一边。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gleamlm.types import ConfigValidationError
from gleamlm.utils.config import (
    GleamLMConfig,
    load_config_v2,
    load_yaml,
    validate_required_config_fields,
)

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# load_config_v2 消费方 (manual/pretrain.py _load_training_defaults +
# manual/train_tokenizer.py) 必读字段。故意与 config.py 的
# _REQUIRED_CONFIG_SECTIONS 重复: 任一侧清单被误删, 测试都会先暴露。
_CONSUMED = {
    "training": (
        "epochs", "batch_size", "accumulate_grad", "weight_decay",
        "clip_grad", "log_interval", "save_interval", "seed",
        "label_smoothing",
    ),
    "lr": ("type", "lr", "warmup_ratio", "stable_ratio", "min_lr_ratio"),
    "advanced": ("z_loss_weight", "num_workers"),
    "data": ("tokenizer_path", "data_dir", "checkpoint_dir"),
}


def test_pydantic_defaults_match_base_yaml() -> None:
    """base.yaml 显式定义的每个标量键, 必须与 GleamLMConfig 默认值一致。"""
    base = load_yaml(str(_CONFIGS / "base.yaml"))
    defaults = GleamLMConfig().model_dump()
    diffs: list[str] = []
    for section, fields in defaults.items():
        yaml_sec = base.get(section)
        if not isinstance(yaml_sec, dict):
            continue
        for key, default in fields.items():
            if key not in yaml_sec or default is None:
                continue
            if isinstance(default, dict | list):
                continue  # 嵌套结构 (layer_configs 等) 不在本测试范围
            if isinstance(default, tuple):
                default = list(default)  # pydantic tuple vs yaml list 语义等价
            if yaml_sec[key] != default:
                diffs.append(
                    f"base.yaml {section}.{key}={yaml_sec[key]!r} "
                    f"!= Pydantic 默认 {default!r}"
                )
    assert not diffs, (
        "配置双轨漂移 (先确定哪边为真, 再同步另一边):\n" + "\n".join(diffs)
    )


@pytest.mark.parametrize("name", ["base", "nano", "lite", "pro"])
def test_consumed_fields_present_in_variant(name: str) -> None:
    """消费方必读字段在每个变体 (extends 合并后) 都必须齐全。"""
    data = load_yaml(str(_CONFIGS / f"{name}.yaml"))
    missing: list[str] = []
    for section, keys in _CONSUMED.items():
        sec = data.get(section)
        if not isinstance(sec, dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in sec:
                missing.append(f"{section}.{key}")
    if "data_sources" not in data:
        missing.append("data_sources")
    assert not missing, f"{name}.yaml 缺少消费方必读字段: {', '.join(missing)}"


def test_missing_field_raises_config_error() -> None:
    """缺必读字段必须显式报错 (禁止静默回退 Pydantic 默认)。"""
    data = load_yaml(str(_CONFIGS / "nano.yaml"))
    del data["lr"]["lr"]
    with pytest.raises(ConfigValidationError, match="lr.lr"):
        validate_required_config_fields(data)


def test_all_variants_load_via_load_config_v2() -> None:
    """现有变体配置都能通过 load_config_v2 完整校验 (回归护栏)。"""
    for name in ("base", "nano", "lite", "pro"):
        cfg = load_config_v2(str(_CONFIGS / f"{name}.yaml"))
        assert cfg.model.d_model >= 64
