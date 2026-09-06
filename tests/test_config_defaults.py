"""配置默认值一致性测试 — base.yaml 与 Pydantic 默认值必须一致 (防漂移)。

背景: gleamlm/utils/config.py 的 Pydantic 默认值与 manual/configs/base.yaml 实证值
曾多次漂移 (dpo.lr 1e-7 vs 1e-6、sft.lr 5e-6 vs 1e-4、lr.stable_ratio 0.0
vs 0.8、epochs 4 vs 1 等), YAML 缺键时会静默回退错误的旧默认值。
配置体系单轨化 (2026-09, _DictWrapper 轨删除) 后, 必读键缺失由
validate_required_config_fields 显式报错; 必读清单按消费方拆分 (scope):
  - full      : 全量并集
  - training  : pretrain 训练默认值加载器
  - tokenizer : train_tokenizer (仅 data_sources)
  - sft / dpo : manual/{sft,dpo}.py
  - opd / lora: manual/{opd,sft_lora}.py
本文件继续锁死仍可能回退的默认值 —— 任何一边被单独改动都会让用例变红,
必须显式决定 "哪边为真"再同步另一边。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gleamlm.types import ConfigValidationError
from gleamlm.utils.config import (
    _SCOPE_REQUIRED,
    _SCOPE_TOP_KEYS,
    GleamLMConfig,
    load_config,
    load_yaml,
    validate_required_config_fields,
)

_CONFIGS = Path(__file__).resolve().parents[1] / "manual" / "configs"

# 测试侧维护的 full 必读清单，故意与 config.py _SCOPE_REQUIRED["full"] 重复:
# 任一侧清单被误删, 测试都会先暴露。
_CONSUMED = {
    "model": (
        "d_model",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "d_ff",
        "max_seq_len",
        "vocab_size",
        "dropout",
        "tie_weights",
        "use_flash_attn",
    ),
    "training": (
        "epochs",
        "batch_size",
        "accumulate_grad",
        "weight_decay",
        "clip_grad",
        "log_interval",
        "save_interval",
        "seed",
        "label_smoothing",
    ),
    "lr": ("type", "lr", "warmup_ratio", "stable_ratio", "min_lr_ratio", "wsd_decay_style"),
    "advanced": ("z_loss_weight", "num_workers"),
    "data": ("tokenizer_path", "data_dir", "checkpoint_dir"),
    "sft": (
        "epochs",
        "batch_size",
        "accumulate_grad",
        "lr",
        "lr_scheduler",
        "warmup_ratio",
        "stable_ratio",
        "min_lr_ratio",
        "weight_decay",
        "max_seq_len",
        "inject_system_ratio",
        "data_path",
    ),
    "dpo": (
        "epochs",
        "batch_size",
        "accumulate_grad",
        "lr",
        "beta",
        "max_seq_len",
        "warmup_ratio",
        "min_lr_ratio",
        "data_path",
    ),
    "opd": (
        "epochs",
        "batch_size",
        "n_samples",
        "lr",
        "weight_decay",
        "clip_grad",
        "max_seq_len",
        "max_new_tokens",
        "temperature",
        "entropy_coeff",
        "aux_coeff",
        "log_interval",
        "save_interval",
        "seed",
        "data_path",
    ),
    "lora": (
        "epochs",
        "batch_size",
        "lr",
        "clip_grad",
        "max_seq_len",
        "lora_r",
        "lora_alpha",
        "log_interval",
        "data_path",
    ),
}


def _scopes() -> list[str]:
    return sorted(_SCOPE_REQUIRED)


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
                    f"base.yaml {section}.{key}={yaml_sec[key]!r} != Pydantic 默认 {default!r}"
                )
    assert not diffs, "默认值漂移 (先确定哪边为真, 再同步另一边):\n" + "\n".join(diffs)


@pytest.mark.parametrize("name", ["base", "nano", "lite", "pro"])
def test_full_consumed_fields_present_in_variant(name: str) -> None:
    """full 必读字段在每个变体 (已独立展开, 不依赖 extends) 都必须齐全。"""
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
    assert not missing, f"{name}.yaml 缺少 full 必读字段: {', '.join(missing)}"


@pytest.mark.parametrize("scope", [s for s in _scopes() if s != "full"])
@pytest.mark.parametrize("name", ["base", "nano", "lite", "pro"])
def test_scope_fields_present_in_variant(scope: str, name: str) -> None:
    """每个 scope 的必读字段在变体中都必须齐全 (full ⊇ scope 关系护栏)。"""
    data = load_yaml(str(_CONFIGS / f"{name}.yaml"))
    missing: list[str] = []
    for section, keys in _SCOPE_REQUIRED[scope].items():
        sec = data.get(section)
        if not isinstance(sec, dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in sec:
                missing.append(f"{section}.{key}")
    for key in _SCOPE_TOP_KEYS[scope]:
        if key not in data:
            missing.append(key)
    assert not missing, f"{name}.yaml 缺少 scope={scope} 必读字段: {', '.join(missing)}"


def test_full_matches_required_sections() -> None:
    """测试侧 _CONSUMED 与运行侧 _SCOPE_REQUIRED["full"] 必须逐字一致。

    任一侧清单被单独改动（加字段/删字段/改键）都会让本用例变红 —— 消除
    清单漂移的第二来源：生产代码与测试各自维护一份必读字段清单。
    """
    assert _SCOPE_REQUIRED["full"] == _CONSUMED
    assert _SCOPE_TOP_KEYS["full"] == ("data_sources",)


@pytest.mark.parametrize("scope", _scopes())
def test_scope_is_subset_of_full(scope: str) -> None:
    """每个 scope 的段/键必须是 full 的子集, 防止 scope 出现 full 没有的键。"""
    if scope == "full":
        return
    for section, keys in _SCOPE_REQUIRED[scope].items():
        assert section in _SCOPE_REQUIRED["full"], f"scope={scope} 段 {section} 不在 full"
        for key in keys:
            assert key in _SCOPE_REQUIRED["full"][section], (
                f"scope={scope} {section}.{key} 不在 full"
            )
    for key in _SCOPE_TOP_KEYS[scope]:
        assert key in _SCOPE_TOP_KEYS["full"], f"scope={scope} 顶层键 {key} 不在 full"


def test_missing_field_raises_config_error() -> None:
    """缺必读字段必须显式报错 (禁止静默回退 Pydantic 默认)。"""
    data = load_yaml(str(_CONFIGS / "nano.yaml"))
    del data["lr"]["lr"]
    with pytest.raises(ConfigValidationError, match="lr.lr"):
        validate_required_config_fields(data)


def test_missing_scope_field_raises() -> None:
    """scope 内部缺键同样报错 (sft scope 缺 sft.lr)。"""
    data = load_yaml(str(_CONFIGS / "nano.yaml"))
    del data["sft"]["lr"]
    with pytest.raises(ConfigValidationError, match="sft.lr"):
        validate_required_config_fields(data, scope="sft")


def test_unknown_scope_raises() -> None:
    with pytest.raises(ConfigValidationError, match="scope"):
        validate_required_config_fields({}, scope="nope")


def test_all_variants_load_via_load_config() -> None:
    """现有变体配置都能通过 load_config(full) 完整校验 (回归护栏)。"""
    for name in ("base", "nano", "lite", "pro"):
        cfg = load_config(str(_CONFIGS / f"{name}.yaml"))
        assert cfg.model.d_model >= 64


# ── 反向护栏: scope 收窄后, 最小 YAML 应能在对应 scope 下通过 ──────────
# (full 对这些最小文件仍应报错: 收窄不等于放松全局默认)

_TRAINING_ONLY_YAML = """
training:
  epochs: 1
  batch_size: 8
  accumulate_grad: 8
  weight_decay: 0.01
  clip_grad: 1.0
  log_interval: 50
  save_interval: 2000
  seed: 42
  label_smoothing: 0.1
lr:
  type: wsd
  lr: 0.0004
  warmup_ratio: 0.02
  stable_ratio: 0.8
  min_lr_ratio: 0.1
  wsd_decay_style: linear
advanced:
  z_loss_weight: 0.0001
  num_workers: 0
data:
  tokenizer_path: ""
  data_dir: ""
  checkpoint_dir: ""
"""

_TOKENIZER_ONLY_YAML = """
data_sources: []
"""


def _write(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_training_only_yaml_accepted_under_training_scope(tmp_path: Path) -> None:
    """pretrain 训练默认值加载器: 只有 training/lr/advanced/data 的 YAML 可用。"""
    path = _write(tmp_path, "training_only.yaml", _TRAINING_ONLY_YAML)
    cfg = load_config(path, scope="training")
    assert cfg.training.epochs == 1
    with pytest.raises(ConfigValidationError):
        load_config(path, scope="full")


def test_tokenizer_only_yaml_accepted_under_tokenizer_scope(tmp_path: Path) -> None:
    """train_tokenizer: 只有 data_sources 的 YAML 可用。"""
    path = _write(tmp_path, "data_only.yaml", _TOKENIZER_ONLY_YAML)
    cfg = load_config(path, scope="tokenizer")
    assert cfg.data_sources == []
    with pytest.raises(ConfigValidationError):
        load_config(path, scope="full")
