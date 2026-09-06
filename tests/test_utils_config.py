"""YAML 配置 加载/extends/校验 测试 (单轨: load_config → GleamLMConfig)"""

import os
import tempfile

from gleamlm.types import ConfigValidationError
from gleamlm.utils.config import (
    _deep_merge,
    extract_checkpoint_config,
    load_config,
    load_yaml,
)

# _deep_merge


def test_deep_merge():
    base = {"a": 1, "b": {"x": 2}}
    override = {"b": {"y": 3}}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"x": 2, "y": 3}}


def test_deep_merge_override():
    result = _deep_merge({"a": 1}, {"a": 2})
    assert result == {"a": 2}


# load_yaml


def test_load_yaml_with_extends():
    with tempfile.TemporaryDirectory() as tmp:
        base_path = os.path.join(tmp, "base.yaml")
        nano_path = os.path.join(tmp, "nano.yaml")

        with open(base_path, "w", encoding="utf-8") as f:
            f.write("model:\n  d_model: 512\ntraining:\n  epochs: 4\n")
        with open(nano_path, "w", encoding="utf-8") as f:
            f.write("extends: base.yaml\nmodel:\n  d_model: 256\n")

        cfg = load_yaml(nano_path)
        assert cfg["model"]["d_model"] == 256
        assert cfg["training"]["epochs"] == 4


def test_load_yaml_no_extends():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "standalone.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("model:\n  d_model: 768\n")
        cfg = load_yaml(cfg_path)
        assert cfg["model"]["d_model"] == 768


# load_config (单轨 Pydantic)


def test_load_config_requires_full_sections():
    """缺必读段直接报错, 不静默回退 Pydantic 默认 (单轨语义锁死)。"""
    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "minimal.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("model:\n  d_model: 512\n")
        with pytest.raises(ConfigValidationError, match="data_sources|sft"):
            load_config(cfg_path)


def test_load_config_model_only_under_full_scope_still_rejected():
    """即使 YAML 含 model 段, 缺 training/lr 等段时 full 仍报错 (防全局放松)。"""
    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "model_only.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(
                "model:\n"
                "  d_model: 512\n  num_layers: 12\n  num_heads: 8\n  num_kv_heads: 4\n"
                "  d_ff: 1365\n  max_seq_len: 1024\n  vocab_size: 12002\n"
                "  dropout: 0.1\n  tie_weights: true\n  use_flash_attn: false\n"
            )
        with pytest.raises(ConfigValidationError, match="training"):
            load_config(cfg_path, scope="full")


def test_load_config_real_nano():
    cfg = load_config("configs/nano.yaml")
    assert cfg.model.d_model == 512
    assert cfg.model.num_layers == 12
    assert cfg.model.num_heads == 8
    assert cfg.model.num_kv_heads == 4
    assert cfg.training.epochs == 1
    assert cfg.advanced.bf16 is True
    # sft/dpo 段属性链直读 (原 cfg_to_namespace 拍平字段的语义)
    assert cfg.sft.epochs == 3
    assert cfg.sft.lr == 1e-4
    assert cfg.dpo.epochs == 1
    assert cfg.optimizer.type == "adamw"


def test_load_config_missing_consumed_key_raises():
    """单段内缺必读键 (sft.lr) 同样显式报错。"""
    import pytest

    cfg_path = "configs/nano.yaml"
    data = load_yaml(cfg_path)
    del data["sft"]["lr"]
    from gleamlm.utils.config import validate_required_config_fields

    with pytest.raises(ConfigValidationError, match="sft.lr"):
        validate_required_config_fields(data)


def test_load_config_scope_specific_missing_key_raises():
    """scope 收窄后, scope 内缺键仍需报错 (dpo scope 缺 dpo.lr)。"""
    import pytest

    cfg_path = "configs/nano.yaml"
    data = load_yaml(cfg_path)
    del data["dpo"]["lr"]
    from gleamlm.utils.config import validate_required_config_fields

    with pytest.raises(ConfigValidationError, match="dpo.lr"):
        validate_required_config_fields(data, scope="dpo")


def test_narrow_scope_ignores_irrelevant_missing_keys():
    """收窄收益: dpo scope 只校验自己读的键, 不读的缺失键/s段不报错。"""
    from gleamlm.utils.config import validate_required_config_fields

    cfg_path = "configs/nano.yaml"
    data = load_yaml(cfg_path)

    # dpo scope 不读 sft 段 / model.num_layers / model.vocab_size → 删除后仍应通过
    del data["sft"]
    for k in ("num_layers", "vocab_size", "d_model", "d_ff", "tie_weights"):
        del data["model"][k]
    validate_required_config_fields(data, scope="dpo")

    # 反之, full scope 对同一份"删了 model.num_layers 等键"的 YAML 会报错
    import pytest

    with pytest.raises(ConfigValidationError):
        validate_required_config_fields(data, scope="full")


# ── extract_checkpoint_config: checkpoint 结构快照，唯一格式 `_config` ──


def test_extract_checkpoint_config_returns_config_snapshot():
    ckpt = {"_config": {"d_model": 512, "num_layers": 12}, "model_state_dict": {}}
    cfg = extract_checkpoint_config(ckpt)
    assert cfg["d_model"] == 512
    assert cfg["num_layers"] == 12


def test_extract_checkpoint_config_requires_underscore_config():
    # 旧格式 (args/config) 已随格式统一移除 → 缺 _config 一律显式报错
    import pytest

    for stale in ({"args": None}, {"config": {"d_model": 512}}, {"model_state_dict": {}}):
        with pytest.raises(ConfigValidationError, match="_config"):
            extract_checkpoint_config(stale)
