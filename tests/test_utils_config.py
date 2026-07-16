"""YAML 配置 加载/extends/CLI覆盖/校验 测试"""

import os
import tempfile

from gleamlm.utils.config import (
    _apply_overrides,
    _deep_merge,
    _DictWrapper,
    _parse_cli_overrides,
    _validate_config,
    load_config,
    load_config_as_args,
    load_yaml,
    to_namespace,
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


# _DictWrapper


def test_dict_wrapper():
    dw = _DictWrapper({"model": {"d_model": 512}, "training": {"epochs": 5}})
    assert dw.model.d_model == 512
    assert dw.training.epochs == 5


def test_dict_wrapper_get():
    dw = _DictWrapper({"a": 1, "b": {"c": 2}})
    assert dw.get("a") == 1
    assert dw.get("b").c == 2
    assert dw.get("nonexist") is None
    assert dw.get("nonexist", 0) == 0


def test_dict_wrapper_setattr():
    dw = _DictWrapper({"a": 1})
    dw.b = 2
    assert dw.b == 2
    assert dw._data["b"] == 2


def test_dict_wrapper_contains():
    dw = _DictWrapper({"a": 1})
    assert "a" in dw
    assert "b" not in dw


def test_dict_wrapper_to_dict():
    dw = _DictWrapper({"a": 1, "b": {"c": 2}})
    d = dw.to_dict()
    assert d == {"a": 1, "b": {"c": 2}}
    d["a"] = 99
    assert dw.a == 1


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


# _parse_cli_overrides / _apply_overrides


def test_parse_cli_overrides():
    cfg = {"model": {"d_model": 512}, "training": {"epochs": 4}}
    # no CLI args → empty overrides
    overrides = _parse_cli_overrides(cfg)
    assert isinstance(overrides, dict)


def test_apply_overrides_flat():
    cfg = {"model": {"d_model": 512, "dropout": 0.1}}
    _apply_overrides(cfg, {"model.d_model": 768, "model.dropout": 0.0})
    assert cfg["model"]["d_model"] == 768
    assert cfg["model"]["dropout"] == 0.0


def test_apply_overrides_str_to_bool():
    cfg = {"model": {"tie_weights": True}}
    _apply_overrides(cfg, {"model.tie_weights": "false"})
    assert cfg["model"]["tie_weights"] is False


def test_apply_overrides_str_to_int():
    cfg = {"training": {"epochs": 4}}
    _apply_overrides(cfg, {"training.epochs": "10"})
    assert cfg["training"]["epochs"] == 10


def test_apply_overrides_str_to_float():
    cfg = {"lr": {"lr": 0.001}}
    _apply_overrides(cfg, {"lr.lr": "0.0003"})
    assert cfg["lr"]["lr"] == 0.0003


def test_apply_overrides_new_section():
    cfg = {"model": {"d_model": 512}}
    _apply_overrides(cfg, {"training.epochs": 5})
    assert cfg["training"]["epochs"] == 5


# to_namespace


def test_to_namespace_basic():
    cfg = _DictWrapper(
        {
            "model": {"d_model": 512, "num_layers": 12},
            "training": {"epochs": 4},
            "optimizer": {"lr": 0.001},
            "distributed": {"world_size": 1},
        }
    )
    ns = to_namespace(cfg)
    assert ns.d_model == 512
    assert ns.num_layers == 12
    assert ns.epochs == 4
    assert ns.optimizer_lr == 0.001
    assert ns.distributed_world_size == 1


def test_to_namespace_key_collision_warning():
    cfg = _DictWrapper(
        {
            "model": {"epochs": 1},
            "training": {"epochs": 4},
        }
    )
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ns = to_namespace(cfg)
        assert ns.epochs == 4


def test_to_namespace_non_dict_section():
    cfg = _DictWrapper({"experiment_name": "test_run"})
    ns = to_namespace(cfg)
    assert ns.experiment_name == "test_run"


# load_config (real YAML)


def test_load_config_full():
    with tempfile.TemporaryDirectory() as tmp:
        base_path = os.path.join(tmp, "base.yaml")
        cfg_path = os.path.join(tmp, "nano.yaml")

        with open(base_path, "w", encoding="utf-8") as f:
            f.write(
                "model:\n  d_model: 512\n  num_layers: 12\n"
                "training:\n  epochs: 4\n  batch_size: 8\n"
                "lr:\n  lr: 0.0003\n  warmup_ratio: 0.01\n"
            )
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("extends: base.yaml\nmodel:\n  d_model: 768\n")

        cfg = load_config(cfg_path, model_name="nano")
        assert cfg.model.d_model == 768
        assert cfg.model.num_layers == 12
        assert cfg.training.epochs == 4
        assert hasattr(cfg, "data")


def test_load_config_as_args():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "nano.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(
                "model:\n  d_model: 512\n  num_layers: 12\n  num_heads: 8\n"
                "training:\n  epochs: 4\n  batch_size: 8\n"
                "lr:\n  lr: 0.0003\n  warmup_ratio: 0.01\n"
            )

        args = load_config_as_args(cfg_path)
        assert args.d_model == 512
        assert args.num_layers == 12
        assert args.epochs == 4


def test_load_config_real_nano():
    cfg = load_config("configs/nano.yaml", model_name="nano")
    assert cfg.model.d_model == 512
    assert cfg.model.num_layers == 12
    assert cfg.model.num_heads == 8
    assert cfg.model.num_kv_heads == 4
    assert cfg.training.epochs == 1
    assert cfg.advanced.bf16 is True


# 配置校验

SAMPLE_VALID_CFG = {
    "model": {
        "d_model": 512,
        "num_layers": 12,
        "num_heads": 8,
        "num_kv_heads": 4,
        "d_ff": 1365,
        "max_seq_len": 1024,
        "vocab_size": 12002,
        "dropout": 0.1,
        "tie_weights": True,
        "use_flash_attn": False,
    },
    "training": {
        "epochs": 5,
        "batch_size": 8,
        "accumulate_grad": 8,
        "clip_grad": 1.0,
        "weight_decay": 0.01,
        "seed": 42,
        "max_train_chars": 1_200_000_000,
    },
    "lr": {
        "lr": 0.0003,
        "warmup_ratio": 0.01,
        "min_lr_ratio": 0.1,
    },
    "advanced": {
        "z_loss_weight": 0.0,
        "bf16": False,
    },
}


def test_validate_config_valid():
    _validate_config(SAMPLE_VALID_CFG)


def test_validate_config_type_error():
    bad = dict(SAMPLE_VALID_CFG)
    bad["model"] = dict(bad["model"], d_model="512")
    import pytest

    with pytest.raises(ValueError, match="d_model"):
        _validate_config(bad)


def test_validate_config_range_error():
    bad = dict(SAMPLE_VALID_CFG)
    bad["model"] = dict(bad["model"], num_layers=0)
    import pytest

    with pytest.raises(ValueError, match="num_layers"):
        _validate_config(bad)


def test_validate_config_dropout_range():
    bad = dict(SAMPLE_VALID_CFG)
    bad["model"] = dict(bad["model"], dropout=1.5)
    import pytest

    with pytest.raises(ValueError, match="dropout"):
        _validate_config(bad)


def test_validate_config_bool_field():
    bad = dict(SAMPLE_VALID_CFG)
    bad["model"] = dict(bad["model"], tie_weights=1)
    import pytest

    with pytest.raises(ValueError, match="tie_weights"):
        _validate_config(bad)
