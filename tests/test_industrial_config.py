"""工业轨配置护栏测试 — pretrain 配置必读键齐全 + 与手动轨关键值对齐。

背景: 工业轨 industrial/configs/*.yaml 是 Megatron 键名语义的自包含配置,
其与手动轨 manual/configs/nano.yaml 的对齐承诺 (lr / z-loss / 有效 batch /
架构维度) 此前只活在注释里; 本文件用断言锁死: 任一轨被单独改动让对齐
断裂, 用例先红, 必须显式决定哪边为真再同步另一边。
"""

from pathlib import Path

import pytest
import yaml

import industrial.industrial_config as ic

_ROOT = Path(__file__).resolve().parents[1]
_IND = _ROOT / "industrial" / "configs"
_MAN = _ROOT / "manual" / "configs"


def _load(name: str) -> dict:
    with open(_IND / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 消费端必读校验 (load 即校验): 现网两份配置必须能通过 ──────────────────


@pytest.mark.parametrize("name", ["nano", "0.6b"])
def test_industrial_configs_pass_validation(name: str) -> None:
    cfg = ic.load_config(str(_IND / f"{name}.yaml"))  # 校验失败即抛错
    assert cfg["model"]["num_layers"] > 0


def test_missing_required_key_raises() -> None:
    """缺 z_loss_weight (曾静默回退 0=关 z-loss) 必须报错列名。"""
    data = _load("nano")
    del data["training"]["z_loss_weight"]
    with pytest.raises(ValueError, match="z_loss_weight"):
        ic.validate_industrial_config(data)


def test_missing_section_raises() -> None:
    with pytest.raises(ValueError, match="parallel"):
        ic.validate_industrial_config({"model": {}, "training": {}})


def test_wsd_requires_decay_style() -> None:
    """wsd 必须显式给 lr_wsd_decay_style (曾静默默认 cosine, 丢 nano linear)。"""
    data = _load("0.6b")  # 以 cosine 配置打底再改 wsd
    data["training"]["lr_decay_style"] = "wsd"
    with pytest.raises(ValueError, match="lr_wsd_decay_style"):
        ic.validate_industrial_config(data)


def test_warmup_alternative_ok() -> None:
    """lr_warmup_iters 与 lr_warmup_ratio 二选一 (0.6b 用 iters 形态)。"""
    cfg = _load("0.6b")
    assert "lr_warmup_iters" in cfg["training"]
    ic.validate_industrial_config(cfg)


# ── 对齐锁: industrial/configs/nano.yaml ↔ manual/configs/nano.yaml ────────


def test_industrial_nano_aligns_with_manual_nano() -> None:
    ind = _load("nano")
    with open(_MAN / "nano.yaml", encoding="utf-8") as f:
        man = yaml.safe_load(f)

    # 架构: Megatron 键名 ↔ 手动轨键名
    mi, mm = ind["model"], man["model"]
    assert mi["hidden_size"] == mm["d_model"]
    assert mi["num_layers"] == mm["num_layers"]
    assert mi["num_attention_heads"] == mm["num_heads"]
    assert mi["num_query_groups"] == mm["num_kv_heads"]
    assert mi["ffn_hidden_size"] == mm["d_ff"]
    assert mi["seq_length"] == mm["max_seq_len"]
    assert mi["vocab_size"] == mm["vocab_size"]

    # 训练: lr 调度对齐 (wsd + linear + 实证比例)
    ti, tm = ind["training"], man["training"]
    assert ti["lr"] == man["lr"]["lr"]
    assert ti["lr_decay_style"] == man["lr"]["type"] == "wsd"
    assert ti["lr_warmup_ratio"] == man["lr"]["warmup_ratio"]
    assert ti["stable_ratio"] == man["lr"]["stable_ratio"]
    assert ti["lr_wsd_decay_style"] == man["lr"]["wsd_decay_style"] == "linear"
    assert ti["min_lr"] == pytest.approx(man["lr"]["lr"] * man["lr"]["min_lr_ratio"])
    assert ti["z_loss_weight"] == man["advanced"]["z_loss_weight"]
    assert ti["bf16"] == man["advanced"]["bf16"]
    # 有效 batch 一致 (工业 micro×accum = 手写 batch×accum)
    assert (
        ti["micro_batch_size"] * ti["accumulate_grad"] == tm["batch_size"] * tm["accumulate_grad"]
    )


# ── 对齐锁: 0.6b 与项目 0.6B 定义 (头注释承诺: 37L×1024d×GQA16/8×4096ffn×seq4096×BBPE24K) ─


def test_industrial_06b_matches_project_definition() -> None:
    m = _load("0.6b")["model"]
    assert m["num_layers"] == 37
    assert m["hidden_size"] == 1024
    assert m["num_attention_heads"] == 16
    assert m["num_query_groups"] == 8
    assert m["ffn_hidden_size"] == 4096
    assert m["seq_length"] == 4096
    assert m["max_position_embeddings"] == 4096
    assert m["vocab_size"] == 24002
