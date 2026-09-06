"""验证 GleamLMForCausalLM save_pretrained → from_pretrained 往返一致性。

重点：高级 config 字段（rope_scale, attn_type, layer_configs 等）不能丢失。"""

import gc
import json
import tempfile
from pathlib import Path

import torch

from hf.hf_config import GleamLMConfig
from hf.hf_model import GleamLMForCausalLM

CFG = {
    "vocab_size": 3200,
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 340,
    "max_position_embeddings": 256,
    "use_flash_attn": False,
    "attn_type": "gqa",
    "ffn_type": "mlp",
    "num_experts": 8,
    "top_k": 2,
    "rope_scale": 4.0,
    "rope_factor": 16.0,
    "rope_theta": 500000.0,
    "layer_configs": None,
}

SENSITIVE_KEYS = {
    "rope_scale",
    "rope_factor",
    "rope_theta",
    "attn_type",
    "ffn_type",
    "num_experts",
    "top_k",
}
# PretrainedConfig 序列化 self._top_k（带下划线），config.json 里 key 也是 _top_k
JSON_FIELDS = {
    "rope_scale",
    "rope_factor",
    "rope_theta",
    "attn_type",
    "ffn_type",
    "num_experts",
    "_top_k",
}
JSON_VALUES = {
    "rope_scale": 4.0,
    "rope_factor": 16.0,
    "rope_theta": 500000.0,
    "attn_type": "gqa",
    "ffn_type": "mlp",
    "num_experts": 8,
    "_top_k": 2,
}


# ── 1. 权重往返 ──
def test_weight_roundtrip():
    original = GleamLMForCausalLM(GleamLMConfig(**CFG))
    original.eval()

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "test-model"
        original.save_pretrained(str(save_dir))

        loaded = GleamLMForCausalLM.from_pretrained(str(save_dir))
        loaded.eval()

        original_sd = original.state_dict()
        loaded_sd = loaded.state_dict()
        assert set(original_sd.keys()) == set(loaded_sd.keys()), "state_dict keys 不一致"
        for key in original_sd:
            assert torch.equal(original_sd[key], loaded_sd[key]), f"权重不一致: {key}"
        del original_sd, loaded_sd, original, loaded
        gc.collect()


# ── 2. config.json 序列化 ──
def test_config_json_roundtrip():
    original_config = GleamLMConfig(**CFG)

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "test-model"
        original_config.save_pretrained(str(save_dir))

        config_path = save_dir / "config.json"
        assert config_path.exists(), "config.json 未生成"

        raw = json.loads(config_path.read_text())

    for key in JSON_FIELDS:
        assert key in raw, f"config.json 缺少关键字段: {key}"
        assert raw[key] == JSON_VALUES[key], (
            f"config.json 字段 {key} 值不匹配: {raw[key]} != {JSON_VALUES[key]}"
        )


# ── 3. 从 config.json 加载后字段恢复 ──
def test_config_reload():
    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "test-model"
        original_config = GleamLMConfig(**CFG)
        original_config.save_pretrained(str(save_dir))

        loaded_config = GleamLMConfig.from_pretrained(str(save_dir))

    for key in SENSITIVE_KEYS:
        loaded_val = getattr(loaded_config, key)
        assert loaded_val == CFG[key], f"config 字段 {key} 加载后不匹配: {loaded_val} != {CFG[key]}"

    # top_k 用 _top_k 存储，验证属性访问正确
    assert loaded_config.top_k == CFG["top_k"], "top_k 属性访问异常"


# ── 4. 权重往返后 forward 一致 ──
def test_forward_roundtrip():
    original = GleamLMForCausalLM(GleamLMConfig(**CFG))
    original.eval()

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "test-model"
        original.save_pretrained(str(save_dir))
        loaded = GleamLMForCausalLM.from_pretrained(str(save_dir))
        loaded.eval()

        input_ids = torch.randint(0, CFG["vocab_size"], (1, 32))
        with torch.no_grad():
            out_orig = original(input_ids)
            out_load = loaded(input_ids)

        assert torch.allclose(out_orig.logits, out_load.logits), "forward logits 不一致"
        del loaded, original  # windows: release mmap before temp dir cleanup
        gc.collect()
