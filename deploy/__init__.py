"""GleamLM 模型部署 — HF 导出 / 量化 / ONNX。

export.py       — checkpoint → HF 格式 (safetensors + config + tokenizer)，供 vLLM/TRL 加载
quantize.py     — FP16 / INT8 / INT4 量化 (torchao)
export_onnx.py  — ONNX 导出 + 推理 (prefill-only)
"""

from gleamlm.deploy.export import convert_checkpoint, get_vllm_config, VLLMEngine
from gleamlm.deploy.quantize import quantize_ckpt

__all__ = [
    "convert_checkpoint",
    "get_vllm_config",
    "VLLMEngine",
    "quantize_ckpt",
]
