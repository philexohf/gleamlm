"""GleamLM 模型导出 — HF 格式（safetensors）+ vLLM 推理适配。

用法:
    # 导出 HF 格式（供 vLLM / TRL / lm-eval 加载）
    python deploy/export.py --input checkpoints/best_model.pt --output ./exported/

    # vLLM 启动
    python -m vllm.entrypoints.openai.api_server --model ./exported/ --trust-remote-code
"""

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from hf.hf_config import gleamlm_config_from_core
from hf.hf_model import GleamLMForCausalLM
from gleamlm.utils.config import extract_checkpoint_config


def convert_checkpoint(
    gleamlm_ckpt_path: str,
    output_dir: str,
    config_overrides: dict | None = None,
    tokenizer_dir: str | None = None,
) -> str:
    """将 gleamlm checkpoint 转为完整 HF 格式。

    输出目录结构:
      {output_dir}/
        config.json        ← GleamLMConfig → HF config
        model.safetensors  ← state_dict (带 model. 前缀)
        tokenizer.json     ← 由 tokenizer_dir 导出
    """
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(gleamlm_ckpt_path, map_location="cpu", weights_only=True)
    model_dict = extract_checkpoint_config(ckpt)

    overrides = config_overrides or {}
    model_dict.update(overrides)

    config = gleamlm_config_from_core(model_dict)
    config.save_pretrained(output_dir)

    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # HF 需要 model. 前缀
    hf_sd = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in sd.items()}

    # tied weights: lm_head 与 token_embed 共享 storage 时删除 lm_head，
    # 靠 config.tie_word_embeddings 在加载时自动重建（LLaMA/Qwen 标准做法）
    if (
        "model.lm_head.weight" in hf_sd
        and "model.token_embed.weight" in hf_sd
        and hf_sd["model.lm_head.weight"] is hf_sd["model.token_embed.weight"]
    ):
        hf_sd.pop("model.lm_head.weight", None)

    from safetensors.torch import save_file
    save_file(hf_sd, os.path.join(output_dir, "model.safetensors"))

    # 导出 HF tokenizer
    if tokenizer_dir:
        from gleamlm.tokenizer.tokenizer import BBPETokenizer
        BBPETokenizer.load(tokenizer_dir).export_to_hf_format(output_dir)

    print(f"Checkpoint converted: {gleamlm_ckpt_path} → {output_dir}")
    return output_dir


def get_vllm_config() -> dict[str, Any]:
    """vLLM 模型兼容配置参考。

    若 vLLM 不原生支持 gleam_lm model_type:
      - 用 --trust-remote-code 强制走 HF AutoModel 路径
      - 或在 vLLM 注册: {"gleam_lm": "vllm.model_executor.models.gleam_lm"}
    """
    return {
        "model_type": "gleam_lm",
        "trust_remote_code": True,
        "config_format": "hf",
    }


class VLLMEngine:
    """vLLM 引擎封装 — 异步流式推理。

    依赖: pip install vllm

    engine = VLLMEngine("./exported/")
    async for chunk in engine.generate("Hello", max_tokens=256):
        print(chunk, end="", flush=True)
    """

    def __init__(self, model_path: str, **engine_kwargs):
        self.model_path = model_path
        self.engine_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 4096,
            **engine_kwargs,
        }
        self._engine = None

    def _lazy_init(self):
        if self._engine is not None:
            return
        try:
            from vllm import AsyncLLMEngine, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            self._sampling_params_cls = SamplingParams
            args = AsyncEngineArgs(**self.engine_kwargs)
            self._engine = AsyncLLMEngine.from_engine_args(args)
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, **kwargs):
        self._lazy_init()
        params = self._sampling_params_cls(
            max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        request_id = f"req_{id(prompt)}"
        async for result in self._engine.generate(prompt, params, request_id):
            yield result.outputs[0].text

    def close(self):
        if self._engine:
            self._engine.shutdown_background_loop()


# ── CLI ──

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GleamLM HF format export")
    p.add_argument("--input", type=str, required=True, help="Checkpoint .pt path")
    p.add_argument("--output", type=str, required=True, help="Output directory")
    p.add_argument("--tokenizer_dir", type=str, default=None, help="Tokenizer directory")
    args = p.parse_args()
    convert_checkpoint(args.input, args.output, tokenizer_dir=args.tokenizer_dir)
