"""ONNX 导出 + 推理 — 生产部署格式（prefill-only）。

用法:
    python -c "from deploy.export_onnx import export_to_onnx; export_to_onnx('model.pt', 'model.onnx')"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any

import torch
import torch.nn.functional as F


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    max_seq_len: int = 512,
    opset_version: int = 17,
) -> str:
    """导出 GleamLM 到 ONNX (prefill-only 静态图)。

    限制: ONNX 不支持动态 KV cache 推理。decoding 阶段建议用 vLLM。
    """
    from gleamlm.utils.config import extract_checkpoint_config
    from hf.hf_config import gleamlm_config_from_core
    from hf.hf_model import GleamLMForCausalLM, load_from_checkpoint

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = extract_checkpoint_config(ckpt)
    config = gleamlm_config_from_core(cfg)
    config.max_position_embeddings = max_seq_len

    model = GleamLMForCausalLM(config)
    load_from_checkpoint(model, ckpt, strict=True)
    model.eval()

    dummy_input = torch.randint(0, config.vocab_size, (1, max_seq_len), dtype=torch.long)

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            logits, _, _, _ = self.m.model(input_ids, past_kv_list=None)
            return logits

    wrapped = ONNXWrapper(model)
    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"ONNX exported: {output_path}")
    return output_path


class ONNXInference:
    """ONNX Runtime 推理 (prefill-only)。

    inf = ONNXInference("model.onnx")
    logits = inf.run(torch.tensor([[1, 2, 3]]))
    """

    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. pip install onnxruntime-gpu")

        self._sess = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        result = self._sess.run(
            ["logits"], {"input_ids": input_ids.numpy().astype("int64")},
        )
        return torch.from_numpy(result[0])

    @torch.no_grad()
    def generate(
        self, prompt: str, tokenizer: Any, max_new_tokens: int = 128, temperature: float = 0.0,
    ) -> str:
        input_ids = tokenizer.encode(prompt)
        for _ in range(max_new_tokens):
            logits = self.run(torch.tensor([input_ids], dtype=torch.long))
            next_logits = logits[0, -1, :]
            if temperature == 0.0:
                next_id = next_logits.argmax(-1).item()
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            input_ids.append(next_id)
        return tokenizer.decode(input_ids)
