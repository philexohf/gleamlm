"""GleamLM API — convenience entry point for inference."""

from __future__ import annotations

from typing import Any

import torch

from gleamlm import load_model_for_inference
from gleamlm.inference.generate import generate_response
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.types import ConfigValidationError


class GleamLM:
    """Inference client wrapping model and tokenizer.

    Usage:
        model = GleamLM.from_checkpoint("checkpoints/nano/best_model.pt")
        print(model.generate("你好"))
        result = model.evaluate("data/nano/pretrain")

    也可直接构造（测试友好）:
        model = GleamLM(m, tok, cfg, torch.device("cuda"))
    """

    def __init__(
        self,
        model: GleamLMModel,
        tokenizer: BBPETokenizer,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: str = "auto", tokenizer_path: str | None = None
    ) -> GleamLM:
        """Load model from checkpoint."""
        dev = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        model, config = load_model_for_inference(checkpoint_path, dev)

        tk_path = tokenizer_path or config.get("tokenizer_path")
        if not tk_path:
            raise ConfigValidationError("无法确定 tokenizer 路径，请通过 tokenizer_path 参数指定")
        tokenizer = BBPETokenizer.load(tk_path)

        total, _ = model.get_num_params()
        print(f"Model: {total / 1e6:.2f}M params, device: {dev}")

        return cls(model, tokenizer, config, dev)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Generate a response with input validation."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt 须为 str，收到 {type(prompt)}")
        if not 0 < temperature <= 2.0:
            raise ValueError(f"temperature 须 0 < t ≤ 2.0，收到 {temperature}")

        return generate_response(
            self._model,
            self._tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    def evaluate(
        self,
        data_dir: str,
        *,
        dataset: str = "test",
        max_seq_len: int = 1024,
        batch_size: int = 4,
        max_batches: int | None = None,
    ) -> dict:
        """Evaluate PPL on a dataset."""
        from gleamlm.evaluation.ppl import evaluate_ppl

        result = evaluate_ppl(
            self._model,
            self._tokenizer,
            data_dir,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=str(self._device),
            dataset=dataset,
            max_batches=max_batches,
        )
        return result.to_dict()
