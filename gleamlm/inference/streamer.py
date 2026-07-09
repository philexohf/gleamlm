"""GleamLM 流式文本生成器。KV Cache + 采样，逐 token 输出"""

from __future__ import annotations

from collections.abc import Generator

import torch

from gleamlm.inference.sampler import sample_token
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer


class TextStreamer:
    """流式文本生成器。逐步生成 token，支持实时输出"""

    def __init__(self, tokenizer: BBPETokenizer) -> None:
        self.tokenizer = tokenizer

    def generate(
        self,
        model: GleamLMModel,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_id: int | None = None,
    ) -> Generator[int, None, None]:
        if eos_id is None:
            eos_id = self.tokenizer.eos_id

        assert prompt_ids.size(0) == 1, (
            f"generate() 仅支持 batch_size=1，当前 batch={prompt_ids.size(0)}"
        )
        device = next(model.parameters()).device
        input_ids = prompt_ids.to(device)
        generated_ids: list[int] = prompt_ids[0].tolist()

        with torch.no_grad():
            logits, past_kv = model(input_ids)

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]
            next_token = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )

            token_id = next_token.item()
            if token_id == eos_id:
                return
            yield token_id
            generated_ids.append(token_id)

            with torch.no_grad():
                logits, past_kv = model(next_token.unsqueeze(0), past_kv_list=past_kv)

    def generate_text(
        self,
        model: GleamLMModel,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_on_endoftext: bool = False,
    ) -> Generator[str, None, None]:
        prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        generated_ids: list[int] = []
        byte_buffer = bytearray()
        total_decoded = ""
        for token_id in self.generate(
            model, prompt_tensor, max_new_tokens, temperature, top_k, top_p, repetition_penalty
        ):
            generated_ids.append(token_id)
            byte_buffer.extend(self.tokenizer.id_to_byte.get(token_id, b"?"))

            if len(generated_ids) % 4 == 0:
                try:
                    text = byte_buffer.decode("utf-8")
                    new_part = text[len(total_decoded) :]
                    total_decoded = ""
                    byte_buffer = bytearray()
                except UnicodeDecodeError as e:
                    if e.start == 0:
                        continue
                    text = byte_buffer[: e.start].decode("utf-8")
                    new_part = text[len(total_decoded) :]
                    total_decoded = text
                    byte_buffer = byte_buffer[e.start :]

                if stop_on_endoftext and "<|endoftext|>" in new_part:
                    new_part = new_part.split("<|endoftext|>")[0]
                    yield new_part
                    return
                if new_part:
                    yield new_part

        if byte_buffer:
            final_text = byte_buffer.decode("utf-8", errors="replace")
            if stop_on_endoftext and "<|endoftext|>" in final_text:
                final_text = final_text.split("<|endoftext|>")[0]
            yield final_text
