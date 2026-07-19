"""GleamLM 流式文本生成器。KV Cache + 采样，逐 token 输出"""

from __future__ import annotations

from collections.abc import Generator

from gleamlm.inference.generator import generate_tokens
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer


class TextStreamer:
    """Streaming text generator with byte-level incremental decoding."""

    def __init__(self, tokenizer: BBPETokenizer) -> None:
        self.tokenizer = tokenizer

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
        """Generate text chunks incrementally via byte-level UTF-8 decoding."""
        prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        device = next(model.parameters()).device

        generated_ids: list[int] = []
        byte_buffer = bytearray()
        total_decoded = ""

        for _i, token_id in enumerate(
            generate_tokens(
                model,
                prompt_ids,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_ids={self.tokenizer.eos_id},
            )
        ):
            generated_ids.append(token_id)
            byte_buffer.extend(self.tokenizer.id_to_byte.get(token_id, b"?"))

            if len(generated_ids) % 4 == 0:
                result = _decode_incremental(byte_buffer)
                if result is None:
                    continue
                decoded_text, byte_buffer = result
                new_text = decoded_text[len(total_decoded) :]
                total_decoded = decoded_text

                if (
                    stop_on_endoftext
                    and self.tokenizer.eos_token
                    and self.tokenizer.eos_token in new_text
                ):
                    yield new_text.split(self.tokenizer.eos_token)[0]
                    return
                if new_text:
                    yield new_text

        if byte_buffer:
            final_text = byte_buffer.decode("utf-8", errors="replace")
            if (
                stop_on_endoftext
                and self.tokenizer.eos_token
                and self.tokenizer.eos_token in final_text
            ):
                final_text = final_text.split(self.tokenizer.eos_token)[0]
            yield final_text


def _decode_incremental(byte_buffer: bytearray) -> tuple[str, bytearray] | None:
    """Attempt incremental UTF-8 decode. Returns (full_text, remaining_bytes) or None."""
    try:
        text = byte_buffer.decode("utf-8")
        return (text, bytearray())
    except UnicodeDecodeError as e:
        if e.start == 0:
            return None
        text = byte_buffer[: e.start].decode("utf-8")
        return (text, byte_buffer[e.start :])
