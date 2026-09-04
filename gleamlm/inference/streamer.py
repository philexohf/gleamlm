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
                # 终止符: eos(<|im_end|>) 是文档/SFT 终止符（预训练 .bin 以 im_end 收尾），
                # pad(<|endoftext|>) 正常文本不会出现（仅 batch 对齐），带上仅为防御性停止
                stop_ids={self.tokenizer.eos_id, self.tokenizer.pad_id},
            )
        ):
            generated_ids.append(token_id)
            byte_buffer.extend(self.tokenizer.id_to_byte.get(token_id, b"?"))

            if len(generated_ids) % 4 == 0:
                decoded = _decode_cumulative(byte_buffer)
                if decoded is None:
                    continue
                new_text = decoded[len(total_decoded) :]
                if not new_text:
                    continue
                total_decoded = decoded

                if (
                    stop_on_endoftext
                    and self.tokenizer.eos_token
                    and self.tokenizer.eos_token in new_text
                ):
                    yield new_text.split(self.tokenizer.eos_token)[0]
                    return
                yield new_text

        if byte_buffer:
            # 生成结束 (max_new_tokens/stop token): 解码剩余字节
            final_text = _decode_cumulative(byte_buffer) or ""
            final_new = final_text[len(total_decoded) :]
            if final_new:
                yield final_new


def _decode_cumulative(byte_buffer: bytearray) -> str | None:
    """Decode the whole accumulated buffer as far as valid UTF-8 allows.

    关键: buffer 永不清空 (累积式)。多字节 UTF-8 字符可能跨 4-token 窗口边界，
    若尾部是不完整序列，只解码其前缀、保留全部字节等后续补齐；否则下一次
    slice 用字符长度 (len(total_decoded)) 会对不上。返回值: 可解码的累积文本。
    """
    try:
        return byte_buffer.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start == 0:
            return None  # 前几个字节就不完整，等更多 token
        return byte_buffer[: e.start].decode("utf-8")
