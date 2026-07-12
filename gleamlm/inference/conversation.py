"""Multi-turn conversation manager with KV cache reuse."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from gleamlm.inference.generator import generate_tokens
from gleamlm.types import PastKeyValueList


class Conversation:
    """Multi-turn dialogue state manager.

    Maintains the full message history and reuses KV cache across turns
    for efficient multi-turn inference. Each call to generate_response()
    only encodes the new user message, reusing cached context from prior turns.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        system_prompt: str = "",
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        penalty_window: int = 50,
        lower_bound: int = 0,
        use_amp: bool = True,
        amp_dtype: torch.dtype | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.penalty_window = penalty_window
        self.lower_bound = lower_bound
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

        self.messages: list[dict[str, str]] = []
        self.past_kv: PastKeyValueList | None = None

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def append_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def generate_response(self) -> str:
        tokens: list[int] = []
        for token_id in self.stream_response():
            tokens.append(token_id)
        return self._decode_tokens(tokens)

    def stream_response(self) -> Iterator[int]:
        device = next(self.model.parameters()).device
        im_end_id = self.tokenizer.special_tokens.get("<|im_end|>")
        stop_ids: set[int] = {self.tokenizer.eos_id, self.tokenizer.pad_id}
        if im_end_id is not None:
            stop_ids.add(im_end_id)

        if self.past_kv is None:
            prompt_text = self._build_full_chatml()
        else:
            last_user = self.messages[-1]["content"]
            prompt_text = (
                f"<|im_start|><|user|>\n{last_user}<|im_end|>\n<|im_start|><|assistant|>\n"
            )

        prompt_ids = self.tokenizer.encode(prompt_text, add_bos=False, add_eos=False)

        kv_sink: list[PastKeyValueList | None] = [None]

        SENTENCE_ENDS = ("。", "！", "？", "；", "\n")
        LOWER = self.lower_bound if self.lower_bound > 0 else 0

        generated_tokens: list[int] = []
        _buffer: list[int] = []
        _stop_yielding = False
        _clean_cutoff = -1

        for token_id in generate_tokens(
            self.model,
            prompt_ids,
            device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            penalty_window=self.penalty_window,
            stop_ids=stop_ids,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            past_kv=self.past_kv,
            _kv_sink=kv_sink,
        ):
            generated_tokens.append(token_id)

            if _stop_yielding:
                continue

            if LOWER == 0 or len(generated_tokens) < LOWER:
                yield token_id
                continue

            _buffer.append(token_id)
            tail = self.tokenizer.decode(_buffer, skip_special=True)

            if tail and tail[-1] in SENTENCE_ENDS:
                for t in _buffer:
                    yield t
                _clean_cutoff = len(generated_tokens)
                _stop_yielding = True
                _buffer.clear()

        self.past_kv = kv_sink[0]

        if _stop_yielding and _clean_cutoff > 0:
            generated_tokens = generated_tokens[:_clean_cutoff]
        elif _buffer and LOWER > 0:
            tail = self.tokenizer.decode(_buffer, skip_special=True)
            cutoff = -1
            for sep in SENTENCE_ENDS:
                idx = tail.rfind(sep)
                if idx > cutoff:
                    cutoff = idx
            if cutoff >= 0:
                clean_ids = self.tokenizer.encode(tail[: cutoff + 1], add_bos=False, add_eos=False)
                generated_tokens = generated_tokens[: -len(_buffer)] + clean_ids

        stopped_clean = len(generated_tokens) < self.max_new_tokens

        if stopped_clean and im_end_id is not None:
            im_end_input = torch.tensor([[im_end_id]], dtype=torch.long, device=device)
            with torch.no_grad():
                if self.use_amp:
                    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.amp.autocast(amp_device, dtype=self.amp_dtype):  # type: ignore[attr-defined]
                        _, self.past_kv = self.model(im_end_input, past_kv_list=self.past_kv)
                else:
                    _, self.past_kv = self.model(im_end_input, past_kv_list=self.past_kv)

        self.messages.append(
            {"role": "assistant", "content": self._decode_tokens(generated_tokens)}
        )

    def _build_full_chatml(self) -> str:
        parts: list[str] = []
        for msg in self.messages[:-1]:
            role = msg["role"]
            if role == "system":
                parts.append(f"<|im_start|><|system|>\n{msg['content']}<|im_end|>\n")
            elif role == "user":
                parts.append(f"<|im_start|><|user|>\n{msg['content']}<|im_end|>\n")
            elif role == "assistant":
                parts.append(f"<|im_start|><|assistant|>\n{msg['content']}<|im_end|>\n")
        last_user = self.messages[-1]["content"]
        parts.append(f"<|im_start|><|user|>\n{last_user}<|im_end|>\n<|im_start|><|assistant|>\n")
        return "".join(parts)

    def _decode_tokens(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special=True)

    def get_history(self) -> list[dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []
        self.past_kv = None
