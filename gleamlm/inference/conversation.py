"""Multi-turn conversation manager with KV cache reuse."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from gleamlm.inference.chatml import format_chatml
from gleamlm.inference.generator import generate_tokens
from gleamlm.types import PastKeyValueList
from gleamlm.utils.torch_utils import safe_autocast


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
            prompt_text = format_chatml(
                self.messages, add_generation_prompt=True
            )
        else:
            last_user = self.messages[-1]["content"]
            prompt_text = format_chatml(
                [{"role": "user", "content": last_user}],
                add_generation_prompt=True,
            )

        prompt_ids = self.tokenizer.encode(prompt_text, add_bos=False, add_eos=False)

        kv_sink: list[PastKeyValueList | None] = [None]

        SENTENCE_ENDS = ("。", "！", "？", "；", "\n", ".", "!", "?")
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

        _total_before_trim = len(generated_tokens)
        self.past_kv = kv_sink[0]

        if _stop_yielding and _clean_cutoff > 0:
            generated_tokens = generated_tokens[:_clean_cutoff]
            removed = _total_before_trim - len(generated_tokens)
            if removed > 0:
                self.past_kv = [
                    (k[:, :, :-removed], v[:, :, :-removed])
                    for k, v in kv_sink[0]
                ]
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
                with safe_autocast(enabled=self.use_amp, dtype=self.amp_dtype or torch.bfloat16):
                    _, self.past_kv = self.model(im_end_input, past_kv_list=self.past_kv)

        self.messages.append(
            {"role": "assistant", "content": self._decode_tokens(generated_tokens)}
        )

    def _decode_tokens(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special=True)

    def get_history(self) -> list[dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []
        self.past_kv = None
