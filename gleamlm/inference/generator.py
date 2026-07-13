"""Unified autoregressive token generator — the single KV-cache loop used by all callers.

Replaces the duplicated per-file generate loops in generate_response(), TextStreamer.generate(),
knowledge._simple_generate(), and scripts/eval_knowledge.generate().
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from gleamlm.inference.sampler import sample_token
from gleamlm.types import PastKeyValueList
from gleamlm.utils.torch_utils import safe_autocast


def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: list[int],
    device: torch.device,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    penalty_window: int = 0,
    stop_ids: set[int] | None = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
    past_kv: PastKeyValueList | None = None,
    _kv_sink: list[PastKeyValueList | None] | None = None,
) -> Iterator[int]:
    """Autoregressive token generation with KV cache.

    Yields one token ID at a time. Stops when a token in `stop_ids` is produced
    or `max_new_tokens` is reached.

    Args:
        model: The GleamLMModel (or DDP-wrapped).
        prompt_ids: Pre-encoded prompt token IDs (without BOS/EOS padding).
        device: torch device.
        max_new_tokens: Maximum tokens to generate.
            Actual RoPE cache upper bound is model.rope_max_len, which may
            exceed the configured model.max_seq_len (due to pre-allocation).
        temperature, top_k, top_p, repetition_penalty: Sampling parameters.
        penalty_window: Sliding window size for repetition penalty (0 = all tokens).
        stop_ids: Set of token IDs that trigger generation stop. None = no early stop.
        use_amp: If True, wraps forward pass in torch.amp.autocast.
        amp_dtype: AMP dtype (bfloat16, float16). None = default autocast behaviour.
        past_kv: Initial KV cache from previous turns (multi-turn conversation).
        _kv_sink: Optional mutable list; [0] receives final KV cache after generation ends.
    """
    model.eval()
    generated_ids: list[int] = prompt_ids.copy()

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, past_kv = _forward(model, input_ids, use_amp, amp_dtype, past_kv)

    for _ in range(max_new_tokens):
        next_token = sample_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids,
            penalty_window=penalty_window,
        )
        token_id = int(next_token.item())

        if stop_ids and token_id in stop_ids:
            if _kv_sink is not None:
                _kv_sink[0] = past_kv
            return

        yield token_id
        generated_ids.append(token_id)
        next_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, past_kv = _forward(model, next_input, use_amp, amp_dtype, past_kv)

    if _kv_sink is not None:
        _kv_sink[0] = past_kv


def _forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    past_kv: PastKeyValueList | None = None,
) -> tuple[torch.Tensor, PastKeyValueList]:
    if use_amp:
        with safe_autocast(dtype=amp_dtype or torch.bfloat16):
            return model(input_ids, past_kv_list=past_kv)  # type: ignore[no-any-return]
    return model(input_ids, past_kv_list=past_kv)  # type: ignore[no-any-return]
