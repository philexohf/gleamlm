"""Unified autoregressive token generator — KV-cache loop + sampling, used by all callers."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from gleamlm.types import PastKeyValueList
from gleamlm.utils.torch_utils import safe_autocast


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
    penalty_window: int = 0,
) -> torch.Tensor:

    if repetition_penalty != 1.0 and generated_ids is not None:
        if logits.requires_grad:
            logits = logits.clone()
        window_ids = generated_ids[-penalty_window:] if penalty_window > 0 else generated_ids
        for gid in set(window_ids):
            scores = logits[..., gid]
            logits[..., gid] = torch.where(
                scores < 0, scores * repetition_penalty, scores / repetition_penalty
            )

    # temperature ≤ 0 → greedy: 与文档声明的语义一致 (其他调用方特判行为统一到此)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


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
    """KV Cache 自回归生成，逐 token yield。遇 stop_ids 或达 max_new_tokens 停止。"""
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
            logits, kv_list, *_ = model(input_ids, past_kv_list=past_kv)  # type: ignore[arg-type]
    else:
        logits, kv_list, *_ = model(input_ids, past_kv_list=past_kv)  # type: ignore[arg-type]
    return logits, kv_list
