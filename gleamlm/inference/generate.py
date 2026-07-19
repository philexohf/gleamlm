"""共享的生成工具函数，供 SFT/DPO 评估等模块复用"""

from __future__ import annotations

import torch

from gleamlm.inference.chatml import format_chatml
from gleamlm.inference.generator import generate_tokens
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer


@torch.no_grad()
def generate_response(
    model: GleamLMModel,
    tokenizer: BBPETokenizer,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> str:
    """Generate a chat response with ChatML formatting. Stops at <|im_end|> or <|endoftext|>."""
    device = next(model.parameters()).device

    prompt_text = format_chatml(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)

    stop_ids = {
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.im_end_id,
    }
    stop_ids.discard(None)
    assert None not in stop_ids  # type guard for mypy

    generated_tokens: list[int] = []
    for i, token_id in enumerate(
        generate_tokens(
            model,
            prompt_ids,
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_ids=stop_ids,  # type: ignore[arg-type]
        )
    ):
        generated_tokens.append(token_id)
        if (i + 1) % 4 == 0:
            draft = tokenizer.decode(generated_tokens, skip_special=False)
            if tokenizer.eos_token and tokenizer.eos_token in draft:
                break

    response = tokenizer.decode(generated_tokens, skip_special=False)
    if tokenizer.eos_token and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response
