"""共享的生成工具函数，供 SFT/DPO 评估等模块复用"""

from __future__ import annotations

import torch

from gleamlm.inference.sampler import sample_token
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
    """生成对话回复，遇到 <|im_end|> 或 <|endoftext|> 自动截断"""
    model.eval()
    device = next(model.parameters()).device

    prompt_text = f"<|im_start|><|user|>\n{instruction}<|im_end|>\n<|im_start|><|assistant|>\n"
    prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    generated_ids = prompt_ids.copy()

    stop_ids = {
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.special_tokens.get("<|im_end|>"),
    }
    stop_ids.discard(None)

    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.amp.autocast(amp_device):  # type: ignore[attr-defined]
        logits, past_kv = model(prompt_tensor)

    for i in range(max_new_tokens):
        next_token = sample_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids,
        )
        token_id = next_token.item()

        if token_id in stop_ids:
            break

        generated_ids.append(token_id)

        if (i + 1) % 4 == 0:
            draft = tokenizer.decode(generated_ids[len(prompt_ids) :], skip_special=False)
            if "<|endoftext|>" in draft:
                break

        next_input = torch.tensor([[token_id]], dtype=torch.long).to(device)
        with torch.amp.autocast(amp_device):  # type: ignore[attr-defined]
            logits, past_kv = model(next_input, past_kv_list=past_kv)

    response = tokenizer.decode(generated_ids[len(prompt_ids) :], skip_special=False)
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]
    return response
