"""XFIND-LLM 流式文本生成器。KV Cache + 采样，逐 token 输出"""

import torch
import torch.nn.functional as F

from inference.sampler import sample_token


class TextStreamer:
    """流式文本生成器。逐步生成 token，支持实时输出"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, model, prompt_ids, max_new_tokens=256,
                 temperature=1.0, top_k=0, top_p=0.0,
                 repetition_penalty=1.0, eos_id=None):
        if eos_id is None:
            eos_id = self.tokenizer.eos_id

        device = next(model.parameters()).device
        input_ids = prompt_ids.to(device)
        generated_ids = prompt_ids[0].tolist()

        with torch.no_grad():
            logits, past_kv = model(input_ids)
            next_token_logits = logits[:, -1, :]

        next_token = sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids
        )

        for _ in range(max_new_tokens):
            token_id = next_token.item()
            if token_id == eos_id:
                return
            yield token_id
            generated_ids.append(token_id)

            with torch.no_grad():
                logits, past_kv = model(
                    next_token.unsqueeze(0),  # [1, 1]
                    past_kv_list=past_kv
                )
                next_token_logits = logits[:, -1, :]

            next_token = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids
            )

    def generate_text(self, model, prompt, max_new_tokens=256,
                      temperature=1.0, top_k=50, top_p=0.9,
                      repetition_penalty=1.0):
        prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        generated_ids = []
        stopped = False
        for token_id in self.generate(
            model, prompt_tensor, max_new_tokens,
            temperature, top_k, top_p, repetition_penalty
        ):
            generated_ids.append(token_id)
            if stopped:
                continue
            # 每生成几个 token 解码一次
            if len(generated_ids) % 4 == 0:
                text = self.tokenizer.decode(generated_ids, skip_special=True)
                # 遇到文档分隔符则截断
                if "<|endoftext|>" in text:
                    text = text.split("<|endoftext|>")[0]
                    stopped = True
                yield text

        if stopped:
            return

        # 最后解码剩余 token（如果上次 yield 后还有新增）
        if generated_ids and len(generated_ids) % 4 != 0:
            text = self.tokenizer.decode(generated_ids, skip_special=True)
            if "<|endoftext|>" in text:
                text = text.split("<|endoftext|>")[0]
            yield text
