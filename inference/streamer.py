"""
XFIND-LLM 流式文本生成器

封装 KV Cache + 采样策略，提供逐 token 输出的流式生成接口。
"""

import torch
import torch.nn.functional as F

from inference.sampler import sample_token


class TextStreamer:
    """
    流式文本生成器

    逐步生成文本，每生成一个 token 就 yield 出来，
    支持实时输出显示。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, model, prompt_ids, max_new_tokens=256,
                 temperature=1.0, top_k=0, top_p=0.0,
                 repetition_penalty=1.0, eos_id=None):
        """
        流式生成文本

        Args:
            model: XfindModel 实例
            prompt_ids: 输入 token IDs [1, prompt_len]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 参数
            top_p: Top-P 参数
            repetition_penalty: 重复惩罚（>1.0 惩罚已出现 token，1.0=禁用）
            eos_id: 停止 token ID

        Yields:
            token_id: 每个新生成的 token ID
        """
        if eos_id is None:
            eos_id = self.tokenizer.eos_id

        device = next(model.parameters()).device
        input_ids = prompt_ids.to(device)
        generated_ids = prompt_ids[0].tolist()

        # 预填充：处理 prompt，获取 KV Cache
        with torch.no_grad():
            logits, past_kv = model(input_ids)
            next_token_logits = logits[:, -1, :]  # 最后一个位置

        # 采样第一个 token
        next_token = sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids
        )

        # 逐步生成
        for _ in range(max_new_tokens):
            token_id = next_token.item()

            # 遇到 EOS 停止
            if token_id == eos_id:
                return

            yield token_id
            generated_ids.append(token_id)

            # 前向传播（仅处理新 token，复用 KV Cache）
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
        """
        流式生成文本并解码为字符串

        Args:
            model: XfindModel 实例
            prompt: 提示文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 参数
            top_p: Top-P 参数
            repetition_penalty: 重复惩罚（>1.0 惩罚已出现 token，1.0=禁用）

        Yields:
            text_chunk: 每次 yield 一段新生成的文本
        """
        prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        generated_ids = []
        for token_id in self.generate(
            model, prompt_tensor, max_new_tokens,
            temperature, top_k, top_p, repetition_penalty
        ):
            generated_ids.append(token_id)
            # 每生成几个 token 解码一次
            if len(generated_ids) % 4 == 0:
                text = self.tokenizer.decode(generated_ids, skip_special=True)
                yield text

        # 最后解码剩余 token
        if generated_ids:
            text = self.tokenizer.decode(generated_ids, skip_special=True)
            yield text
