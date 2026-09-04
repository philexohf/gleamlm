"""
RAGPipeline — 检索增强生成完整流程。

流程:
  Query → Retriever → Top-K Chunks → Prompt Assembly → Generator → Answer
"""

from __future__ import annotations

from typing import Optional

import torch

from gleamlm.rag.retriever import SimpleRetriever
from gleamlm.tokenizer.tokenizer import BBPETokenizer
# 注意: 不 import hf.hf_model 的 GleamLMForCausalLM —— 它只出现在
# 类型注解里 (from __future__ import annotations 下为惰性字符串, 不求值),
# 且 hf 依赖 core, core 反向 import 会形成循环依赖。


SYSTEM_PROMPT = "基于以下资料回答问题。如果资料不足以回答，请说'资料不足'。"


def _format_prompt(query: str, chunks: list[str]) -> str:
    ctx = "\n---\n".join(chunks)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n资料:\n{ctx}\n\n问题: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class RAGPipeline:
    def __init__(
        self,
        model: GleamLMForCausalLM,
        tokenizer: BBPETokenizer,
        device: torch.device,
        retriever: Optional[SimpleRetriever] = None,
    ):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.retriever = retriever or SimpleRetriever()

    def add_knowledge(self, docs: list[str], chunk_size: int = 512):
        self.retriever.add_documents(docs, chunk=True, chunk_size=chunk_size)

    @torch.no_grad()
    def query(
        self, question: str, top_k: int = 3, max_new_tokens: int = 256,
        temperature: float = 0.8, top_k_sampling: int = 50,
    ) -> dict:
        chunks = self.retriever.retrieve(question, top_k=top_k)
        prompt = _format_prompt(question, [c for c, _ in chunks])
        # prompt 已以 <|im_start|> 开头，无需再 add_bos (避免重复 im_start)
        input_ids = self.tokenizer.encode(prompt, add_bos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits, _, _, _ = self.model.model(generated)
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            if top_k_sampling > 0:
                vals, _ = logits.topk(top_k_sampling, dim=-1)
                logits[logits < vals[:, -1:]] = float("-inf")
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=-1)
            if nxt.item() == self.tokenizer.eos_id:
                break

        answer = self.tokenizer.decode(generated[0].tolist(), skip_special=True)
        return {
            "question": question,
            "answer": answer,
            "chunks": chunks,
        }
