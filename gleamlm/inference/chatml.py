"""ChatML 格式化工具。统一所有模板调用点，方便未来切换格式（Qwen / Llama 等）。"""

from __future__ import annotations


def format_chatml(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)
