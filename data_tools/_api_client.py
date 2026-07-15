"""共享 DeepSeek API 客户端 — 提取自 4 个数据管线脚本。"""

from __future__ import annotations

import os
import time

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def get_client(api_key: str | None = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""), base_url=base_url)


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_retries: int = 3,
    extra_kwargs: dict | None = None,
) -> str | None:
    extra = extra_kwargs or {}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "insufficient" in err or "balance" in err:
                print("  余额不足，请充值。")
                return None
            if "content" in err and "risk" in err:
                return None
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = 5 * (2**attempt) if ("429" in err or "rate" in err) else 2**attempt
                time.sleep(delay)
    return None
