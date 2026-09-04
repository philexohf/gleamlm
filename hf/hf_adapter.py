"""
HuggingFace tokenizers 适配器。

用法:
  tokenizer = HFBBPETokenizer.load("checkpoints/hf_tokenizer")
  ids = tokenizer.encode("你好", add_bos=True)
"""

from __future__ import annotations

import json
import os
from typing import Any


class HFBBPETokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._has_transformers_wrapper = "transformers" in type(tokenizer).__module__ if hasattr(type(tokenizer), "__module__") else False
        self._special_ids = self._extract_special_ids()

    def _extract_special_ids(self) -> dict[str, int]:
        ids: dict[str, int] = {}
        if self._has_transformers_wrapper:
            ids["<|endoftext|>"] = self._tokenizer.convert_tokens_to_ids("<|endoftext|>") or 0
            ids["<|im_start|>"] = self._tokenizer.convert_tokens_to_ids("<|im_start|>") or 1
            ids["<|im_end|>"] = self._tokenizer.convert_tokens_to_ids("<|im_end|>") or 2
            return ids
        try:
            for tid, token_data in self._tokenizer.added_tokens_decoder.items():
                if isinstance(token_data, str):
                    ids[token_data] = tid
                elif hasattr(token_data, "content"):
                    ids[token_data.content] = tid
        except Exception:
            pass
        if not ids:
            vocab = getattr(self._tokenizer, "get_vocab", lambda: {})()
            for name in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
                if name in vocab:
                    ids[name] = vocab[name]
        return ids

    @property
    def pad_id(self) -> int:
        return self._special_ids.get("<|endoftext|>", 0)

    @property
    def bos_id(self) -> int:
        return self._special_ids.get("<|im_start|>", 1)

    @property
    def eos_id(self) -> int:
        return self._special_ids.get("<|im_end|>", 2)

    @property
    def im_start_id(self) -> int:
        return self._special_ids.get("<|im_start|>", 1)

    @property
    def im_end_id(self) -> int:
        return self._special_ids.get("<|im_end|>", 2)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if self._has_transformers_wrapper:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
        else:
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
            ids = encoding.ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if self._has_transformers_wrapper:
            return self._tokenizer.decode(ids, skip_special_tokens=skip_special)
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special)

    def get_vocab_size(self) -> int:
        if self._has_transformers_wrapper:
            return self._tokenizer.vocab_size
        return self._tokenizer.get_vocab_size()

    @classmethod
    def load(cls, save_dir: str) -> HFBBPETokenizer:
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError:
            raise ImportError("需要安装 HuggingFace tokenizers: pip install tokenizers")

        json_path = os.path.join(save_dir, "tokenizer.json")
        legacy_path = os.path.join(save_dir, "bbpe_tokenizer.json")

        if os.path.exists(json_path):
            hf_tok = HFTokenizer.from_file(json_path)
        elif os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"发现 bbpe_tokenizer.json（原生格式）。"
                f"请先用 gleamlm/tokenizer 下的工具转换为 HF 格式。"
            )
        else:
            raise FileNotFoundError(f"未找到 tokenizer 文件。需要 {json_path}（HF 格式）")

        return cls(hf_tok)
