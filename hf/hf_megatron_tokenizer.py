"""MegatronTokenizerBase 适配器 — 把项目 BBPETokenizer 暴露为 megatron-core 可消费的 tokenizer。

megatron 的 GPTDataset/GPTDatasetConfig 硬性要求 `tokenizer` 对象提供
`vocab_size` / `eod` / `pad` / `special_tokens_dict`（详见
megatron/core/datasets/gpt_dataset.py 与 megatron_dataset.py）。项目自研
BBPETokenizer 有 pad_id(0)/eos_id(2)/get_vocab_size()，但命名与 megatron
约定不同。本适配器补齐接口，使工业轨数据构建可用官方 GPTDataset 类。
"""

from __future__ import annotations

from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase

from gleamlm.tokenizer.tokenizer import BBPETokenizer


class MegatronBBPETokenizer(MegatronTokenizerBase):
    """把 GleamLM 的 BBPETokenizer 包装成 MegatronTokenizerBase 子类。

    仅用于数据构建（vocab_size/eod/pad 属性），不承担真实编码——
    .bin/.idx 数据已离线 tokenize，GPTDataset 不会调用 tokenize()。
    """

    def __init__(self, tokenizer: BBPETokenizer, config: dict | None = None) -> None:
        super().__init__(path="", config=config or {})
        self._tok = tokenizer
        # megatron 约定: eod = end-of-document token。
        # 项目约定: id0=pad/unk 仅用于 batch 内样本长度对齐；预训练文档边界与
        # SFT 终止符共用 eos = <|im_end|>(2)，.bin 每文档以 2 收尾（与现有语料一致）。
        self.eod = tokenizer.eos_id
        self.pad = tokenizer.pad_id
        self.eos = tokenizer.eos_id
        self.bos = tokenizer.bos_id
        # megatron 检查 pad 是否与其他特殊 token 冲突（_PAD_TOKEN_ID 兜底）
        self.special_tokens_dict = {
            k: v for k, v in tokenizer.special_tokens.items()
        }

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def unique_identifiers(self) -> dict:
        """megatron 缓存 key 序列化 tokenizer 时需要；空即可（数据已离线 tokenize）。"""
        return {"class": "MegatronBBPETokenizer", "vocab_size": self.vocab_size}

    def vocab(self) -> dict:
        """返回词表（id→token 映射）；GPTDataset 未用，保持接口完整。"""
        return {}

    def tokenize(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)

    def apply_chat_template(self, *args, **kwargs) -> str:
        return ""
