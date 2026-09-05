"""GleamLM API — convenience entry point for inference."""

import argparse
import os

import torch

from .hf_adapter import HFBBPETokenizer
from .hf_config import gleamlm_config_from_core
from .hf_model import GleamLMForCausalLM, load_from_checkpoint
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config

# PyTorch ≥2.6 的 weights_only 默认白名单不含 argparse.Namespace，
# 而训练产物（sft/dpo/opd 等）嵌有 "args" 元数据（Namespace）→ 显式放行
torch.serialization.add_safe_globals([argparse.Namespace])


class GleamLM:
    def __init__(self, model, tokenizer, device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="auto", tokenizer_path=None):
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cfg = extract_checkpoint_config(ckpt)
        hf_config = gleamlm_config_from_core(cfg)
        model = GleamLMForCausalLM(hf_config)
        missing, unexpected = load_from_checkpoint(model, ckpt, strict=True)
        if missing or unexpected:
            print(f"[warn] api load — missing={len(missing)} unexpected={len(unexpected)}")
        model.to(device)
        model.eval()

        tokenizer = cls._load_tokenizer(checkpoint_path, tokenizer_path)

        total = sum(p.numel() for p in model.parameters())
        print(f"Model: {total / 1e6:.2f}M params, device: {device}")
        return cls(model, tokenizer, device)

    @staticmethod
    def _load_tokenizer(checkpoint_path, tokenizer_path):
        """加载 tokenizer；未显式指定时按常见位置自动推断。"""
        candidates = []
        if tokenizer_path:
            candidates.append(tokenizer_path)
        else:
            candidates.append(DEFAULT_TOKENIZER_PATH)
            ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            candidates.append(os.path.join(ckpt_dir, "tokenizer"))
            candidates.append(os.path.join(ckpt_dir, "bbpe_12k"))
            # 若 checkpoint 在 checkpoints/nano/step_xxx.pt，尝试 checkpoints/nano/tokenizer
            candidates.append(os.path.join(os.path.dirname(ckpt_dir), "tokenizer"))
            candidates.append(os.path.join(os.path.dirname(ckpt_dir), "bbpe_12k"))

        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                # HF tokenizer.json 优先（HF 格式更通用）
                if os.path.exists(os.path.join(path, "tokenizer.json")):
                    return HFBBPETokenizer.load(path)
                # 原生 BBPE 格式
                if os.path.exists(os.path.join(path, "bbpe_tokenizer.json")):
                    return BBPETokenizer.load(path)
            except Exception as exc:
                print(f"[warn] failed to load tokenizer from {path}: {exc}")
                continue

        print("[warn] no tokenizer found; generate() will fail. "
              "Provide tokenizer_path or place tokenizer next to checkpoint.")
        return None

    @staticmethod
    def _get_pad_id(tokenizer):
        if isinstance(tokenizer, BBPETokenizer):
            return tokenizer.pad_id
        return getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "pad_id", 0)

    @staticmethod
    def _get_eos_id(tokenizer):
        if isinstance(tokenizer, BBPETokenizer):
            return tokenizer.eos_id
        return getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "eos_id", 2)

    @torch.no_grad()
    def generate(self, prompt, *, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9):
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt)}")
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer available")

        # 两种 tokenizer（原生 BBPE / HF tokenizers 适配器）都实现 encode/decode，
        # 统一走 encode 避免 HFBBPETokenizer 不可调用的问题
        input_ids = self._tokenizer.encode(prompt, add_bos=False, add_eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self._device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._get_pad_id(self._tokenizer),
            "eos_token_id": self._get_eos_id(self._tokenizer),
        }
        do_sample = temperature > 0
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        out = self._model.generate(input_ids, **gen_kwargs)
        # HFBBPETokenizer 的 decode 参数名是 skip_special（不是 HF 的 skip_special_tokens）
        return self._tokenizer.decode(out[0].tolist(), skip_special=True)
