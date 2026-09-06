"""AMP (Automatic Mixed Precision) context manager.

Provides safe_autocast — cross-cutting utility used by both training
and inference code. LR schedulers have been moved to trainer/schedulers.py.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch


def clean_state_dict(state_dict: dict) -> dict:
    """规整 checkpoint state_dict 键名：剥离 torch.compile/_orig_mod、DDP/module 前缀。

    torch.compile 会给被编译 module 的键加 `_orig_mod.` 前缀，DDP 加 `module.`，
    加载到原始模型前需剥离。这是所有下游脚本（sft/dpo/grpo/infer/eval）加载
    训练产物时的统一入口，避免每个脚本重复处理。
    """
    sd = state_dict
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    if any(k.startswith("model.") for k in sd):
        # HF wrapper 产物常见前缀 (旧 gleamlm_hf 格式)
        sd = {k[len("model.") :]: v for k, v in sd.items()}
    return sd


@contextmanager
def safe_autocast(
    enabled: bool = True, *, dtype: torch.dtype = torch.bfloat16
) -> Generator[None, None, None]:
    """Safe autocast context manager with automatic backend selection."""
    if not enabled:
        yield
        return

    if torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=dtype):  # type: ignore[attr-defined]
            yield
        return

    if (
        dtype == torch.bfloat16
        and hasattr(torch, "cpu")
        and callable(getattr(torch.cpu, "is_bf16_supported", None))
        and torch.cpu.is_bf16_supported()  # type: ignore[attr-defined]
    ):
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):  # type: ignore[attr-defined]
            yield
        return

    yield
