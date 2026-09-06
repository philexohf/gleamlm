"""LoRA 参数高效微调: W' = W + B·A，冻结 W 只训练 B/A，推理时合并回 W。

微调时 ΔW 的有效秩很低，可用低秩矩阵 BA 近似 (r << min(d,k))。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn


# lora_alpha/r 缩放: BA 初始≈0，alpha/r 让更新幅度不随 r 变化，换 r 无需重调学习率。
# target_modules 默认 Q/K/V/O：加 FFN 层提升容量但显存线性增长。
# fan_in_fan_out 兼容 GPT-2 的 fan-in 权重初始化（影响 merge 的矩阵排列）。
@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["W_q", "W_k", "W_v", "W_o"])
    lora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: str = "none"  # "none" | "all" | "lora_only"

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r


class LoraLinear(nn.Module):
    """LoRA 旁路 + 原始 Linear 的组合模块。"""

    def __init__(self, base: nn.Linear, config: LoraConfig):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        in_features, out_features = base.weight.shape[1], base.weight.shape[0]
        self.r = min(config.r, in_features, out_features)  # r 不能超过矩阵维度
        # scaling 必须用 clamp 后的真实 rank，否则 alpha/r 与 BA 的实际维度不一致
        self.scaling = config.lora_alpha / self.r

        # A: d_in→r, B: r→d_out；输出 = Wx + (B·A·x) × scaling
        self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()

        # B 全零保证第一轮输出 = Wx（不偏离预训练权重）；A 非零保证 A 有梯度
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = cast(torch.Tensor, self.base(x))
        lora = cast(torch.Tensor, self.dropout(x)) @ self.lora_A.T @ self.lora_B.T
        return result + lora * self.scaling

    # merge 把 BA 合回 W: W' = W + (alpha/r)·BA，推理时零额外计算
    def merge(self) -> nn.Linear:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        merged = nn.Linear(
            self.base.in_features, self.base.out_features, bias=self.base.bias is not None
        )
        merged.weight = nn.Parameter(self.base.weight.data + delta.to(self.base.weight.device))
        if self.base.bias is not None:
            merged.bias = nn.Parameter(self.base.bias.data.clone())
        return merged


# named_modules() 返回扁平路径，需映射回 parent + child_name 再 setattr 替换
def _match_module(name: str, target_patterns: list[str]) -> bool:
    return any(re.search(pat, name) for pat in target_patterns)


def apply_lora_to_model(
    model: nn.Module,
    config: LoraConfig | None = None,
) -> list[tuple[str, LoraLinear]]:
    config = config or LoraConfig()
    replaced = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _match_module(name, config.target_modules):
            continue
        parent = _get_parent(model, name)
        child_name = name.rsplit(".", 1)[-1]
        lora_linear = LoraLinear(module, config)
        # 新 LoRA 参数跟随 base 权重所在设备/类型（防 CPU/CUDA 混用）
        if module.weight is not None:
            lora_linear.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, child_name, lora_linear)
        replaced.append((name, lora_linear))
    return replaced


def _get_parent(model: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj


def merge_lora_weights(model: nn.Module, replace: bool = True) -> list[tuple[str, nn.Linear]]:
    merged = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        parent = _get_parent(model, name)
        child_name = name.rsplit(".", 1)[-1]
        merged_linear = module.merge()
        setattr(parent, child_name, merged_linear if replace else module)
        merged.append((name, merged_linear))
    return merged


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
