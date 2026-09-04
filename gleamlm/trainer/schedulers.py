"""Learning rate schedulers — training-only utilities.

Provides get_lr_cosine and get_lr_wsd (Warmup-Stable-Decay).

WSD vs Cosine:
  Cosine: warmup → smooth cosine decay from LR_max to 0
  WSD:    warmup → stable (LR_max) → decay (LR_max → 0)
  WSD advantage: in the stable phase you can stop training at any point
  and lose <1% quality (Chinchilla engineering practice). Cosine has
  no stable phase — every step reduces LR, so early stopping is costly.
"""

from __future__ import annotations

import math


def get_lr_cosine(
    step: int, total_steps: int, warmup_ratio: float = 0.01, min_lr_ratio: float = 0.1
) -> float:
    """Cosine annealing with warmup. Returns multiplier in [0, 1]."""
    warmup_steps = int(total_steps * warmup_ratio)

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    else:
        # +1 使最后一步 (step=total_steps-1) 正好 progress=1 → cos(π) → min_lr_ratio
        progress = (step - warmup_steps + 1) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def get_lr_wsd(
    step: int,
    total_steps: int,
    warmup_ratio: float = 0.02,
    stable_ratio: float = 0.80,
    min_lr_ratio: float = 0.05,
    decay_style: str = "linear",
) -> float:
    """WSD (Warmup-Stable-Decay) scheduler.

    三段式: warmup (0→LR_max) → stable (LR_max) → decay (LR_max→0)。
    对比 Cosine: WSD 在 stable 段做绝大部分训练，decay 只占 5-10% 步数。
    效果: 在 decay 前随时中断训练，损失 <1% 质量 (Chinchilla 工程实践)。

    decay_style:
      - linear:  线性衰减到 min_lr_ratio（默认，对齐 nano 实际 nano_wsd_linear_v2
                 / SmolLM3: config_smollm3 lr_decay_style: linear，min_decay_lr=0）
      - cosine:  余弦衰减到 min_lr_ratio（平滑）
    """
    warmup_steps = int(total_steps * warmup_ratio)
    stable_steps = int(total_steps * stable_ratio)
    decay_steps = total_steps - warmup_steps - stable_steps

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < warmup_steps + stable_steps:
        return 1.0
    elif decay_steps <= 0:
        # warmup+stable 已覆盖全部步数: 无 decay 段，稳定期结束后直接落到 min_lr
        return min_lr_ratio
    else:
        progress = (step - warmup_steps - stable_steps + 1) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        if decay_style == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        # cosine
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
