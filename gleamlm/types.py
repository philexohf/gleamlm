"""GleamLM 共享类型别名"""

from __future__ import annotations

import torch

PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValueList = list[PastKeyValue]
