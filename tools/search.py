"""
超参搜索 — Grid / Random search。

用法:
  space = {
      "lr": [1e-4, 3e-4, 1e-3],
      "batch_size": [4, 8],
      "wd": {"type": "log_uniform", "min": 0.01, "max": 0.1},
  }
  configs = hyperparameter_search(space, method="grid", n_trials=10)
  # 返回 [{"lr": ..., "batch_size": ..., ...}, ...]
  # 每个 config 可以直接传给 pretrain.py 的 --config 或命令行参数
"""

import copy
import itertools
import random
from typing import Any


def _sample_uniform(min_v: float, max_v: float) -> float:
    return random.uniform(min_v, max_v)


def _sample_log_uniform(min_v: float, max_v: float) -> float:
    log_min, log_max = __import__("math").log(min_v), __import__("math").log(max_v)
    return __import__("math").exp(random.uniform(log_min, log_max))


def _sample_categorical(options: list) -> Any:
    return random.choice(options)


def _sample_int(min_v: int, max_v: int) -> int:
    return random.randint(min_v, max_v)


def _sample_from_spec(spec: Any) -> Any:
    if isinstance(spec, list):
        return random.choice(spec)
    if isinstance(spec, dict):
        method = spec.get("type", "uniform")
        if method == "uniform":
            return _sample_uniform(spec["min"], spec["max"])
        if method == "log_uniform":
            return _sample_log_uniform(spec["min"], spec["max"])
        if method == "int":
            return _sample_int(spec["min"], spec["max"])
        if method == "choice":
            return random.choice(spec["options"])
    return spec


def _expand_grid(space: dict[str, Any]) -> list[dict[str, Any]]:
    keys, values = zip(*space.items())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def hyperparameter_search(
    space: dict[str, Any],
    method: str = "random",
    n_trials: int = 20,
    base_config: dict | None = None,
) -> list[dict[str, Any]]:
    if method == "grid":
        configs = _expand_grid(space)
    elif method == "random":
        configs = []
        for _ in range(n_trials):
            config = {}
            for key, spec in space.items():
                config[key] = _sample_from_spec(spec)
            configs.append(config)
    else:
        raise ValueError(f"Unknown method: {method} (use 'grid' or 'random')")

    if base_config:
        configs = [{**base_config, **c} for c in configs]

    for i, cfg in enumerate(configs):
        cfg.setdefault("run_name", f"trial_{i:03d}")
    return configs
