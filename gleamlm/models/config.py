"""GleamLM 全局配置"""

from __future__ import annotations

import argparse


def get_args() -> argparse.Namespace:
    """通过 YAML 配置文件获取命令行参数"""
    parser = argparse.ArgumentParser(description="GleamLM 大模型训练配置")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    config_args, _ = parser.parse_known_args()

    if not config_args.config:
        parser.error("请通过 --config 指定 YAML 配置文件（如 --config configs/nano.yaml）")

    from gleamlm.utils.config import load_config_as_args

    args = load_config_as_args(config_args.config, cli_overrides=True)
    return args
