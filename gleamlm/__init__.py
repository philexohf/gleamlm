from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .models.model import GleamLMModel
from .utils.config import extract_checkpoint_config

try:
    __version__ = version("gleamlm")
except PackageNotFoundError:
    __version__ = "0.0.0"


def _model_from_config(config: dict) -> GleamLMModel:
    """从 checkpoint 提取的 config 重建 GleamLMModel。

    extract_checkpoint_config 返回的 config 含 attn_type/ffn_type（字符串），
    但 GleamLMModel 接受 attn_variant/ffn_variant（类）。这里做字符串→类的
    映射（与 manual/pretrain.py 的 ATTN_REGISTRY 一致），其余架构参数透传。
    """
    from .models.attention_variants import AliBiGQA, NoPEGQA, SlidingWindowGQA
    from .models.model import GQA, MLP, MoE

    attn_registry = {"gqa": GQA, "nope": NoPEGQA, "alibi": AliBiGQA, "sliding": SlidingWindowGQA}
    ffn_registry = {"mlp": MLP, "moe": MoE}

    config = dict(config)
    attn_type = config.pop("attn_type", "gqa")
    ffn_type = config.pop("ffn_type", "mlp")
    return GleamLMModel(
        **config,
        attn_variant=attn_registry.get(attn_type, GQA),
        ffn_variant=ffn_registry.get(ffn_type, MLP),
    )


def load_model_for_inference(
    model_path: str, device: str = "cuda", checkpoint: dict | None = None
) -> tuple[GleamLMModel, dict]:
    import torch

    if checkpoint is None:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

    config = extract_checkpoint_config(checkpoint)

    if "args" in checkpoint:
        tokenizer_path = getattr(checkpoint["args"], "tokenizer_path", None)
    elif "config" in checkpoint:
        tokenizer_path = checkpoint["config"].get("tokenizer_path", None)
    else:
        tokenizer_path = None

    config["dropout"] = 0.0

    model = _model_from_config(config).to(device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        from gleamlm.utils.torch_utils import clean_state_dict

        state_dict = clean_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys in checkpoint: {unexpected}")

    model.eval()

    if checkpoint.get("dtype") == "float16":
        model = model.half()

    if not tokenizer_path:
        from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

        tokenizer_path = DEFAULT_TOKENIZER_PATH
    config["tokenizer_path"] = tokenizer_path

    return model, config
