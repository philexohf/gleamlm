from __future__ import annotations

from .models.model import GleamLMModel


def load_model_for_inference(
    model_path: str, device: str = "cuda", checkpoint: dict | None = None
) -> tuple[GleamLMModel, dict]:
    """从 checkpoint 加载模型用于推理/评估。"""
    import torch

    if checkpoint is None:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

    if "args" in checkpoint:
        args = checkpoint["args"]
        config = {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_kv_heads": getattr(args, "num_kv_heads", args.num_heads),
            "d_ff": args.d_ff,
            "dropout": 0.0,
            "max_seq_len": args.max_seq_len,
            "pad_token_id": getattr(args, "pad_token_id", 0),
            "tie_weights": getattr(args, "tie_weights", True),
            "use_flash_attn": getattr(args, "use_flash_attn", False),
        }
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError(
            "Checkpoint 缺少模型结构信息。请确保 checkpoint 包含 'args' 或 'config' 字段。"
        )

    config["dropout"] = 0.0

    model = GleamLMModel(**config).to(device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys in checkpoint: {unexpected}")

    model.eval()

    if checkpoint.get("dtype") == "float16":
        model = model.half()

    return model, config
