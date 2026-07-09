"""快速查看 checkpoint 信息。
用法: python scripts/check_ckpt.py [--path checkpoints/best_model.pt]
"""

import argparse

import torch

parser = argparse.ArgumentParser(description="查看 checkpoint 信息")
parser.add_argument(
    "--path", type=str, default="checkpoints/best_model.pt", help="checkpoint 文件路径"
)
args = parser.parse_args()

ckpt = torch.load(args.path, map_location="cpu", weights_only=False)
print("Keys:", list(ckpt.keys()))
if "args" in ckpt:
    a = ckpt["args"]
    print(
        f"d_model={a.d_model}, layers={a.num_layers}, heads={a.num_heads}, kv_heads={a.num_kv_heads}, d_ff={a.d_ff}, vocab={a.vocab_size}"
    )
if "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
    for k, v in sd.items():
        if "embed" in k or "lm_head" in k:
            print(f"{k}: {v.shape}")
        if "layers.0" in k and "weight" in k:
            print(f"{k}: {v.shape}")
    print(f"dtype: {sd[list(sd.keys())[0]].dtype}")
print(f"epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')}, val_ppl={ckpt.get('val_ppl')}")
print(f"dtype in meta: {ckpt.get('dtype')}")
