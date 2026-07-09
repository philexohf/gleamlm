"""Flash Attention vs 标准 Attention 对比测试
运行 100 步训练，对比 loss 曲线的一致性。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(use_flash_attn):
    return GleamLMModel(
        vocab_size=12002,
        d_model=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=6,
        d_ff=2048,
        max_seq_len=2048,
        dropout=0.0,
        pad_token_id=259,
        tie_weights=True,
        use_flash_attn=use_flash_attn,
    )


class TinyDataset(Dataset):
    """从 npy 加载前 N 个 token，构建固定 slice 的种子化数据集"""

    def __init__(self, ids_path, max_tokens=8_000_000, seq_len=256):
        self.seq_len = seq_len
        all_ids = np.load(ids_path, mmap_mode="r")
        n = min(len(all_ids), max_tokens)
        self.data = torch.from_numpy(all_ids[:n].astype(np.int64))
        self.num_samples = max(0, (len(self.data) - seq_len - 1) // (seq_len // 2) + 1)
        self.num_samples = min(
            self.num_samples, 1600
        )  # 足够 100 步 (batch=2, accum=16 => 32 steps per 1000 samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len // 2)
        start = min(start, len(self.data) - self.seq_len - 1)
        return self.data[start : start + self.seq_len + 1]


def collate(batch, pad_id):
    max_len = max(len(s) for s in batch)
    padded = []
    for s in batch:
        if len(s) < max_len:
            s = torch.cat([s, torch.full((max_len - len(s),), pad_id, dtype=torch.long)])
        padded.append(s)
    x = torch.stack(padded)
    return x[:, :-1], x[:, 1:]


def train_100_steps(use_flash, device, dataset, batch_size=2, accum=16):
    set_seed(42)
    model = build_model(use_flash).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu", enabled=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate(b, pad_id=259)
    )

    losses = []
    model.train()
    step = 0
    opt.zero_grad()

    for batch_idx, (inp, tgt) in enumerate(loader):
        inp, tgt = inp.to(device), tgt.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
            loss = criterion(logits.view(-1, 12002), tgt.view(-1))

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            step += 1
            losses.append(loss.item())

            if step >= 100:
                break

    return losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载分词器确认路径
    tok = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
    pids = tok.pad_id
    print(f"Tokenizer loaded, pad_id={pids}, vocab={tok.get_vocab_size()}")

    # 使用 lite 训练数据的前 2M tokens，seq=256（比 2048 快很多）
    ids_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "lite_data", "train_ids.npy"
    )
    if not os.path.exists(ids_path):
        print(f"[ERROR] train_ids.npy not found at {ids_path}")
        return

    dataset = TinyDataset(ids_path, max_tokens=2_000_000, seq_len=256)
    print(f"Tiny dataset: {len(dataset)} samples (from train_ids.npy), seq_len=256")

    # Flash Attention
    print("\n=== Flash Attention: 100 steps ===")
    flash_losses = train_100_steps(True, device, dataset)
    for i, l in enumerate(flash_losses):
        if i % 10 == 0:
            print(f"  step {i:4d}: loss={l:.4f}")

    # Non-Flash Attention
    print("\n=== Non-Flash Attention: 100 steps ===")
    nonflash_losses = train_100_steps(False, device, dataset)
    for i, l in enumerate(nonflash_losses):
        if i % 10 == 0:
            print(f"  step {i:4d}: loss={l:.4f}")

    # 对比
    print("\n=== 对比结果 ===")
    max_diff = 0.0
    max_diff_step = 0
    n_steps = min(len(flash_losses), len(nonflash_losses))
    for i in range(n_steps):
        diff = abs(flash_losses[i] - nonflash_losses[i])
        diff_pct = diff / max(abs(flash_losses[i]), 1e-8) * 100
        if diff > max_diff:
            max_diff = diff
            max_diff_step = i
        if i < 5 or i % 20 == 0:
            print(
                f"  step {i:4d}: flash={flash_losses[i]:.4f}  noflash={nonflash_losses[i]:.4f}  diff={diff:.6f} ({diff_pct:.2f}%)"
            )

    print(f"\n最大偏差: step {max_diff_step}, diff={max_diff:.6f}")
    if max_diff < 0.05 or (max_diff / max(abs(flash_losses[max_diff_step]), 1e-8) * 100 < 1):
        print("结论: Flash 与非 Flash 路径一致 ✓")
    elif max_diff / max(abs(flash_losses[max_diff_step]), 1e-8) * 100 < 5:
        print("结论: 偏差在可接受范围内 (<5%) ✓")
    else:
        print("结论: 偏差较大，需要排查 ✗")


if __name__ == "__main__":
    main()
