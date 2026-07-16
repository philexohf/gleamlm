"""全流程冒烟测试 — 参数化覆盖 nano/lite/pro 三个变体。

验证: 数据加载 → 模型初始化 → 前向/反向 → 优化器步进 → loss 下降 → checkpoint 保存/加载
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.training.base_trainer import set_seed
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 12002

VARIANT_CFG = {
    "nano": {
        "d_model": 512,
        "num_layers": 2,
        "num_heads": 8,
        "num_kv_heads": 4,
        "d_ff": 1365,
        "max_seq_len": 64,
        "dropout": 0.0,
        "use_flash_attn": False,
    },
    "lite": {
        "d_model": 768,
        "num_layers": 2,
        "num_heads": 12,
        "num_kv_heads": 6,
        "d_ff": 2048,
        "max_seq_len": 64,
        "dropout": 0.0,
        "use_flash_attn": False,
    },
    "pro": {
        "d_model": 768,
        "num_layers": 2,
        "num_heads": 12,
        "num_kv_heads": 6,
        "d_ff": 2048,
        "max_seq_len": 64,
        "dropout": 0.0,
        "use_flash_attn": False,
    },
}


@pytest.mark.parametrize("variant", ["nano", "lite", "pro"])
def test_smoke_train(variant):
    """全流程冒烟测试：数据 → 模型 → 训练 → loss 下降 → checkpoint"""
    cfg = VARIANT_CFG[variant]
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
    assert tokenizer.get_vocab_size() == VOCAB_SIZE

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        train_txt = os.path.join(tmpdir, "train.txt")
        val_txt = os.path.join(tmpdir, "valid.txt")
        lines = ["这是一个用于冒烟测试的中文句子。深度学习是人工智能的重要分支。\n"] * 200
        with open(train_txt, "w", encoding="utf-8") as f:
            f.writelines(lines)
        with open(val_txt, "w", encoding="utf-8") as f:
            f.writelines(lines[:20])

        train_ds = LMDataset(tmpdir, tokenizer, cfg["max_seq_len"], "train")
        assert len(train_ds) > 0

        model = GleamLMModel(
            vocab_size=VOCAB_SIZE,
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
            tie_weights=True,
            use_flash_attn=cfg["use_flash_attn"],
            pad_token_id=tokenizer.pad_id,
        ).to(device)
        total, _ = model.get_num_params()
        assert total > 0

        dataloader = DataLoader(
            train_ds,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        model.train()
        losses = []
        for step, (input_ids, target_ids) in enumerate(dataloader):
            if step >= 20:
                break
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        assert len(losses) >= 5
        # Loss should decrease over the first 20 steps on average
        first_half = sum(losses[: len(losses) // 2]) / max(1, len(losses) // 2)
        second_half = sum(losses[len(losses) // 2 :]) / max(1, len(losses) - len(losses) // 2)
        assert second_half < first_half, (
            f"Loss not decreasing: {first_half:.2f} → {second_half:.2f}"
        )

        del train_ds, dataloader

        # Checkpoint save / load
        ckpt_dir = os.path.join(tmpdir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "test.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        model2 = GleamLMModel(
            vocab_size=VOCAB_SIZE,
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
            tie_weights=True,
            use_flash_attn=cfg["use_flash_attn"],
            pad_token_id=tokenizer.pad_id,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"])

        x = torch.randint(0, VOCAB_SIZE, (1, 32)).to(device)
        with torch.no_grad():
            logits1, _ = model(x)
            logits2, _ = model2(x)
        assert torch.allclose(logits1, logits2, atol=1e-5), "Checkpoint reload mismatch"
