"""GleamLM-Lite 全流程冒烟测试
验证: 数据加载 → 模型初始化 → 前向 → 反向 → 优化器步进 → 评估 → checkpoint save/load
"""

import math
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data', 'splits')

# 测试配置：用真实架构但 tiny 数据
TEST_CONFIG = {
    "vocab_size": 12002,
    "d_model": 768,
    "num_layers": 2,       # 只 2 层，跑得快
    "num_heads": 12,
    "num_kv_heads": 6,
    "d_ff": 2048,
    "max_seq_len": 256,    # 短序列
    "dropout": 0.0,
    "tie_weights": True,
    "use_flash_attn": True,
    "batch_size": 4,
    "accumulate_grad": 1,
    "epochs": 2,
    "lr": 4e-4,
    "warmup_ratio": 0.01,
    "min_lr_ratio": 0.1,
    "label_smoothing": 0.0,
    "weight_decay": 0.01,
    "clip_grad": 1.0,
    "z_loss_weight": 1e-4,
    "seed": 42,
    "bf16": True,
}


def make_tiny_data(data_dir):
    """生成 ~500KB 中文语料（足够跑几步训练）"""
    os.makedirs(data_dir, exist_ok=True)
    real_train = os.path.join(DEFAULT_DATA_DIR, 'train.txt')
    if os.path.exists(real_train):
        lines = []
        with open(real_train, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 200: break
                lines.append(line)
    else:
        # 回退：合成语料
        lines = [
            "中国是一个历史悠久、文化丰富的国家。\n",
            "人工智能技术正在改变世界的方方面面。\n",
            "深度学习是机器学习的一个重要分支。\n",
            "Python是数据科学中最流行的编程语言之一。\n",
        ] * 50

    train_path = os.path.join(data_dir, 'train.txt')
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    valid_path = os.path.join(data_dir, 'valid.txt')
    with open(valid_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[:10])
    return train_path, valid_path


def test_full_pipeline():
    print("=" * 60)
    print("GleamLM-Lite 冒烟测试 - 全流程")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    torch.manual_seed(TEST_CONFIG['seed'])
    np.random.seed(TEST_CONFIG['seed'])

    # 临时数据目录
    tmp_dir = tempfile.mkdtemp(prefix='gleamlm_lite_test_')
    data_dir = os.path.join(tmp_dir, 'splits')
    ckpt_dir = os.path.join(tmp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    train_path, valid_path = make_tiny_data(data_dir)
    print(f"数据: {train_path}")

    # 1. 分词器
    print("\n[1] 加载分词器...")
    from gleamlm.tokenizer.tokenizer import BBPETokenizer
    from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH
    tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
    assert len(tokenizer) == 12002, f"预期 12002 tokens, 实际 {len(tokenizer)}"
    print(f"  分词器词表: {len(tokenizer)} [OK]")

    # 2. 数据集
    print("\n[2] 加载数据集...")
    from torch.utils.data import DataLoader

    from gleamlm.dataset.dataset import LMDataset, collate_fn

    train_ds = LMDataset(data_dir, tokenizer, TEST_CONFIG['max_seq_len'], 'train', max_chars=500_000, augment=False)
    valid_ds = LMDataset(data_dir, tokenizer, TEST_CONFIG['max_seq_len'], 'valid', augment=False)
    print(f"  训练样本: {len(train_ds)}, 验证样本: {len(valid_ds)} [OK]")

    train_loader = DataLoader(
        train_ds, batch_size=TEST_CONFIG['batch_size'], shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id), num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=TEST_CONFIG['batch_size'], shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id), num_workers=0
    )

    # 3. 模型初始化
    print("\n[3] 构建模型...")
    from gleamlm.models.model import GleamLMModel
    from gleamlm.utils.torch_utils import safe_autocast

    model = GleamLMModel(
        vocab_size=TEST_CONFIG['vocab_size'],
        d_model=TEST_CONFIG['d_model'],
        num_layers=TEST_CONFIG['num_layers'],
        num_heads=TEST_CONFIG['num_heads'],
        num_kv_heads=TEST_CONFIG['num_kv_heads'],
        d_ff=TEST_CONFIG['d_ff'],
        dropout=TEST_CONFIG['dropout'],
        max_seq_len=TEST_CONFIG['max_seq_len'],
        pad_token_id=tokenizer.pad_id,
        tie_weights=TEST_CONFIG['tie_weights'],
        use_flash_attn=TEST_CONFIG['use_flash_attn'],
    ).to(device)

    total, trainable = model.get_num_params()
    print(f"  参数: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable [OK]")

    # 4. 优化器 + 调度器 + scaler
    print("\n[4] 设置优化器/调度器/scaler...")
    def get_lr_cosine(step, total_steps, warmup_ratio=0.01, min_lr_ratio=0.1):
        """Cosine Annealing + Warmup"""
        warmup_steps = int(total_steps * warmup_ratio)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TEST_CONFIG['lr'],
        betas=(0.9, 0.95), eps=1e-8, weight_decay=TEST_CONFIG['weight_decay']
    )
    total_steps = len(train_loader) * TEST_CONFIG['epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(
            step, total_steps,
            TEST_CONFIG['warmup_ratio'], TEST_CONFIG['min_lr_ratio']
        )
    )
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.cuda.amp.GradScaler()
    print(f"  总步数: {total_steps}, 调度器: Cosine [OK]")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=TEST_CONFIG['label_smoothing'])
    eval_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction='sum')

    # 5. 训练
    print(f"\n[5] 训练 {TEST_CONFIG['epochs']} epoch(s)...")
    losses = []

    for epoch in range(TEST_CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        epoch_steps = 0

        for step, (input_ids, target_ids, attention_mask) in enumerate(train_loader):
            input_ids, target_ids, attention_mask = (
                input_ids.to(device), target_ids.to(device), attention_mask.to(device)
            )
            optimizer.zero_grad()

            with safe_autocast(enabled=TEST_CONFIG['bf16']):
                logits, _ = model(input_ids, attention_mask=attention_mask)
                ce_loss = criterion(
                    logits.view(-1, TEST_CONFIG['vocab_size']),
                    target_ids.view(-1)
                )
                log_z = torch.logsumexp(logits, dim=-1)
                z_loss = TEST_CONFIG['z_loss_weight'] * (log_z ** 2).mean()
                loss = (ce_loss + z_loss) / TEST_CONFIG['accumulate_grad']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TEST_CONFIG['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += ce_loss.item()
            epoch_steps += 1

            if epoch == 0 and step == 0:
                max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
                print(f"  Step 0: loss={ce_loss.item():.4f}, max_grad={max_grad:.4f}")

        avg_loss = epoch_loss / max(1, epoch_steps)
        losses.append(avg_loss)

        # 6. 评估
        print(f"\n[6] 评估 epoch {epoch}...")
        model.eval()
        val_total_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for input_ids, target_ids, attention_mask in valid_loader:
                input_ids, target_ids, attention_mask = (
                    input_ids.to(device), target_ids.to(device), attention_mask.to(device)
                )
                with safe_autocast():
                    logits, _ = model(input_ids, attention_mask=attention_mask)
                loss = eval_criterion(
                    logits.view(-1, TEST_CONFIG['vocab_size']),
                    target_ids.view(-1)
                )
                val_total_loss += loss.item()
                val_tokens += (target_ids != tokenizer.pad_id).sum().item()

        val_avg = val_total_loss / max(1, val_tokens)
        val_ppl = math.exp(val_avg)
        print(f"  Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_avg:.4f}, val_ppl={val_ppl:.2f}")

    # 7. Loss 下降检查
    print(f"\n[7] Loss 趋势: {[f'{l:.4f}' for l in losses]}")
    if len(losses) >= 2:
        assert losses[-1] < losses[0] * 0.95, (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
        print(f"  Loss 下降: [OK]")

    # 8. checkpoint 保存
    print("\n[8] 保存 checkpoint...")
    ckpt_path = os.path.join(ckpt_dir, "test_best.pt")
    torch.save({
        'epoch': TEST_CONFIG['epochs'] - 1,
        'global_step': epoch_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': losses[-1],
        'val_loss': val_avg,
        'val_ppl': val_ppl,
    }, ckpt_path)
    ckpt_size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"  已保存: {ckpt_path} ({ckpt_size_mb:.1f} MB) [OK]")

    # 9. checkpoint 加载验证
    print("\n[9] 重新加载 checkpoint...")
    model2 = GleamLMModel(
        vocab_size=TEST_CONFIG['vocab_size'],
        d_model=TEST_CONFIG['d_model'],
        num_layers=TEST_CONFIG['num_layers'],
        num_heads=TEST_CONFIG['num_heads'],
        num_kv_heads=TEST_CONFIG['num_kv_heads'],
        d_ff=TEST_CONFIG['d_ff'],
        dropout=TEST_CONFIG['dropout'],
        max_seq_len=TEST_CONFIG['max_seq_len'],
        pad_token_id=tokenizer.pad_id,
        tie_weights=TEST_CONFIG['tie_weights'],
        use_flash_attn=TEST_CONFIG['use_flash_attn'],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt['model_state_dict'])
    model2.eval()

    with torch.no_grad():
        test_input = torch.randint(1, 1000, (1, 32), device=device)
        with safe_autocast():
            logits1, _ = model(test_input)
            logits2, _ = model2(test_input)
        diff = (logits1 - logits2).abs().max().item()
        assert diff < 1e-5, f"Reloaded logit diff too large: {diff:.2e}"
        print(f"  重载后 logit 最大差异: {diff:.2e} [OK]")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("全部检查通过")
    print("  [+] 分词器加载")
    print("  [+] 数据集流水线")
    print("  [+] 模型前向 + 反向")
    print("  [+] 优化器步进 + Cosine LR 调度")
    print("  [+] 验证 (PPL)")
    print("  [+] Checkpoint 保存/加载 (一致性验证)")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
