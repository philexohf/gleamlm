"""快速 PPL 评估。用法: python scripts/eval_ppl.py [--max_batches 100]"""

import argparse
import os

import torch

from gleamlm import load_model_for_inference
from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_NANO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = _NANO_DIR  # scripts/ 已在项目根下，_NANO_DIR 即项目根
DEFAULT_CHECKPOINT_DIR = os.path.join(_NANO_DIR, "checkpoints")
# 自动检测数据目录：优先 nano_data，其次 lite_data，最后 splits
DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "nano_data")
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "lite_data")
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "splits")
import math

from torch.utils.data import DataLoader


@torch.no_grad()
def compute_ppl_fast(model, data_loader, device, max_batches=None, pad_token_id=0):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_token_id)
    for i, (input_ids, target_ids) in enumerate(data_loader):
        if max_batches and i >= max_batches:
            break
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits, _ = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        total_loss += loss.item()
        total_tokens += (target_ids != pad_token_id).sum().item()
    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser(description="命令行 PPL 评估")
    parser.add_argument(
        "--model", type=str, default=os.path.join(DEFAULT_CHECKPOINT_DIR, "best_model.pt")
    )
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=f"数据目录 (默认自动检测，当前: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, config = load_model_for_inference(args.model, device)
    total, _ = model.get_num_params()
    max_seq_len = config.get("max_seq_len", 1024)
    print(f"Model: {total / 1e6:.2f}M params, max_seq_len={max_seq_len}")

    tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
    print(f"Vocab: {len(tokenizer)}")

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    print(f"Data dir: {data_dir}")

    for split in ["valid", "test"]:
        txt_path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(txt_path):
            print(f"  Skip {split}: no data file")
            continue
        print(f"\nEvaluating {split} set...")
        ds = LMDataset(data_dir, tokenizer, max_seq_len, split)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
            num_workers=0,
        )
        n_batches = min(args.max_batches, len(dl)) if args.max_batches else len(dl)
        print(f"  Samples: {len(ds)}, Batches: {n_batches}")
        avg = compute_ppl_fast(model, dl, device, args.max_batches, tokenizer.pad_id)
        print(f"  {split}: avg_loss={avg:.4f}, PPL={math.exp(avg):.2f}")


if __name__ == "__main__":
    main()
