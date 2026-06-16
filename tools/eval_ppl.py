"""快速 PPL 评估。用法: python tools/eval_ppl.py [--max_batches 100]"""
import torch, sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model_for_inference
from xfind_dataset import LMDataset, collate_fn
from tokenizer.xfind_tokenizer import XfindTokenizer
from torch.utils.data import DataLoader
import math

@torch.no_grad()
def compute_ppl_fast(model, data_loader, device, max_batches=None):
    """快速计算困惑度，可限制批次数"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    for i, (input_ids, target_ids) in enumerate(data_loader):
        if max_batches and i >= max_batches:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )

        total_loss += loss.item()
        total_tokens += (target_ids != 0).sum().item()

    if total_tokens == 0:
        return 0, 1

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--max_batches', type=int, default=None, help='最多评估批次数（None=全量）')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 加载模型（配置从 checkpoint 自动读取）
    model, config = load_model_for_inference(args.model, device)

    total, _ = model.get_num_params()
    print(f"Model: {total/1e6:.2f}M params")

    # Tokenizer
    tokenizer = XfindTokenizer('./tokenizer/checkpoints/bpe_32k')
    print(f"Vocab: {len(tokenizer)}")

    for split in ['valid', 'test']:
        txt_path = f'data/splits/{split}.txt'
        if not os.path.exists(txt_path):
            print(f"  Skip {split}: no data file")
            continue

        print(f"\nEvaluating {split} set...")
        ds = LMDataset('./data/splits', tokenizer, 1024, split)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate_fn, num_workers=0)

        n_batches = min(args.max_batches, len(dl)) if args.max_batches else len(dl)
        print(f"  Samples: {len(ds)}, Batches: {n_batches}")

        loss, ppl = compute_ppl_fast(model, dl, device, args.max_batches)
        print(f"  {split.upper()} -> Loss: {loss:.4f}, PPL: {ppl:.2f}")


if __name__ == '__main__':
    main()
