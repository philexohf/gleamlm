"""
Knowledge Distillation — 大模型 (Teacher) 教小模型 (Student)。

核心公式:
  L = α * L_hard(Student, labels) + (1-α) * L_soft(Student, Teacher)
  L_soft = KL(σ(T_logits / τ) || σ(S_logits / τ)) * τ²

用法:"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.distill_loss import distill_loss
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from gleamlm.utils.torch_utils import clean_state_dict


class DistillDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len
        self.data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text", item.get("prompt", item.get("instruction", "")))
                if item.get("response"):
                    text = text + item["response"]
                if item.get("output"):
                    text = text + item["output"]
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_seq_len):
    all_ids = []
    for text in batch:
        ids = tokenizer.encode(text, add_bos=True)[:max_seq_len]
        all_ids.append(ids)
    max_len = max(len(x) for x in all_ids)
    pad_id = tokenizer.pad_id
    input_ids = [x + [pad_id] * (max_len - len(x)) for x in all_ids]
    return torch.tensor(input_ids)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)

    dataset = DistillDataset(args.data, max_seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, args.seq_len))

    def _load_model(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = extract_checkpoint_config(ckpt)
        model = GleamLMModel(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg.get("dropout", 0.0),
            max_seq_len=args.seq_len,
            pad_token_id=tokenizer.pad_id,
            use_flash_attn=cfg.get("use_flash_attn", False),
        ).to(device)
        model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=False)
        return model, cfg

    teacher, _ = _load_model(args.teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student, cfg = _load_model(args.student)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"Distill — Teacher: {t_params/1e6:.2f}M, Student: {s_params/1e6:.2f}M, τ={args.temperature}, α={args.alpha}")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for step, input_ids in enumerate(loader):
            input_ids = input_ids.to(device)

            labels = input_ids.clone()

            with torch.no_grad():
                t_logits, _, _, _ = teacher(input_ids)

            s_logits, _, aux_loss, _ = student(input_ids)
            loss = distill_loss(s_logits, t_logits, labels, temperature=args.temperature, alpha=args.alpha)
            loss = loss + aux_loss * 0.01

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step} loss={loss.item():.4f}")

    out_path = os.path.join(args.output_dir, "distilled.pt")
    torch.save({"model_state_dict": student.state_dict(), "_config": cfg}, out_path)
    print(f"Distilled model saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM Knowledge Distillation")
    p.add_argument("--teacher", required=True, help="Teacher model checkpoint")
    p.add_argument("--student", required=True, help="Student model checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", default="./checkpoints/distill")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer_path", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
