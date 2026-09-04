"""
LoRA SFT 微调 — 冻结预训练权重，只更新低秩 adapter。

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
from gleamlm.utils.chatml import format_chatml
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, extract_checkpoint_config
from gleamlm.utils.torch_utils import clean_state_dict
from gleamlm.trainer.lora import LoraConfig, apply_lora_to_model, merge_lora_weights


class SFTDataset(Dataset):
    """统一 ChatML：兼容单轮 {instruction,output} 与多轮 {messages}（与 core SFT 同语义）。

    返回 (prompt_text, resp_text)：prompt 含到 assistant 头为止的历史+指令，
    resp 为最后一条 assistant 内容；collate 里 assistant 起始之前全部 -100。
    """

    def __init__(self, data_path: str, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len
        self.data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "messages" in item:
                    msgs = item["messages"]
                    if not msgs or msgs[-1].get("role") != "assistant":
                        continue
                    history = msgs[:-1]
                    prompt_text = (
                        format_chatml(history, add_generation_prompt=True)
                        if history else ""
                    )
                    resp_text = msgs[-1]["content"] + "<|im_end|>"
                else:
                    prompt_text = format_chatml(
                        [{"role": "user", "content": item.get("instruction", "")}],
                        add_generation_prompt=True,
                    )
                    resp_text = item.get("output", "") + "<|im_end|>"
                if not prompt_text.strip() or not resp_text.strip():
                    continue
                self.data.append((prompt_text, resp_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_seq_len):
    prompts, responses = zip(*batch)
    input_ids, labels_list = [], []
    for p, r in zip(prompts, responses):
        ids = tokenizer.encode(p + r, add_bos=True)
        ids = ids[:max_seq_len]
        p_len = len(tokenizer.encode(p, add_bos=True))
        label = [-100] * (p_len - 1) + ids[p_len - 1:]
        label = label[:max_seq_len]
        if len(label) < len(ids):
            label = label + [-100] * (len(ids) - len(label))
        input_ids.append(ids)
        labels_list.append(label)
    max_len = max(len(x) for x in input_ids)
    pad_id = tokenizer.pad_id
    input_ids = [x + [pad_id] * (max_len - len(x)) for x in input_ids]
    labels_list = [x + [-100] * (max_len - len(x)) for x in labels_list]
    return torch.tensor(input_ids), torch.tensor(labels_list)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BBPETokenizer.load(args.tokenizer_path or DEFAULT_TOKENIZER_PATH)

    dataset = SFTDataset(args.data, max_seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, args.seq_len))

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
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
    model.train()

    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["W_q", "W_k", "W_v", "W_o"])
    replaced = apply_lora_to_model(model, lora_cfg)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    lora_count = sum(p.numel() for p in lora_params)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    print(f"LoRA — base: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, trainable: {lora_count/1e3:.1f}K, r={args.lora_r}")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for step, (input_ids, labels) in enumerate(loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits, _, aux_loss, _ = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss + aux_loss * 0.01

            loss.backward()
            nn.utils.clip_grad_norm_(lora_params, args.clip)
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step} loss={loss.item():.4f}")

    save_path = os.path.join(args.output_dir, "lora.pt")
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"lora": lora_state, "_config": cfg, "lora_config": {"r": args.lora_r, "alpha": args.lora_alpha}}, save_path)
    print(f"LoRA weights saved: {save_path}")

    if args.merge:
        merged = merge_lora_weights(model)
        full_path = os.path.join(args.output_dir, "merged.pt")
        torch.save({"model_state_dict": model.state_dict(), "_config": cfg}, full_path)
        print(f"Merged model saved: {full_path}")


def parse_args():
    p = argparse.ArgumentParser(description="GleamLM LoRA SFT")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", default="./checkpoints/lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--merge", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
