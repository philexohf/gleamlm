"""XFIND-LLM SFT 指令微调脚本。基于 best_model.pt，纯文本 Q:/A: 格式 + loss mask

用法：
    python xfind_sft.py --data_path ./data/sft_data.jsonl --model_path ./checkpoints/best_model.pt
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
import random
import numpy as np
import math
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.xfind_model import XfindModel
from tokenizer.xfind_tokenizer import build_tokenizer
from inference.sampler import sample_token

# ---- 系统消息池 ----
SYSTEM_PROMPTS = [
    "你是一个有帮助的AI助手。",
    "你是一个友善的中文对话助手，请用简洁清晰的语言回答问题。",
    "你是一个知识渊博的助手，请准确回答问题。",
    "You are a helpful AI assistant.",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# SFT 数据集
# ============================================================

class SFTDataset(Dataset):
    """SFT 数据集：JSONL → 纯文本 Q:/A: 格式 → loss mask

    格式：
      无系统消息: "Q: {instruction}\nA: {output}<|endoftext|>"
      有系统消息: "system: {system_prompt}\nQ: {instruction}\nA: {output}<|endoftext|>"

    loss mask：Q: 和 system: 部分 label=-100，只对 A: 部分计算损失
     """

    def __init__(self, data_path, tokenizer, max_seq_len=512,
                 inject_system_ratio=0.2):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.inject_system_ratio = inject_system_ratio
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        # 加载 JSONL 数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append({
                    'instruction': item['instruction'],
                    'output': item['output'],
                })
        print(f"Loaded {len(self.data)} SFT samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def _encode(self, text):
        """编码文本为 token ID 列表（不自动添加 BOS/EOS）"""
        return self.tokenizer.sp.encode(text, out_type=int)

    def _build_prompt(self, instruction, system_prompt=""):
        """构建 prompt 文本（到 \nA: 为止，不含回答）"""
        if system_prompt:
            return f"system: {system_prompt}\nQ: {instruction}\nA:"
        else:
            return f"Q: {instruction}\nA:"

    def _build_full(self, instruction, output, system_prompt=""):
        """构建完整文本（含回答）"""
        if system_prompt:
            return f"system: {system_prompt}\nQ: {instruction}\nA: {output}<|endoftext|>"
        else:
            return f"Q: {instruction}\nA: {output}<|endoftext|>"

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        output = item['output']

        # 20% 概率注入系统消息
        system_prompt = ""
        if random.random() < self.inject_system_ratio:
            system_prompt = random.choice(SYSTEM_PROMPTS)

        # 编码：prompt + 完整文本
        prompt_text = self._build_prompt(instruction, system_prompt)
        full_text = self._build_full(instruction, output, system_prompt)

        prompt_ids = self._encode(prompt_text)
        full_ids = self._encode(full_text)

        # input_ids = 前 N-1 个 token（预测下一个）
        # labels    = 后 N-1 个 token（模型应预测的目标）
        P = min(len(prompt_ids), self.max_seq_len - 2)  # 上限，至少留 1 个 output token

        # 截断到 max_seq_len（含 eos，确保 input_ids 可统一 padding）
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]

        N = len(full_ids)
        input_ids = full_ids[:-1]       # shape: [N-1]
        labels = full_ids[1:]           # shape: [N-1]

        # loss mask：prompt 部分（user/system）label = -100
        # labels[0] 预测 position 1，labels[P-1] 预测 position P（第一个 output token）
        labels = list(labels)           # 转为 Python list 以便赋值
        mask_end = min(P, len(labels))  # P 个 prompt token → mask 前 P-1 个 label，第 P 个是 output 首 token
        for i in range(mask_end - 1):
            labels[i] = -100

        # Padding 到 max_seq_len（统一长度）
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    @staticmethod
    def collate_fn(batch):
        """堆叠 batch，input 和 target 左移一位的关系已内置在 labels 中"""
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return input_ids, labels


# ============================================================
# 学习率调度
# ============================================================

def get_lr_cosine(step, total_steps, warmup_ratio=0.01, min_lr_ratio=0.05):
    """Cosine Annealing + Warmup"""
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================
# 训练循环
# ============================================================

def train_one_epoch(model, train_loader, optimizer, scheduler, device,
                    epoch, args, global_step, scaler, log_interval=50):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch}", mininterval=3, miniters=20)

    for batch_idx, (input_ids, labels) in enumerate(pbar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.amp.autocast('cuda'):
            logits, _ = model(input_ids)
            # 用 ignore_index=-100 的 loss，自动跳过 prompt 部分
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        loss = loss / args.accumulate_grad
        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item() * args.accumulate_grad
        num_batches += 1

        if batch_idx % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item() * args.accumulate_grad:.4f}",
                "lr": f"{lr:.2e}"
            })

    return total_loss / num_batches, global_step


# ============================================================
# 生成评估
# ============================================================

@torch.no_grad()
def generate_response(model, tokenizer, instruction, max_new_tokens=256,
                      temperature=0.8, top_k=50, top_p=0.9):
    """SFT 后生成对话回复，遇到 <|endoftext|> 自动截断"""
    model.eval()
    device = next(model.parameters()).device

    # 构建 prompt
    prompt_text = f"Q: {instruction}\nA:"
    prompt_ids = tokenizer.sp.encode(prompt_text, out_type=int)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    generated_ids = prompt_ids.copy()
    stopped = False

    # 预填充 KV Cache
    with torch.amp.autocast('cuda'):
        logits, past_kv = model(prompt_tensor)

    for i in range(max_new_tokens):
        next_logits = logits[:, -1, :]  # [1, vocab]
        next_token = sample_token(
            next_logits, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=1.15,
            generated_ids=generated_ids,
        )
        token_id = next_token.item()

        if token_id == tokenizer.eos_id:
            break
        if token_id == tokenizer.pad_id:
            break

        generated_ids.append(token_id)

        # 每 4 个 token 解码检查 <|endoftext|>
        if not stopped and (i + 1) % 4 == 0:
            draft = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special=True)
            if "<|endoftext|>" in draft:
                stopped = True
                break

        # 推理下一个 token（仅传入新 token）
        next_input = torch.tensor([[token_id]], dtype=torch.long).to(device)
        with torch.amp.autocast('cuda'):
            logits, past_kv = model(next_input, past_kv_list=past_kv)

    model.train()
    response = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special=True)
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]
    return response


def evaluate_sft(model, tokenizer, test_prompts):
    """生成对话样例评估"""
    model.eval()
    print("\n" + "=" * 60)
    print("SFT 生成评估")
    print("=" * 60)
    results = []
    for prompt in test_prompts:
        response = generate_response(model, tokenizer, prompt)
        results.append((prompt, response))
        print(f"\n[User] {prompt}")
        print(f"[Assistant] {response}")
        print("-" * 40)
    model.train()
    return results


# ============================================================
# 主函数
# ============================================================

def get_sft_args():
    parser = argparse.ArgumentParser(description='XFIND-LLM SFT 指令微调')

    # 数据与模型路径
    parser.add_argument("--data_path", type=str, required=True,
                        help='SFT JSONL 数据路径')
    parser.add_argument("--model_path", type=str,
                        default="./checkpoints/best_model.pt",
                        help='预训练模型路径')
    parser.add_argument("--tokenizer_path", type=str,
                        default="./tokenizer/checkpoints/bpe_32k",
                        help='分词器模型前缀')
    parser.add_argument("--save_dir", type=str, default="./checkpoints/sft",
                        help='SFT 模型保存目录')

    # 训练参数
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7,
                        help='SFT 学习率（预训练 LR 的 1/600）')
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # 系统消息注入
    parser.add_argument("--inject_system_ratio", type=float, default=0.2,
                        help='系统消息随机注入比例')

    # 模型架构（需与 best_model.pt 一致）
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1365)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = get_sft_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("XFIND-LLM SFT 指令微调")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"LR: {args.lr:.1e}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    # 1. 加载分词器
    tokenizer = build_tokenizer(
        text_files=[],  # 空列表，直接加载已有模型
        vocab_size=args.vocab_size,
        model_prefix=args.tokenizer_path,
    )
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # 2. 加载预训练模型
    model = XfindModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_id,
        tie_weights=True,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained model: {args.model_path}")
    total, trainable = model.get_num_params()
    print(f"Model params: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    # 3. 构建 SFT 数据集
    train_dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        inject_system_ratio=args.inject_system_ratio,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTDataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # 4. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    total_steps = math.ceil(len(train_loader) / args.accumulate_grad) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(step, total_steps, args.warmup_ratio),
    )
    scaler = torch.amp.GradScaler('cuda')

    # 5. 评估提示词
    eval_prompts = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？",
        "请用一句话描述北京的秋天。",
        "写一首关于春天的五言诗。",
        "请解释一下什么是光合作用。",
    ]

    # 6. 训练前生成基线
    print("\n--- SFT 前生成基线 ---")
    evaluate_sft(model, tokenizer, eval_prompts)

    # 7. 训练循环
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args, global_step, scaler,
        )

        # 每个 epoch 后生成评估
        print(f"\n--- SFT Epoch {epoch} 生成评估 ---")
        evaluate_sft(model, tokenizer, eval_prompts)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # 保存 checkpoint
        ckpt_name = f"sft_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'args': args,
        }, os.path.join(args.save_dir, ckpt_name))

        # 跟踪 best（按 loss）
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(args.save_dir, "sft_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, best_path)
            print(f"  Saved best SFT model (loss={train_loss:.4f}) -> {best_path}")

    # 8. 训练后最终评估
    print("\n" + "=" * 60)
    print("SFT 训练完成，最终生成评估")
    print("=" * 60)
    evaluate_sft(model, tokenizer, eval_prompts)
    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
