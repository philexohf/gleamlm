"""GleamLM-Lite 87M SFT 指令微调脚本。基于 best_model.pt，ChatML 格式 + loss mask

用法：
    python sft.py --data_path ./data/sft_data.jsonl --model_path ./checkpoints/best_model.pt
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
import numpy as np
import math
import argparse
from tqdm import tqdm

from gleamlm.models.model import GleamLMModel
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.inference.generate import generate_response
from gleamlm.utils.torch_utils import get_lr_cosine

# 路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, 'checkpoints')

# 系统消息池
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


# SFT 数据集
class SFTDataset(Dataset):
    """SFT 数据集：JSONL → ChatML 格式 → loss mask

    ChatML 格式：
      <|im_start|><|user|>\n{instruction}<|im_end|>\n<|im_start|><|assistant|>\n{output}<|im_end|>

    有系统消息时：
      <|im_start|><|system|>\n{system_prompt}<|im_end|>\n
      <|im_start|><|user|>\n{instruction}<|im_end|>\n
      <|im_start|><|assistant|>\n{output}<|im_end|>

    loss mask：user/system 部分 label=-100，只对 assistant 部分计算损失
    """

    def __init__(self, data_path, tokenizer, max_seq_len=2048,
                 inject_system_ratio=0.2):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.inject_system_ratio = inject_system_ratio
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        # 加载 JSONL 数据
        self.data = []
        required_keys = {'instruction', 'output'}
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping line {i} in {data_path}: {e}")
                    continue
                missing = required_keys - set(item.keys())
                if missing:
                    raise ValueError(f"Line {i}: missing required keys {missing} in {data_path}")
                self.data.append({
                    'instruction': item['instruction'],
                    'output': item['output'],
                })
        print(f"Loaded {len(self.data)} SFT samples from {data_path}")

        # 预生成 system prompt，用固定种子保证可复现
        rng = random.Random(42)
        self._system_prompts = []
        for _ in range(len(self.data)):
            if rng.random() < inject_system_ratio:
                self._system_prompts.append(rng.choice(SYSTEM_PROMPTS))
            else:
                self._system_prompts.append("")

    def __len__(self):
        return len(self.data)

    def _encode(self, text):
        """编码文本为 token ID 列表（不添加 BOS/EOS）"""
        return self.tokenizer.encode(text, add_bos=False, add_eos=False)

    def _build_prompt(self, instruction, system_prompt=""):
        """构建 prompt 文本（到 assistant 开头，不含回答）"""
        if system_prompt:
            return (f"<|im_start|><|system|>\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|><|user|>\n{instruction}<|im_end|>\n"
                    f"<|im_start|><|assistant|>\n")
        else:
            return (f"<|im_start|><|user|>\n{instruction}<|im_end|>\n"
                    f"<|im_start|><|assistant|>\n")

    def _build_full(self, instruction, output, system_prompt=""):
        """构建完整 ChatML 文本（含回答）"""
        if system_prompt:
            return (f"<|im_start|><|system|>\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|><|user|>\n{instruction}<|im_end|>\n"
                    f"<|im_start|><|assistant|>\n{output}<|im_end|>")
        else:
            return (f"<|im_start|><|user|>\n{instruction}<|im_end|>\n"
                    f"<|im_start|><|assistant|>\n{output}<|im_end|>")

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        output = item['output']

        # 使用预生成的 system prompt（固定种子，可复现）
        system_prompt = self._system_prompts[idx]

        # 编码：prompt + 完整文本
        prompt_text = self._build_prompt(instruction, system_prompt)
        full_text = self._build_full(instruction, output, system_prompt)

        prompt_ids = self._encode(prompt_text)
        full_ids = self._encode(full_text)

        # input_ids = 前 N-1 个 token（预测下一个）
        # labels    = 后 N-1 个 token（模型应预测的目标）
        P = min(len(prompt_ids), self.max_seq_len - 2)  # 上限，至少留 1 个 output token

        # 截断到 max_seq_len（确保 <|im_end|> 不被切掉）
        if len(full_ids) > self.max_seq_len:
            im_end_ids = self._encode("<|im_end|>")
            full_ids = full_ids[:self.max_seq_len]
            # 如果截断后末尾不是完整的 <|im_end|>，用 <|im_end|> 替代末尾
            if len(full_ids) >= len(im_end_ids) and full_ids[-len(im_end_ids):] != im_end_ids:
                full_ids = full_ids[:self.max_seq_len - len(im_end_ids)] + im_end_ids

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


# 训练循环
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

        amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(amp_device):
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


# 生成评估
@torch.no_grad()
def generate_response_sft(model, tokenizer, instruction, max_new_tokens=256,
                          temperature=0.8, top_k=50, top_p=0.9):
    """SFT 后生成对话回复（委托共享实现）"""
    return generate_response(model, tokenizer, instruction, max_new_tokens,
                             temperature, top_k, top_p)


def evaluate_sft(model, tokenizer, test_prompts):
    """生成对话样例评估"""
    model.eval()
    print("\n" + "=" * 60)
    print("SFT 生成评估")
    print("=" * 60)
    results = []
    for prompt in test_prompts:
        response = generate_response_sft(model, tokenizer, prompt)
        results.append((prompt, response))
        print(f"\n[User] {prompt}")
        print(f"[Assistant] {response}")
        print("-" * 40)
    return results


# 参数解析
def get_sft_args():
    parser = argparse.ArgumentParser(description='GleamLM-Lite 87M SFT 指令微调')

    # 数据与模型路径
    parser.add_argument("--data_path", type=str, default="data/sft_data.jsonl",
                        help='SFT JSONL 数据路径')
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(_CHECKPOINT_DIR, 'best_model.pt'),
                        help='预训练模型路径')
    parser.add_argument("--tokenizer_path", type=str,
                        default=DEFAULT_TOKENIZER_PATH,
                        help='BBPE 分词器目录路径')
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(_CHECKPOINT_DIR, 'sft'),
                        help='SFT 模型保存目录')

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help='SFT 学习率')
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    # 系统消息注入
    parser.add_argument("--inject_system_ratio", type=float, default=0.2,
                        help='系统消息随机注入比例')

    parser.add_argument("--resume", type=str, default=None,
                        help='从指定 checkpoint 续训（如 ./checkpoints/sft/sft_epoch_0.pt）')

    # Lite 87M 模型架构（默认值）
    parser.add_argument("--vocab_size", type=int, default=12002)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_kv_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_flash_attn", action='store_true', default=True)
    parser.add_argument("--no_flash_attn", dest='use_flash_attn', action='store_false')

    return parser.parse_args()


# 主函数
def main():
    args = get_sft_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("GleamLM-Lite 87M SFT 指令微调")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"LR: {args.lr:.1e}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    model = GleamLMModel(
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
        use_flash_attn=args.use_flash_attn,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Loaded pretrained model: {args.model_path}")
    total, trainable = model.get_num_params()
    print(f"Model params: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

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
    )

    # 优化器
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
        lambda step: get_lr_cosine(step, total_steps, args.warmup_ratio, min_lr_ratio=0.05),
    )
    scaler = torch.amp.GradScaler('cuda')

    # 断点续训
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
        start_epoch = resume_ckpt['epoch'] + 1
        global_step = resume_ckpt.get('global_step', 0)
        best_loss = resume_ckpt.get('train_loss', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, global_step={global_step}, best_loss={best_loss:.4f}")

    # 评估提示词
    eval_prompts = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？",
        "请用一句话描述北京的秋天。",
        "写一首关于春天的五言诗。",
        "请解释一下什么是光合作用。",
    ]

    print("\n--- SFT 前生成基线 ---")
    model.eval()
    evaluate_sft(model, tokenizer, eval_prompts)
    model.train()

    os.makedirs(args.save_dir, exist_ok=True)
    if not args.resume:
        global_step = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args, global_step, scaler,
        )

        # 每个 epoch 后生成评估
        print(f"\n--- SFT Epoch {epoch} 生成评估 ---")
        model.eval()
        evaluate_sft(model, tokenizer, eval_prompts)
        model.train()

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

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
                'args': args,
            }, best_path)
            print(f"  Saved best SFT model (loss={train_loss:.4f}) -> {best_path}")

    print("\n" + "=" * 60)
    print("SFT 训练完成，最终生成评估")
    print("=" * 60)
    model.eval()
    evaluate_sft(model, tokenizer, eval_prompts)
    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
