"""烁珑GleamLM DPO 偏好对齐脚本。基于 sft_best.pt 策略模型 + 冻结参考模型。

用法：
    python gleamlm_dpo.py --data_path ./data/dpo_data.jsonl \
                        --model_path ./checkpoints/sft/sft_best.pt
"""
import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.gleamlm_model import GleamLMModel
from models.gleamlm_config import DEFAULT_TOKENIZER_PATH, DEFAULT_CHECKPOINT_DIR
from tokenizer.bbpe_tokenizer import BBPETokenizer
from inference.sampler import sample_token


PAD_ID = 0


def dpad_collate(batch):
    """将 chosen_ids 和 rejected_ids 填充到 batch 内最大长度 + 合并 mask"""
    B = len(batch)

    # Prefix lengths (prompt length) may vary slightly due to BPE, take max
    max_c = max(b["chosen_ids"].size(0) for b in batch)
    max_r = max(b["rejected_ids"].size(0) for b in batch)

    chosen_ids = torch.full((B, max_c), PAD_ID, dtype=torch.long)
    rejected_ids = torch.full((B, max_r), PAD_ID, dtype=torch.long)
    chosen_mask = torch.zeros(B, max_c - 1)
    rejected_mask = torch.zeros(B, max_r - 1)

    for i, b in enumerate(batch):
        Lc = b["chosen_ids"].size(0)
        Lr = b["rejected_ids"].size(0)
        chosen_ids[i, :Lc] = b["chosen_ids"]
        rejected_ids[i, :Lr] = b["rejected_ids"]
        chosen_mask[i, :b["chosen_mask"].size(0)] = b["chosen_mask"]
        rejected_mask[i, :b["rejected_mask"].size(0)] = b["rejected_mask"]

    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
    }


# DPO Dataset
class DPODataset(Dataset):
    """DPO 数据集：chosen/rejected 对，prompt 部分 loss mask 为 0"""

    def __init__(self, data_path, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        instruction = s["instruction"]
        chosen = s["chosen"]
        rejected = s["rejected"]

        prompt_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = self.tokenizer.encode(prompt_text, add_bos=False, add_eos=False)

        # 完整序列（含 prompt + answer + eos）
        chosen_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>"
        rejected_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{rejected}<|im_end|>"

        chosen_ids = self.tokenizer.encode(chosen_text, add_bos=False, add_eos=False)
        rejected_ids = self.tokenizer.encode(rejected_text, add_bos=False, add_eos=False)

        # 取 min 长度以对齐 batch + 截断到 max_seq_len
        L = min(len(chosen_ids), len(rejected_ids), self.max_seq_len)
        chosen_ids = chosen_ids[:L]
        rejected_ids = rejected_ids[:L]

        # 确保 prompt 部分一致
        P = min(len(prompt_ids), L)

        # loss mask: 仅 answer 部分为 1（shift 一位用于 next-token prediction）
        chosen_mask = torch.zeros(L - 1, dtype=torch.float32)
        rejected_mask = torch.zeros(L - 1, dtype=torch.float32)
        chosen_mask[max(0, P - 1):] = 1.0
        rejected_mask[max(0, P - 1):] = 1.0

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
        }


# DPO Loss
def get_lr_cosine(step, total_steps, warmup_ratio=0.01, min_lr_ratio=0.05):
    """Cosine Annealing + Warmup"""
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

def compute_log_probs(logits, input_ids, mask):
    """计算每个 token 的 log 概率，仅 mask 部分计入"""
    # logits: [B, L, V]; input_ids: [B, L]; mask: [B, L-1]
    log_probs_all = F.log_softmax(logits, dim=-1)  # [B, L, V]
    # gather: 取每个位置实际 token 的 log_prob
    log_probs_token = log_probs_all[:, :-1, :].gather(
        2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # [B, L-1]
    return (log_probs_token * mask).sum(dim=-1)  # [B]


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp,
             beta=0.1):
    """DPO 损失"""
    # log-ratio: 当前策略相比参考策略在 chosen/rejected 上的优势差
    term = (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)
    return -F.logsigmoid(beta * term).mean()


# Training
@torch.no_grad()
def get_reference_logps(ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    """用冻结参考模型计算 chosen 和 rejected 的 log 概率"""
    ref_model.eval()
    with torch.amp.autocast('cuda'):
        c_logits, _ = ref_model(chosen_ids)
        r_logits, _ = ref_model(rejected_ids)
    ref_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
    ref_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)
    return ref_cho, ref_rej


def train_one_epoch(model, ref_model, dataloader, optimizer, scheduler,
                    scaler, beta, device, args):
    model.train()
    ref_model.eval()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="DPO")
    for batch_idx, batch in enumerate(pbar):
        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)

        ref_cho, ref_rej = get_reference_logps(
            ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask)

        with torch.amp.autocast('cuda'):
            c_logits, _ = model(chosen_ids)
            r_logits, _ = model(rejected_ids)

        policy_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
        policy_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)

        loss = dpo_loss(policy_cho, policy_rej, ref_cho.detach(), ref_rej.detach(), beta)

        loss = loss / args.accumulate_grad
        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.accumulate_grad == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.accumulate_grad
        n_batches += 1

        if (batch_idx + 1) % args.accumulate_grad == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item() * args.accumulate_grad:.4f}", lr=f"{lr:.2e}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def generate_response(model, tokenizer, instruction, max_new_tokens=256,
                      temperature=0.8, top_k=50, top_p=0.9):
    """生成对话回复，遇到 <|im_end|> 或 <|endoftext|> 自动截断"""
    model.eval()
    device = next(model.parameters()).device
    prompt_text = (f"<|im_start|>user\n{instruction}<|im_end|>\n"
                   f"<|im_start|>assistant\n")
    prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    generated_ids = prompt_ids.copy()
    stopped = False

    with torch.amp.autocast('cuda'):
        logits, past_kv = model(prompt_tensor)

    for i in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        next_token = sample_token(
            next_logits, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=1.15,
            generated_ids=generated_ids,
        )
        token_id = next_token.item()
        # 停止条件：特殊 token
        if token_id == tokenizer.eos_id:
            break
        if token_id == tokenizer.pad_id:
            break
        if token_id == tokenizer.special_tokens.get("<|im_end|>"):
            break
        generated_ids.append(token_id)

        if not stopped and (i + 1) % 4 == 0:
            draft = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special=True)
            if "<|endoftext|>" in draft:
                stopped = True
                break

        next_input = torch.tensor([[token_id]], dtype=torch.long).to(device)
        with torch.amp.autocast('cuda'):
            logits, past_kv = model(next_input, past_kv_list=past_kv)

    model.train()
    response = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special=True)
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]
    return response


def evaluate_dpo(model, tokenizer):
    """用 5 条测试 prompt 评估 DPO 后生成质量"""
    eval_prompts = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？",
        "请用一句话描述北京的秋天。",
        "写一首关于春天的五言诗。",
        "请解释一下什么是光合作用。",
    ]
    model.eval()
    print("\n" + "=" * 60)
    print("DPO 生成评估")
    print("=" * 60)
    for prompt in eval_prompts:
        print(f"\n[User] {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"[Assistant] {response}")
        print("-" * 40)
    model.train()


# Main
def main():
    parser = argparse.ArgumentParser(description="烁珑GleamLM DPO")
    parser.add_argument("--data_path", default="./data/dpo_data.jsonl")
    parser.add_argument("--model_path", default=f"{DEFAULT_CHECKPOINT_DIR}/sft/sft_best.pt")
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vocab_size", type=int, default=12001)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad", type=int, default=2)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", default=f"{DEFAULT_CHECKPOINT_DIR}/dpo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("烁珑GleamLM DPO 偏好对齐")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"LR: {args.lr:.1e}, Beta: {args.beta}, Epochs: {args.epochs}")

    # 1. 分词器
    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    sft_ckpt = torch.load(args.model_path, map_location=device)

    policy_model = GleamLMModel(
        vocab_size=args.vocab_size, d_model=args.d_model,
        max_seq_len=args.max_seq_len,
    ).to(device)
    policy_model.load_state_dict(sft_ckpt["model_state_dict" if "model_state_dict" in sft_ckpt else "model"])
    print(f"Policy model: {sum(p.numel() for p in policy_model.parameters())/1e6:.2f}M params")

    ref_model = GleamLMModel(
        vocab_size=args.vocab_size, d_model=args.d_model,
        max_seq_len=args.max_seq_len,
    ).to(device)
    ref_model.load_state_dict(sft_ckpt["model_state_dict" if "model_state_dict" in sft_ckpt else "model"])
    for p in ref_model.parameters():
        p.requires_grad = False
    print("Reference model: frozen")

    # 3. 数据集
    dataset = DPODataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    print(f"DPO pairs: {len(dataset)}")

    effective_batch = args.batch_size * args.accumulate_grad
    print(f"Batch: {args.batch_size} x {args.accumulate_grad} = {effective_batch}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=dpad_collate)

    # 4. 优化器 & 调度器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = math.ceil(len(dataloader) / args.accumulate_grad) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_cosine(step, total_steps, warmup_ratio=0.01),
    )
    scaler = torch.amp.GradScaler('cuda')

    # 5. DPO 前基线
    print("\n--- DPO 前生成基线 ---")
    evaluate_dpo(policy_model, tokenizer)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            policy_model, ref_model, dataloader, optimizer, scheduler,
            scaler, args.beta, device, args)
        print(f"\nDPO Epoch {epoch}: loss={avg_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "dpo_best.pt")
    torch.save({
        "model_state_dict": policy_model.state_dict(),
        "dpo_loss": avg_loss,
    }, save_path)
    print(f"Model saved: {save_path}")

    # 8. DPO 后评估
    print("\n--- DPO 后最终评估 ---")
    evaluate_dpo(policy_model, tokenizer)


if __name__ == "__main__":
    main()
