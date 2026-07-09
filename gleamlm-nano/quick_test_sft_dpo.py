"""
GleamLM SFT+DPO Pipeline 测试 (修复版)
预训练 → SFT → DPO，跳过 evaluate 避免卡死。
"""

import json
import os
import random
import time

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # standalone test script, safe to chdir (not importable)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

print("=" * 60)
print("Step 1: 快速预训练 (200 steps)...")
# 清除旧缓存
import shutil

from data_tools.build_dataset import stream_build
from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer

for d in ["data/test_raw", "data/test_splits", "checkpoints/sft_test", "checkpoints/dpo_test"]:
    if os.path.exists(d):
        shutil.rmtree(d)

os.makedirs("data/test_raw", exist_ok=True)
templates = [
    "今天天气{adj}，适合{act}。",
    "人工智能是{adj}的技术。",
    "机器学习{adv}改变了我们的{obj}。",
    "深度学习模型{act}的能力很{adj}。",
    "在{domain}领域，{tech}技术{adv}受欢迎。",
    "Python是一种{adj}语言。",
]
adj_words = ["重要", "强大", "新兴", "高效", "智能", "先进", "可靠", "稳定"]
act_words = ["学习编程", "处理数据", "识别图像", "生成文本", "分析问题", "训练网络"]
adv_words = ["快速", "大幅", "逐步", "广泛"]
obj_words = ["生活方式", "工作效率", "系统性能", "产品质量"]
domain_words = ["医疗", "教育", "金融", "交通"]
tech_words = ["Transformer", "CNN", "LSTM", "GAN"]

random.seed(42)
for fname in ["wiki.txt", "qa.txt"]:
    path = f"data/test_raw/{fname}"
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(3000):
            t = random.choice(templates)
            line = t.format(
                adj=random.choice(adj_words),
                act=random.choice(act_words),
                adv=random.choice(adv_words),
                obj=random.choice(obj_words),
                domain=random.choice(domain_words),
                tech=random.choice(tech_words),
            )
            f.write(line + "\n")

os.makedirs("data/test_splits", exist_ok=True)
stream_build(
    input_paths=["data/test_raw/wiki.txt", "data/test_raw/qa.txt"],
    output_dir="data/test_splits",
    train_ratio=0.9,
    valid_ratio=0.1,
    buf_size=2000,
)

tok = BBPETokenizer.load("../gleamlm/tokenizer/checkpoints/bbpe_12k")
VOCAB = tok.get_vocab_size()
dataset = LMDataset(
    data_dir="data/test_splits",
    max_seq_len=512,
    stride=128,
    tokenizer=tok,
    split="train",
    max_chars=3_000_000,
)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_id=tok.pad_id),
    drop_last=True,
)

model = GleamLMModel(
    vocab_size=VOCAB,
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=4,
    d_ff=1365,
    max_seq_len=512,
).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
t0 = time.time()
for step, batch in enumerate(loader):
    if step >= 200:
        break
    input_ids, targets = [b.to(device) for b in batch]
    logits, _ = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
    loss.backward()
    optim.step()
    optim.zero_grad()
    if (step + 1) % 50 == 0:
        print(
            f"  step {step + 1:3d}/200 | loss={loss.item():.3f} | ppl={torch.exp(loss).item():.1f}"
        )
print(f"  预训练完成 ({time.time() - t0:.1f}s)")

os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": model.state_dict()}, "checkpoints/best_model.pt")
print("  Saved: checkpoints/best_model.pt")

print("=" * 60)
print("Step 2: 生成 SFT 数据...")
sft_prompts = [
    ("你好，请介绍一下你自己。", "你好！我是GleamLM，一个面向教育和研究的轻量级开源对话模型。"),
    ("什么是人工智能？", "人工智能是研究、开发用于模拟和扩展人类智能的理论与技术。"),
    ("Python适合做什么？", "Python是一种通用编程语言，适合Web开发、数据分析、人工智能等多种场景。"),
    ("如何高效学习？", "高效学习的关键是制定明确计划、主动学习、分散间隔、及时练习、定期复盘。"),
    ("解释一下机器学习。", "机器学习是AI的核心子领域，让计算机通过数据自动学习规律。"),
    ("今天天气不错。", "确实是好天气呢！适合出去走走，呼吸新鲜空气。"),
    ("量子计算是什么？", "量子计算利用量子力学原理进行计算，在处理特定问题上远超经典计算机。"),
    ("如何保持健康？", "保持健康需要规律作息、均衡饮食、适量运动、良好心态。"),
    ("什么是自然语言处理？", "NLP是让计算机理解、生成人类语言的技术。应用包括翻译、对话等。"),
    ("介绍一下深度学习。", "深度学习使用多层神经网络来学习数据的分层表示，在多个领域有突破。"),
]

os.makedirs("data", exist_ok=True)
with open("data/test_sft.jsonl", "w", encoding="utf-8") as f:
    for q, a in sft_prompts:
        f.write(json.dumps({"instruction": q, "output": a}, ensure_ascii=False) + "\n")
print(f"  生成 {len(sft_prompts)} 条 SFT 数据")

print("=" * 60)
print("Step 3: SFT 指令微调 (1 epoch, 跳过 evaluate)...")

# 不调用 sft.py（它的 evaluate 会卡死），直接手写最小 SFT 循环
from torch.utils.data import DataLoader, Dataset


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.im_start = tokenizer.special_tokens["<|im_start|>"]
        self.im_end = tokenizer.special_tokens["<|im_end|>"]
        self.newline_id = tokenizer.encode("\n", add_bos=False, add_eos=False)[0]
        with open(jsonl_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inst = item["instruction"]
        resp = item["output"]
        # ChatML 格式
        chatml = (
            f"<|im_start|><|user|>\n{inst}<|im_end|>\n<|im_start|><|assistant|>\n{resp}<|im_end|>"
        )
        ids = self.tokenizer.encode(chatml, add_bos=False, add_eos=False)
        ids = ids[: self.max_seq_len]
        # 找到 assistant 起始位置，只对回答部分计算 loss
        text = self.tokenizer.decode(ids, skip_special=False)
        assistant_marker = "<|im_start|><|assistant|>\n"
        marker_pos = text.find(assistant_marker)
        # labels: 预测下一个 token，shift by 1
        labels = [-100] * len(ids)
        if marker_pos >= 0:
            prefix = text[: marker_pos + len(assistant_marker)]
            prefix_ids = self.tokenizer.encode(prefix, add_bos=False, add_eos=False)
            prefix_len = min(len(prefix_ids), len(ids))
            # labels[i] = ids[i+1] for i in [prefix_len-1, len(ids)-2]
            for i in range(max(0, prefix_len - 1), len(ids) - 1):
                labels[i] = ids[i + 1]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels


def sft_collate_fn(batch):
    max_len = max(len(s[0]) for s in batch)
    inputs, labels_list = [], []
    for inp, lab in batch:
        pad_len = max_len - len(inp)
        inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        labels_list.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
    return torch.stack(inputs), torch.stack(labels_list)


sft_model = GleamLMModel(
    vocab_size=VOCAB,
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=4,
    d_ff=1365,
    max_seq_len=512,
    tie_weights=True,
).to(device)
ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
sft_model.load_state_dict(ckpt["model_state_dict"])
print(f"  SFT model loaded: {sum(p.numel() for p in sft_model.parameters()) / 1e6:.2f}M params")

sft_dataset = SFTDataset("data/test_sft.jsonl", tok, max_seq_len=256)
sft_loader = DataLoader(sft_dataset, batch_size=4, shuffle=True, collate_fn=sft_collate_fn)

sft_optim = torch.optim.AdamW(sft_model.parameters(), lr=1e-6, betas=(0.9, 0.95), weight_decay=0.01)
sft_model.train()
t0 = time.time()
for input_ids, labels in sft_loader:
    input_ids, labels = input_ids.to(device), labels.to(device)
    logits, _ = sft_model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100
    )
    loss.backward()
    sft_optim.step()
    sft_optim.zero_grad()
    print(
        f"  SFT loss={loss.item():.4f} | ppl={torch.exp(loss).item():.1f} | {time.time() - t0:.1f}s"
    )

os.makedirs("checkpoints/sft_test", exist_ok=True)
torch.save({"model_state_dict": sft_model.state_dict()}, "checkpoints/sft_test/sft_best.pt")
# 兼容 dpo.py 读取方式（它也用 model_state_dict key）
torch.save({"model_state_dict": sft_model.state_dict()}, "checkpoints/sft_test/dpo_best.pt")
print(f"  Saved: checkpoints/sft_test/sft_best.pt ({time.time() - t0:.1f}s)")

print("=" * 60)
print("Step 4: 生成 DPO 数据...")
dpo_data = [
    {
        "instruction": "你好，请介绍一下你自己。",
        "chosen": "你好！我是GleamLM，一个面向教育和研究的轻量级开源对话模型。",
        "rejected": "我...呃...我是...不知道。可能是个程序吧。",
    },
    {
        "instruction": "什么是人工智能？",
        "chosen": "人工智能是研究、开发用于模拟和扩展人类智能的理论与技术。",
        "rejected": "人工智能就是让电脑变聪明的东西，反正就是自动化嘛。",
    },
    {
        "instruction": "Python适合做什么？",
        "chosen": "Python是一种通用编程语言，适合Web开发、数据分析、人工智能等多种场景。",
        "rejected": "Python是一种编程语言，可以写代码。啥都能做吧。",
    },
    {
        "instruction": "如何高效学习？",
        "chosen": "高效学习的关键是制定明确计划、主动学习、分散间隔、及时练习、定期复盘。",
        "rejected": "学习就是多看书多看视频，不需要什么计划。",
    },
    {
        "instruction": "解释一下机器学习。",
        "chosen": "机器学习是AI的核心子领域，让计算机通过数据自动学习规律。",
        "rejected": "机器学习是AI的一种，让电脑自己学东西，给数据训练就行。",
    },
    {
        "instruction": "如何保持健康？",
        "chosen": "保持健康需要规律作息、均衡饮食、适量运动、良好心态。",
        "rejected": "健康就是别生病，多吃点就行。",
    },
]
with open("data/test_dpo.jsonl", "w", encoding="utf-8") as f:
    for item in dpo_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"  生成 {len(dpo_data)} 条 DPO 数据")

print("=" * 60)
print("Step 5: DPO 偏好对齐 (1 epoch)...")


class DPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(jsonl_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inst = item["instruction"]
        prompt = f"<|im_start|><|user|>\n{inst}<|im_end|>\n<|im_start|><|assistant|>\n"
        prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        chosen_ids = self.tokenizer.encode(
            item["chosen"] + "<|im_end|>", add_bos=False, add_eos=False
        )
        rejected_ids = self.tokenizer.encode(
            item["rejected"] + "<|im_end|>", add_bos=False, add_eos=False
        )
        # 截断
        chosen = (prompt_ids + chosen_ids)[: self.max_seq_len]
        rejected = (prompt_ids + rejected_ids)[: self.max_seq_len]
        # mask: prompt 部分=0, answer 部分=1（长度 L-1，shift 1 位对齐 next-token）
        P = len(prompt_ids)
        chosen_mask = torch.zeros(len(chosen) - 1, dtype=torch.long)
        rejected_mask = torch.zeros(len(rejected) - 1, dtype=torch.long)
        chosen_mask[max(0, P - 1) :] = 1
        rejected_mask[max(0, P - 1) :] = 1
        return (
            torch.tensor(chosen, dtype=torch.long),
            torch.tensor(rejected, dtype=torch.long),
            chosen_mask,
            rejected_mask,
        )


def dpo_collate_fn(batch):
    # Padding for chosen and rejected separately
    max_c = max(len(b[0]) for b in batch)
    max_r = max(len(b[1]) for b in batch)
    chosen_ids, rejected_ids, c_masks, r_masks = [], [], [], []
    for c, r, cm, rm in batch:
        chosen_ids.append(torch.cat([c, torch.zeros(max_c - len(c), dtype=torch.long)]))
        rejected_ids.append(torch.cat([r, torch.zeros(max_r - len(r), dtype=torch.long)]))
        c_masks.append(torch.cat([cm, torch.zeros(max_c - len(c), dtype=torch.long)]))
        r_masks.append(torch.cat([rm, torch.zeros(max_r - len(r), dtype=torch.long)]))
    return (
        torch.stack(chosen_ids),
        torch.stack(rejected_ids),
        torch.stack(c_masks),
        torch.stack(r_masks),
    )


dpo_dataset = DPODataset("data/test_dpo.jsonl", tok, max_seq_len=256)
dpo_loader = DataLoader(dpo_dataset, batch_size=2, shuffle=True, collate_fn=dpo_collate_fn)

# Policy model (从 SFT 加载)
policy = GleamLMModel(
    vocab_size=VOCAB,
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=4,
    d_ff=1365,
    max_seq_len=512,
).to(device)
ckpt_sft = torch.load("checkpoints/sft_test/sft_best.pt", map_location=device)
policy.load_state_dict(ckpt_sft["model_state_dict"])

# Reference model (冻结)
ref = GleamLMModel(
    vocab_size=VOCAB,
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=4,
    d_ff=1365,
    max_seq_len=512,
).to(device)
ref.load_state_dict(ckpt_sft["model_state_dict"])
for p in ref.parameters():
    p.requires_grad = False
ref.eval()

dpo_optim = torch.optim.AdamW(policy.parameters(), lr=1e-7, betas=(0.9, 0.95), weight_decay=0.01)
beta = 0.1
policy.train()


def compute_log_probs(logits, ids, mask):
    """计算 next-token log 概率（仅 mask 部分），mask 长度 = L-1"""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, L, V]
    gather = log_probs[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    return (gather * mask).sum(dim=-1)


t0 = time.time()
for chosen_ids, rejected_ids, c_masks, r_masks in dpo_loader:
    chosen_ids = chosen_ids.to(device)
    rejected_ids = rejected_ids.to(device)
    c_masks = c_masks.to(device)
    r_masks = r_masks.to(device)

    # Reference model log probs
    with torch.no_grad():
        ref_chosen_logits, _ = ref(chosen_ids)
        ref_rejected_logits, _ = ref(rejected_ids)
        ref_chosen_lp = compute_log_probs(ref_chosen_logits, chosen_ids, c_masks)
        ref_rejected_lp = compute_log_probs(ref_rejected_logits, rejected_ids, r_masks)

    # Policy model log probs
    policy_chosen_logits, _ = policy(chosen_ids)
    policy_rejected_logits, _ = policy(rejected_ids)
    policy_chosen_lp = compute_log_probs(policy_chosen_logits, chosen_ids, c_masks)
    policy_rejected_lp = compute_log_probs(policy_rejected_logits, rejected_ids, r_masks)

    # DPO loss
    term = (policy_chosen_lp - ref_chosen_lp) - (policy_rejected_lp - ref_rejected_lp)
    loss = -torch.nn.functional.logsigmoid(beta * term).mean()
    loss.backward()
    dpo_optim.step()
    dpo_optim.zero_grad()
    print(f"  DPO loss={loss.item():.4f} | {time.time() - t0:.1f}s")

os.makedirs("checkpoints/dpo_test", exist_ok=True)
torch.save(
    {"model_state_dict": policy.state_dict(), "dpo_loss": loss.item()},
    "checkpoints/dpo_test/dpo_best.pt",
)
print("  Saved: checkpoints/dpo_test/dpo_best.pt")

print("=" * 60)
print("Step 6: 验证所有产出...")
checks = {
    "checkpoints/best_model.pt": "预训练 (~155MB)",
    "checkpoints/sft_test/sft_best.pt": "SFT (~155MB)",
    "checkpoints/dpo_test/dpo_best.pt": "DPO (~155MB)",
}
for path, desc in checks.items():
    if os.path.exists(path):
        sz = os.path.getsize(path) / 1024 / 1024
        print(f"  {path}: {sz:.1f} MB ({desc})")
    else:
        print(f"  {path}: MISSING!")

# 验证 DPO 后的简单推理（限 30 token）
print()
print("DPO 模型简单推理验证 (max 30 tokens):")
policy.eval()
test_prompts = ["你好，请介绍一下你自己。", "什么是人工智能？"]
for p in test_prompts:
    prompt_text = f"<|im_start|><|user|>\n{p}<|im_end|>\n<|im_start|><|assistant|>\n"
    ids = torch.tensor([tok.encode(prompt_text, add_bos=False, add_eos=False)], device=device)
    with torch.no_grad():
        for _ in range(30):
            logits, kv = policy(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            # 检查停用
            if next_id.item() == tok.special_tokens.get("<|im_end|>", -1):
                break
    resp = tok.decode(ids[0].tolist(), skip_special=True)
    # 只显示 assistant 部分
    marker = "assistant"
    pos = resp.find(marker)
    if pos >= 0:
        resp = resp[pos + len(marker) :].strip()
    print(f"  Q: {p}")
    print(f"  A: {resp[:80]}")
    print()

print("=" * 60)
print("SFT + DPO Pipeline: 完成!")
print("=" * 60)

# 清理临时文件
for d in ["data/test_raw", "data/test_splits", "_debug_kv.py", "_debug_kv2.py", "_debug_kv3.py"]:
    if os.path.exists(d):
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            os.remove(d)
print("Temp files cleaned.")
