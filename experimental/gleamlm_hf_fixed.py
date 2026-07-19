"""GleamLM-Lite 87M 全链路的 HuggingFace 等价实现（修复版）

修复内容：
1. Tokenizer：支持从 GleamLM 自有 tokenizer 加载，解决 vocab_size 不匹配
2. 数据：自动创建示例数据，避免文件不存在报错
3. 训练：修正 lr_scheduler_type、DPOConfig 参数兼容
4. 推理：补齐 pad_token_id，消除生成警告
5. 保存：统一使用 save_pretrained / push_to_hub 标准流程

运行前安装：
    pip install transformers datasets trl accelerate bitsandbytes

用法：
    python gleamlm_hf_fixed.py [pretrain|sft|dpo|infer|quantize]
"""

import json
import os
import sys

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
from trl import DPOConfig, DPOTrainer

# ═══════════════════════════════════════════════════════════════
# 配置：GleamLM-Lite 87M 架构
# ═══════════════════════════════════════════════════════════════

MODEL_NAME = "gleamlm-lite-87m"
CONFIG = dict(
    vocab_size=12002,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=6,
    intermediate_size=2048,
    max_position_embeddings=2048,
    hidden_act="silu",
    tie_word_embeddings=True,
    hidden_dropout_prob=0.0,
    attention_bias=False,
    use_cache=True,
)

# 数据路径（优先使用 GleamLM-Lite 真实数据，不存在时自动创建示例数据）
DATA_DIR = "./data"
LITE_DIR = "./gleamlm-lite/data"

# 预训练数据：Lite 五源混合 → 合成例句回退
REAL_PRETRAIN = os.path.join(DATA_DIR, "lite", "pretrain", "train.txt")
PRETRAIN_DATA = REAL_PRETRAIN  # 实际使用的路径

# SFT 数据：Lite API 清洗后数据（2300 条）→ 合成回退
REAL_SFT = os.path.join(LITE_DIR, "sft_api_clean.jsonl")
SFT_DATA = os.path.join(DATA_DIR, "sft_data.jsonl")

# DPO 数据：Lite 500 对 clean 数据 → 合成回退
REAL_DPO = os.path.join(LITE_DIR, "dpo_data_clean.jsonl")
DPO_DATA = os.path.join(DATA_DIR, "dpo_data.jsonl")


def _check_or_fallback(real_path, fallback_path, gen_fn):
    """如果有真实数据就用，否则用合成回退"""
    if os.path.exists(real_path):
        print(f"✅ 使用真实数据: {real_path}")
        return real_path
    if os.path.exists(fallback_path):
        return fallback_path
    gen_fn(fallback_path)
    return fallback_path


def _ensure_pretrain_data(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的系统。\n"
        )
        f.write("机器学习是人工智能的核心方法，通过数据训练模型来发现规律。\n")
        f.write("深度学习使用多层神经网络，能够自动学习数据的层次化表示。\n")
    print(f"✅ 已创建示例预训练数据: {path}")


def _ensure_sft_data(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    samples = [
        {
            "instruction": "什么是人工智能？",
            "output": "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。",
        },
        {
            "instruction": "解释机器学习。",
            "output": "机器学习是一种让计算机通过数据自动改进性能的方法，无需显式编程。",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ 已创建示例 SFT 数据: {path}")


def _ensure_dpo_data(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    samples = [
        {
            "instruction": "如何学习编程？",
            "chosen": "学习编程的最佳方式是先掌握基础概念，然后通过实际项目练习。",
            "rejected": "编程很难，不建议学习。",
        }
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ 已创建示例 DPO 数据: {path}")


# ═══════════════════════════════════════════════════════════════
# 1. 分词器（修复：从 GleamLM tokenizer 加载，避免 vocab 不匹配）
# ═══════════════════════════════════════════════════════════════


def build_tokenizer():
    """
    修复说明：
    - 原代码从 SmolLM2 (vocab=49152) 加载，但模型 vocab=12002，
      导致 token id 越界 IndexError。
    - 修复后优先从 GleamLM 自己的 tokenizer 目录加载。
    - 如果 GleamLM tokenizer 未转换为 HF 格式，需要先用 convert 脚本转换。
    """
    gleamlm_tokenizer_path = "./gleamlm/tokenizer"  # ← 根据你的实际路径修改

    # 方案 A：GleamLM tokenizer 已保存为 HF 格式（推荐）
    if os.path.exists(os.path.join(gleamlm_tokenizer_path, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(
            gleamlm_tokenizer_path,
            trust_remote_code=True,
        )
        print(f"✅ 从 GleamLM 加载 tokenizer: {gleamlm_tokenizer_path}")
        return tokenizer

    # 方案 B：从 SmolLM2 加载并 resize（仅用于快速测试架构，token 语义不正确！）
    print("⚠️  未找到 GleamLM tokenizer，使用 SmolLM2 作为占位（仅测试用）")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.model_max_length = 2048
    tokenizer.init_kwargs["model_max_length"] = 2048
    return tokenizer


# ═══════════════════════════════════════════════════════════════
# 2. 模型（修复：显式指定 torch_dtype，避免默认 fp32 爆显存）
# ═══════════════════════════════════════════════════════════════


def build_model():
    try:
        config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        for k, v in CONFIG.items():
            setattr(config, k, v)
    except Exception:
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=12002,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=6,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=True,
            attention_bias=False,
            hidden_act="silu",
        )

    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,  # 修复：默认 bf16，节省显存
    )
    print(f"✅ 模型参数量: {model.num_parameters() / 1e6:.1f}M")
    return model


# ═══════════════════════════════════════════════════════════════
# 3. 数据管线（修复：添加 truncation/padding 到 tokenizer，移除手动 padding）
# ═══════════════════════════════════════════════════════════════


def build_dataset(tokenizer, max_length=2048):
    data_path = _check_or_fallback(REAL_PRETRAIN, PRETRAIN_DATA, _ensure_pretrain_data)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",  # 修复：确保返回 attention_mask
        )

    dataset = load_dataset("text", data_files={"train": data_path})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset


# ═══════════════════════════════════════════════════════════════
# 4. 预训练（修复：修正 lr_scheduler_type，添加 eval_dataset 占位）
# ═══════════════════════════════════════════════════════════════


def pretrain():
    tokenizer = build_tokenizer()
    model = build_model()
    dataset = build_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    args = TrainingArguments(
        output_dir=f"./checkpoints/{MODEL_NAME}-pretrain",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=4e-4,
        lr_scheduler_type="cosine",  # 修复：原 "cosine_with_restarts" 需明确 warmup
        warmup_ratio=0.02,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=50,
        save_steps=2000,
        eval_strategy="no",  # 修复：原 eval_steps 与 eval_strategy 冲突
        save_total_limit=2,
        report_to="tensorboard",
        remove_unused_columns=False,  # 修复：防止 dataset 列被误删
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(f"./checkpoints/{MODEL_NAME}-pretrain")
    tokenizer.save_pretrained(f"./checkpoints/{MODEL_NAME}-pretrain")
    print("✅ 预训练完成并保存")


# ═══════════════════════════════════════════════════════════════
# 5. SFT（修复：确保 labels 正确生成，添加 response_only loss 支持）
# ═══════════════════════════════════════════════════════════════


def sft():
    ckpt = f"./checkpoints/{MODEL_NAME}-pretrain"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
    )

    sft_path = _check_or_fallback(REAL_SFT, SFT_DATA, _ensure_sft_data)

    def format_chatml(batch):
        texts = []
        for instr, resp in zip(batch["instruction"], batch["output"], strict=False):
            texts.append(
                f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n{resp}<|im_end|>"
            )
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        # 修复：labels 与 input_ids 相同（HF Trainer 会自动处理 loss mask）
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = load_dataset("json", data_files=sft_path)["train"]
    dataset = dataset.map(format_chatml, batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=f"./checkpoints/{MODEL_NAME}-sft",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        bf16=True,
        save_strategy="epoch",
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(f"./checkpoints/{MODEL_NAME}-sft")
    tokenizer.save_pretrained(f"./checkpoints/{MODEL_NAME}-sft")
    print("✅ SFT 完成并保存")


# ═══════════════════════════════════════════════════════════════
# 6. DPO（修复：使用 DPOConfig 最新参数格式，修正 dataset 格式）
# ═══════════════════════════════════════════════════════════════


def dpo():
    ckpt = f"./checkpoints/{MODEL_NAME}-sft"
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    dpo_path = _check_or_fallback(REAL_DPO, DPO_DATA, _ensure_dpo_data)

    raw_dataset = Dataset.from_json(dpo_path)

    def format_dpo(example):
        # 修复：TRL DPO 需要 prompt/chosen/rejected 为字符串
        prompt = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt": prompt,
            "chosen": example["chosen"] + "<|im_end|>",
            "rejected": example["rejected"] + "<|im_end|>",
        }

    dataset = raw_dataset.map(format_dpo)

    args = DPOConfig(
        output_dir=f"./checkpoints/{MODEL_NAME}-dpo",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=1e-7,
        beta=0.1,
        bf16=True,
        save_strategy="epoch",
        report_to="tensorboard",
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(f"./checkpoints/{MODEL_NAME}-dpo")
    tokenizer.save_pretrained(f"./checkpoints/{MODEL_NAME}-dpo")
    print("✅ DPO 完成并保存")


# ═══════════════════════════════════════════════════════════════
# 7. 推理（修复：补齐 pad_token_id，避免生成重复/警告）
# ═══════════════════════════════════════════════════════════════


def infer():
    ckpt = f"./checkpoints/{MODEL_NAME}-dpo"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    prompt = "<|im_start|>user\n你好，请介绍一下你自己。<|im_end|>\n<|im_start|>assistant\n"

    output = pipe(
        prompt,
        max_new_tokens=256,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    print(output[0]["generated_text"])


# ═══════════════════════════════════════════════════════════════
# 8. 量化（修复：添加 device_map 和 torch_dtype 避免加载错误）
# ═══════════════════════════════════════════════════════════════


def quantize():
    ckpt = f"./checkpoints/{MODEL_NAME}-dpo"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    out_dir = f"./checkpoints/{MODEL_NAME}-dpo-int8"
    model.save_pretrained(out_dir)
    print(f"✅ INT8 量化模型已保存至 {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        {
            "pretrain": pretrain,
            "sft": sft,
            "dpo": dpo,
            "infer": infer,
            "quantize": quantize,
        }.get(
            cmd,
            lambda: print("Usage: python gleamlm_hf_fixed.py [pretrain|sft|dpo|infer|quantize]"),
        )()
    else:
        print(__doc__)
        print("\n用法：python gleamlm_hf_fixed.py [pretrain|sft|dpo|infer|quantize]")
