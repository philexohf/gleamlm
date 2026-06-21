"""用预训练基座模型对 SFT 问题生成差答案，作为 DPO rejected 数据"""
import json
import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.xfind_tokenizer import build_tokenizer
from models.xfind_model import XfindModel
from inference.sampler import sample_token


@torch.no_grad()
def generate_rejected(model, tokenizer, instruction, max_new_tokens=256,
                      temperature=0.8, top_k=50, top_p=0.9):
    """用预训练基座生成'差的'回答（作为 DPO rejected）"""
    model.eval()
    device = next(model.parameters()).device
    prompt_text = f"Q: {instruction}\nA:"
    prompt_ids = tokenizer.sp.encode(prompt_text, out_type=int)
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
        if token_id == tokenizer.eos_id:
            break
        if token_id == tokenizer.pad_id:
            break
        generated_ids.append(token_id)

        # 每 4 token 检查截断
        if not stopped and (i + 1) % 4 == 0:
            draft = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special=True)
            if "<|endoftext|>" in draft:
                stopped = True
                break

        next_input = torch.tensor([[token_id]], dtype=torch.long).to(device)
        with torch.amp.autocast('cuda'):
            logits, past_kv = model(next_input, past_kv_list=past_kv)

    response = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special=True)
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]
    model.train()
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_data", default="data/sft_data_clean.jsonl")
    parser.add_argument("--model_path", default="./checkpoints/best_model.pt")
    parser.add_argument("--output", default="data/dpo_data.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只生成前 N 条")
    args = parser.parse_args()

    print(f"Loading pretrained model: {args.model_path}")
    tokenizer = build_tokenizer(
        text_files=[],
        vocab_size=32000,
        model_prefix="./tokenizer/checkpoints/bpe_32k",
    )
    print(f"Tokenizer vocab: {len(tokenizer)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XfindModel(vocab_size=32000)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Load SFT instructions
    samples = []
    with open(args.sft_data, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]

    print(f"Generating rejected for {len(samples)} instructions...")

    dpo_data = []
    for i, s in enumerate(samples):
        instruction = s["instruction"]
        chosen = s["output"]
        rejected = generate_rejected(model, tokenizer, instruction)
        dpo_data.append({
            "instruction": instruction,
            "chosen": chosen,
            "rejected": rejected,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}]")
            # partial save
            with open(args.output + ".partial", "w", encoding="utf-8") as f:
                for item in dpo_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done -> {args.output} ({len(dpo_data)} pairs)")


if __name__ == "__main__":
    main()
