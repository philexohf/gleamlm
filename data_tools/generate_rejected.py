"""用预训练基座模型对 SFT 问题生成差答案，作为 DPO rejected 数据"""

import argparse
import json
import os

import torch

from gleamlm.inference.sampler import sample_token
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


@torch.no_grad()
def generate_rejected(
    model, tokenizer, instruction, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9
):
    """用预训练基座生成'差的'回答（作为 DPO rejected）"""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    prompt_text = f"<|im_start|><|user|>\n{instruction}<|im_end|>\n<|im_start|><|assistant|>\n"
    prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    generated_ids = prompt_ids.copy()
    stopped = False

    amp_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.amp.autocast(amp_device):
        logits, past_kv = model(prompt_tensor)

    for i in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        next_token = sample_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.15,
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
            draft = tokenizer.decode(generated_ids[len(prompt_ids) :], skip_special=True)
            if "<|endoftext|>" in draft:
                stopped = True
                break

        next_input = torch.tensor([[token_id]], dtype=torch.long).to(device)
        with torch.amp.autocast(amp_device):
            logits, past_kv = model(next_input, past_kv_list=past_kv)

    response = tokenizer.decode(generated_ids[len(prompt_ids) :], skip_special=True)
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]
    model.train(was_training)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_data", default="data/sft_data_clean.jsonl")
    parser.add_argument("--model_path", default="./checkpoints/best_model.pt")
    parser.add_argument("--output", default="data/dpo_data.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只生成前 N 条")
    args = parser.parse_args()

    print(f"Loading pretrained model: {args.model_path}")
    tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device)
    if "args" in ckpt:
        a = ckpt["args"]
        model = GleamLMModel(
            vocab_size=getattr(a, "vocab_size", 12002),
            d_model=getattr(a, "d_model", 512),
            num_layers=getattr(a, "num_layers", 12),
            num_heads=getattr(a, "num_heads", 8),
            num_kv_heads=getattr(a, "num_kv_heads", 4),
            d_ff=getattr(a, "d_ff", 1365),
            dropout=0.0,
            max_seq_len=getattr(a, "max_seq_len", 1024),
            pad_token_id=getattr(a, "pad_token_id", 0),
        )
    else:
        model = GleamLMModel(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=512,
            num_layers=12,
            num_heads=8,
            num_kv_heads=4,
            d_ff=1365,
            max_seq_len=1024,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Load SFT instructions
    samples = []
    with open(args.sft_data, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]

    print(f"Generating rejected for {len(samples)} instructions...")

    dpo_data = []
    for i, s in enumerate(samples):
        instruction = s["instruction"]
        chosen = s["output"]
        rejected = generate_rejected(model, tokenizer, instruction)
        dpo_data.append(
            {
                "instruction": instruction,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(samples)}]")
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
