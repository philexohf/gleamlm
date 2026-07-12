"""Generate multi-turn DPO data using SFT model to produce rejected responses.

Input: multi-turn SFT JSONL (messages format)
Output: DPO JSONL (messages + chosen + rejected format)

Usage:
    python data_tools/generate_rejected_multiturn.py \
        --input gleamlm-lite/data/sft_multiturn.jsonl \
        --model gleamlm-lite/checkpoints/sft6/sft_best.pt \
        --output gleamlm-lite/data/dpo_multiturn.jsonl \
        --limit 300
"""

import argparse
import json

import torch

from gleamlm import load_model_for_inference
from gleamlm.inference.generator import generate_tokens
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def load_data(input_path: str) -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]] = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping line {i}: invalid JSON")
                continue
            msgs = item.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 4:
                continue
            conversations.append(msgs)
    return conversations


def split_context_target(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str] | None:
    """Split into context (all but last assistant) and target (last assistant content)."""
    last = messages[-1]
    if last["role"] != "assistant":
        return None
    context = messages[:-1]
    target = last["content"]
    return context, target


def build_prompt(context: list[dict[str, str]], tokenizer) -> list[int]:
    """Build prompt IDs: conversation context + assistant header."""
    parts: list[str] = []
    for msg in context:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|><|{role}|>\n{content}<|im_end|>\n")
    parts.append("<|im_start|><|assistant|>\n")
    prompt_text = "".join(parts)
    return tokenizer.encode(prompt_text, add_bos=False, add_eos=False)


def generate_rejected(
    model,
    tokenizer,
    context: list[dict[str, str]],
    max_new_tokens=256,
    temperature=0.95,
    top_k=50,
    top_p=0.9,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = build_prompt(context, tokenizer)
    im_end_id = tokenizer.special_tokens.get("<|im_end|>")
    stop_ids: set[int] = {tokenizer.eos_id, tokenizer.pad_id}
    if im_end_id is not None:
        stop_ids.add(im_end_id)

    generated: list[int] = []
    for token_id in generate_tokens(
        model,
        prompt_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.15,
        penalty_window=50,
        stop_ids=stop_ids,
    ):
        generated.append(token_id)

    return tokenizer.decode(generated, skip_special=True)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn DPO rejected data")
    parser.add_argument("--input", required=True, help="Multi-turn SFT JSONL (messages format)")
    parser.add_argument("--model", required=True, help="SFT model checkpoint to generate rejected")
    parser.add_argument("--output", required=True, help="Output DPO JSONL path")
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading SFT model: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model, config = load_model_for_inference(args.model, device, checkpoint=checkpoint)
    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    conversations = load_data(args.input)
    if args.limit and args.limit < len(conversations):
        conversations = conversations[: args.limit]
    print(f"Loaded {len(conversations)} multi-turn conversations")

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, msgs in enumerate(conversations):
            result = split_context_target(msgs)
            if result is None:
                continue
            context, chosen = result
            if len(chosen) < 10:
                continue

            print(f"[{i + 1}/{len(conversations)}] generating rejected...", end=" ", flush=True)
            rejected = generate_rejected(
                model,
                tokenizer,
                context,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            if not rejected.strip():
                print("SKIP (empty)")
                continue

            entry = {
                "messages": context,
                "chosen": chosen,
                "rejected": rejected,
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            print(f"OK (chosen={len(chosen)}c, rejected={len(rejected)}c)")

    print(f"\nDone. {written} DPO samples written to {args.output}")


if __name__ == "__main__":
    main()
