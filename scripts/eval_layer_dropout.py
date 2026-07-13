"""B3: Layer Dropout Simulation — 评估减少层数对生成质量的影响

模拟方法：加载 12 层模型，推理时屏蔽最上层 N 层（跳过），
观察生成文本质量随屏蔽层数增加而退化的速度。

N=0 (完整12L) → N=1 (11L) → N=2 (10L) → N=3 (9L) → N=4 (8L)
"""

import os
import sys

import torch
import torch.nn as nn

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "gleamlm-nano", "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = f"{DEFAULT_CHECKPOINT_DIR}/best_model.pt"
TOKENIZER_PATH = DEFAULT_TOKENIZER_PATH
OUTPUT_FILE = "scripts/eval_layer_result.txt"

# Test prompts covering different aspects
TEST_PROMPTS = [
    ("短句续写", "人工智能是"),
    ("长句续写", "深度学习通过多层神经网络"),
    ("知识问答", "世界上最高的山峰是"),
    ("常识推理", "如果明天下雨，那么"),
    ("逻辑连接", "因为小明忘记带钥匙，所以"),
    (
        "长距离依赖",
        "在遥远的古代，人们使用各种方法记录信息。其中最著名的发明之一是造纸术，它最早出现在",
    ),
]


class LayerLimitedModel(nn.Module):
    """Wrapper that limits the number of active layers"""

    def __init__(self, original_model, num_active_layers):
        super().__init__()
        self.original = original_model
        self.num_active_layers = num_active_layers
        self.d_model = original_model.d_model

    def forward(self, input_ids, past_kv_list=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        x = self.original.token_embed(input_ids)
        x = self.original.emb_dropout(x)

        if past_kv_list is None:
            attn_mask = self.original._create_attn_mask(seq_len, device)
        else:
            attn_mask = None

        new_kv_list = []
        for i, layer in enumerate(self.original.layers):
            if i >= self.num_active_layers:
                # Bypass: just pass through with identity (dummy KV on correct device)
                current_kv = (torch.zeros(1, device=device), torch.zeros(1, device=device))
                new_kv_list.append(current_kv)
                continue
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, current_kv = layer(x, attn_mask, past_kv)
            new_kv_list.append(current_kv)

        x = self.original.final_norm(x)
        logits = self.original.lm_head(x)
        return logits, new_kv_list

    def get_num_params(self):
        return self.original.get_num_params()


def load_model(tokenizer):
    print(f"Loading pretrained model: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    args = checkpoint.get("args", None)

    if args:
        m = GleamLMModel(
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
        )
    else:
        m = GleamLMModel(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=512,
            num_layers=12,
            num_heads=8,
            num_kv_heads=4,
            d_ff=1365,
            dropout=0.1,
            max_seq_len=1024,
            pad_token_id=tokenizer.pad_id,
            tie_weights=True,
        )

    state = checkpoint["model_state_dict"]
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "")
        new_state[new_k] = v
    m.load_state_dict(new_state, strict=False)
    m.to(DEVICE)
    m.eval()
    return m


def generate_text(model, tokenizer, prompt, max_new_tokens=80):
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], device=DEVICE)

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token = next_token_logits.argmax(dim=-1).item()
            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]], device=DEVICE)

    return tokenizer.decode(generated)


def run_layer_dropout_test():
    tok = BBPETokenizer.load(TOKENIZER_PATH)
    base_model = load_model(tok)

    print(f"Model: 12 layers, {sum(p.numel() for p in base_model.parameters()):,} params")
    print("\n" + "=" * 70)
    print("B3: LAYER DROPOUT SIMULATION")
    print("=" * 70)

    all_results = {}

    layer_configs = [12, 11, 10, 9, 8]
    for num_layers in layer_configs:
        print(f"\n--- {num_layers} Active Layers (dropped {12 - num_layers}) ---")
        model = LayerLimitedModel(base_model, num_layers)

        results = []
        for category, prompt in TEST_PROMPTS:
            generated = generate_text(model, tok, prompt, max_new_tokens=50)
            results.append((category, prompt, generated))
            print(f"  [{category}] {prompt} -> {generated[:80]}...")

        all_results[num_layers] = results

    # Quality degradation scoring
    print("\n" + "=" * 70)
    print("DEGRADATION ANALYSIS")
    print("=" * 70)
    print(f"{'Layers':<8} {'Short Text':<12} {'Long Text':<12} {'Knowledge':<12} {'Reasoning':<12}")
    print("-" * 56)

    for num_layers in layer_configs:
        results = all_results[num_layers]
        # Simple heuristic: count unique chars as diversity proxy
        unique_chars = sum(len(set(r[2])) for r in results)
        avg_len = sum(len(r[2]) for r in results) / len(results)
        repetition_count = sum(1 for r in results if _has_repetition(r[2]))
        print(
            f"{num_layers:<8} {unique_chars:<12} {avg_len:<12.1f} {'N/A':<12} {repetition_count}/{len(results)} rep"
        )

    return all_results


def _has_repetition(text, max_rep=3):
    """Check if text has token repetition loops (e.g. consecutive repeated words)."""
    words = text.split()
    if len(words) < max_rep:
        return False
    # 检查连续重复词（如 "啊 啊 啊"）
    for i in range(len(words) - max_rep + 1):
        window = words[i : i + max_rep]
        if len(set(window)) == 1:
            return True
    # 检查短周期循环（如 "A B A B A B"）
    if len(words) >= 6:
        for i in range(len(words) - 5):
            if (
                words[i] == words[i + 2] == words[i + 4]
                and words[i + 1] == words[i + 3] == words[i + 5]
            ):
                return True
    return False


def main():
    sys.stdout = open(OUTPUT_FILE, "w", encoding="utf-8")
    results = run_layer_dropout_test()
    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    r12 = results[12]
    r8 = results[8]
    for (_cat12, p12, g12), (_cat8, _p8, g8) in zip(r12, r8, strict=False):
        if "山峰" in p12:
            print(f"  12L: {g12[:100]}")
            print(f"  8L:  {g8[:100]}")
            break

    print("\n  (Higher unique_chars = better diversity, lower rep = better coherence)")


if __name__ == "__main__":
    main()
