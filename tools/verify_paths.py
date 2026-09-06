"""Verify all import paths and config constants work correctly."""

import argparse
import os

from gleamlm.data.dataset import tokenize_and_group
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, load_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

parser = argparse.ArgumentParser(description="路径验证")
parser.add_argument("--variant", choices=["nano", "lite", "pro"], default="nano")
args = parser.parse_args()

cfg = load_config(
    os.path.join(_PROJECT_ROOT, "manual", "configs", f"{args.variant}.yaml"), _PROJECT_ROOT
)

print("=== Path Resolution ===")
print(f"Tokenizer: {DEFAULT_TOKENIZER_PATH}")
print(f"  exists: {os.path.exists(DEFAULT_TOKENIZER_PATH)}")
print(f"Checkpoint dir: {cfg.data.checkpoint_dir}")
print(f"Data dir: {cfg.data.data_dir}")
print(f"  exists: {os.path.exists(cfg.data.data_dir)}")

tok = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
print(f"Tokenizer: vocab={tok.get_vocab_size()} OK")

m = GleamLMModel(
    vocab_size=cfg.model.vocab_size,
    d_model=cfg.model.d_model,
    num_layers=cfg.model.num_layers,
    num_heads=cfg.model.num_heads,
    num_kv_heads=cfg.model.num_kv_heads,
    d_ff=cfg.model.d_ff,
    dropout=cfg.model.dropout,
    max_seq_len=cfg.model.max_seq_len,
    pad_token_id=tok.pad_id,
    use_flash_attn=cfg.model.use_flash_attn,
)
print(f"Model: {sum(p.numel() for p in m.parameters()):,} params OK")

ds = tokenize_and_group(os.path.join(cfg.data.data_dir, "valid.txt"), tok, 128)
print(f"Dataset: {len(ds)} samples OK")

print("\nAll imports and paths verified.")
