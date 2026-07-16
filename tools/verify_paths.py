"""Verify all import paths and config constants work correctly."""

import argparse
import os

from gleamlm.dataset.dataset import LMDataset
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, cfg_to_namespace, load_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

parser = argparse.ArgumentParser(description="路径验证")
parser.add_argument("--variant", choices=["nano", "lite", "pro"], default="nano")
args = parser.parse_args()

cfg = load_config(os.path.join(_PROJECT_ROOT, "configs", f"{args.variant}.yaml"))
ns = cfg_to_namespace(cfg, _PROJECT_ROOT)

print("=== Path Resolution ===")
print(f"Tokenizer: {DEFAULT_TOKENIZER_PATH}")
print(f"  exists: {os.path.exists(DEFAULT_TOKENIZER_PATH)}")
print(f"Checkpoint dir: {ns.checkpoint_dir}")
print(f"Data dir: {ns.data_dir}")
print(f"  exists: {os.path.exists(ns.data_dir)}")

tok = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
print(f"Tokenizer: vocab={tok.get_vocab_size()} OK")

m = GleamLMModel(
    vocab_size=ns.vocab_size,
    d_model=ns.d_model,
    num_layers=ns.num_layers,
    num_heads=ns.num_heads,
    num_kv_heads=ns.num_kv_heads,
    d_ff=ns.d_ff,
    dropout=ns.dropout,
    max_seq_len=ns.max_seq_len,
    pad_token_id=tok.pad_id,
    use_flash_attn=ns.use_flash_attn,
)
print(f"Model: {sum(p.numel() for p in m.parameters()):,} params OK")

ds = LMDataset(ns.data_dir, tok, 128, "valid")
print(f"Dataset: {len(ds)} samples OK")

print("\nAll imports and paths verified.")
