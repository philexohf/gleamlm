"""Verify all import paths and config constants work correctly."""

import os

from gleamlm.dataset.dataset import LMDataset
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer

# 包内常量（models/config.py 保留的路径常量）
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

# 本地默认路径（PyPI 安装后由用户通过 CLI 覆盖）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "gleamlm-nano", "checkpoints")
DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "nano_data")

print("=== Path Resolution ===")
print(f"Tokenizer: {DEFAULT_TOKENIZER_PATH}")
print(f"  exists: {os.path.exists(DEFAULT_TOKENIZER_PATH)}")
print(f"Checkpoint: {DEFAULT_CHECKPOINT_DIR}")
print(f"  exists: {os.path.exists(DEFAULT_CHECKPOINT_DIR)}")
print(f"Data: {DEFAULT_DATA_DIR}")
print(f"  exists: {os.path.exists(DEFAULT_DATA_DIR)}")

# Test tokenizer loading
tok = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
print(f"Tokenizer: vocab={tok.get_vocab_size()} OK")

# Test model creation (Nano 40M config)
m = GleamLMModel(
    vocab_size=12002,
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=4,
    d_ff=1365,
    dropout=0.0,
    max_seq_len=1024,
)
print(f"Model: {sum(p.numel() for p in m.parameters()):,} params OK")

# Test dataset
ds = LMDataset(DEFAULT_DATA_DIR, tok, 128, "valid")
print(f"Dataset: {len(ds)} samples OK")

print("\nAll imports and paths verified from project root CWD.")
