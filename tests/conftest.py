"""共享 fixtures — 所有测试模块复用"""

import pytest

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


@pytest.fixture(scope="session")
def tokenizer():
    return BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)


@pytest.fixture(scope="session")
def small_model():
    """4层×256dim 小模型，测试用"""
    model = GleamLMModel(
        vocab_size=12002,
        d_model=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        d_ff=682,
        dropout=0.0,
        max_seq_len=128,
        pad_token_id=0,
    )
    model.eval()
    return model
