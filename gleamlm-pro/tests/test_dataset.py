"""GleamLM-Pro 126M 数据集测试 — pytest"""
import os

import pytest

from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "pro_data")


@pytest.fixture(scope="module")
def tokenizer():
    return BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)


def test_dataset_creation(tokenizer):
    if not os.path.exists(os.path.join(_DATA_DIR, "valid.txt")):
        pytest.skip("data/pro_data/valid.txt not found")
    ds = LMDataset(_DATA_DIR, tokenizer, 4096, "valid", max_chars=500_000, augment=False)
    assert len(ds) > 0


def test_getitem_shape(tokenizer):
    if not os.path.exists(os.path.join(_DATA_DIR, "valid.txt")):
        pytest.skip("data/pro_data/valid.txt not found")
    ds = LMDataset(_DATA_DIR, tokenizer, 4096, "valid", max_chars=500_000, augment=False)
    sample = ds[0]
    assert sample.dim() == 1
    assert 1 <= sample.size(0) <= 4097


def test_collate_fn(tokenizer):
    if not os.path.exists(os.path.join(_DATA_DIR, "valid.txt")):
        pytest.skip("data/pro_data/valid.txt not found")
    ds = LMDataset(_DATA_DIR, tokenizer, 4096, "valid", max_chars=500_000, augment=False)
    samples = [ds[i] for i in range(min(4, len(ds)))]
    input_ids, target_ids = collate_fn(samples, pad_id=tokenizer.pad_id)
    assert input_ids.dim() == 2
    assert target_ids.dim() == 2
    assert input_ids.size(0) == len(samples)
    assert target_ids.size(0) == len(samples)
    assert target_ids.size(1) == input_ids.size(1)
