"""Dataset and collate_fn tests"""

import os

import pytest

from gleamlm.dataset.dataset import LMDataset, collate_fn
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "splits")


@pytest.fixture(scope="module")
def tokenizer():
    return BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)


def test_dataset_creation(tokenizer):
    ds = LMDataset(_BASE_DIR, tokenizer, 512, "valid")
    assert len(ds) > 0


def test_getitem_shape(tokenizer):
    ds = LMDataset(_BASE_DIR, tokenizer, 512, "valid")
    sample = ds[0]
    assert sample.dim() == 1
    assert 256 <= sample.size(0) <= 513  # max_seq_len + 1


def test_collate_fn(tokenizer):
    ds = LMDataset(_BASE_DIR, tokenizer, 512, "valid")
    samples = [ds[i] for i in range(min(4, len(ds)))]
    input_ids, target_ids = collate_fn(samples, pad_id=tokenizer.pad_id)
    assert input_ids.dim() == 2
    assert target_ids.dim() == 2
    assert input_ids.size(0) == len(samples)
    assert target_ids.size(0) == len(samples)
    assert target_ids.size(1) == input_ids.size(1)  # same seq len
