"""数据集 创建/getitem/collate/增强 测试"""

import torch

from gleamlm.dataset.dataset import collate_fn


def test_collate_fn_basic(tokenizer):
    samples = [
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        torch.tensor([10, 20, 30], dtype=torch.long),
    ]
    input_ids, target_ids = collate_fn(samples, pad_id=tokenizer.pad_id)
    assert input_ids.shape == (2, 4)  # max_len=5 → input=4
    assert target_ids.shape == (2, 4)  # same
    # 第二个样本最后一位置应为 pad_id
    assert input_ids[1, 3].item() == tokenizer.pad_id


def test_collate_fn_shorter_padded(tokenizer):
    samples = [
        torch.tensor([7, 8, 9, 10], dtype=torch.long),
        torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
    ]
    input_ids, target_ids = collate_fn(samples, pad_id=tokenizer.pad_id)
    # target 是 input 右移一位
    assert torch.equal(target_ids[0, :3], input_ids[0, 1:4])


def test_collate_fn_pad_ignored_in_ce(tokenizer):
    """padding 位置的 target 应为 pad_id，CE 中会被 ignore_index 跳过"""
    samples = [torch.tensor([5, 6, 7, 8], dtype=torch.long), torch.tensor([1, 2], dtype=torch.long)]
    input_ids, target_ids = collate_fn(samples, pad_id=tokenizer.pad_id)
    # 第二个样本的有效 target 只有 1 个位置
    assert (target_ids[1, 1:] == tokenizer.pad_id).all()
