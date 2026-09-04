"""SFT + DPO 全链路快速冒烟测试。"""

import json
import os
import tempfile

import torch
from torch.utils.data import DataLoader

from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.trainer.base_trainer import evaluate_generations, set_seed
from gleamlm.trainer.dpo_loss import compute_log_probs, dpo_loss
from gleamlm.data.sft_data import SFTDataset
from gleamlm.data.dpo_data import DPODataset, dpad_collate
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

VOCAB_SIZE = 12002
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
D_FF = 512
MAX_SEQ_LEN = 64


def _make_model(device, tokenizer):
    return GleamLMModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        d_ff=D_FF,
        dropout=0.0,
        max_seq_len=MAX_SEQ_LEN,
        tie_weights=True,
        pad_token_id=tokenizer.pad_id,
    ).to(device)


class TestSFT:
    def test_sft_dataset(self):
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            data = [
                {"instruction": "什么是AI", "output": "人工智能是计算机科学的分支"},
                {"instruction": "你好", "output": "你好！有什么可以帮助你的？"},
                {"instruction": "推荐一道菜", "output": "西红柿炒鸡蛋简单好做"},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

            ds = SFTDataset(path, tokenizer, max_seq_len=MAX_SEQ_LEN)
            assert len(ds) == 3

            loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
            batch = next(iter(loader))
            assert len(batch) == 2  # input_ids, labels

    def test_sft_forward_backward(self):
        set_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            data = [
                {"instruction": "什么是AI", "output": "人工智能是计算机科学的分支"},
                {"instruction": "你好", "output": "你好！有什么可以帮助你的？"},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

            ds = SFTDataset(path, tokenizer, max_seq_len=MAX_SEQ_LEN)
            loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)

            model = _make_model(device, tokenizer)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

            model.train()
            for step, (input_ids, labels) in enumerate(loader):
                if step >= 10:
                    break
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits, _, _, _ = model(input_ids)
                ce = torch.nn.functional.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100
                )
                optimizer.zero_grad()
                ce.backward()
                optimizer.step()
            assert ce.item() > 0

            model.eval()
            evaluate_generations(model, tokenizer, ["你好"])


class TestDPO:
    def test_dpo_dataset(self):
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            data = [
                {
                    "instruction": "什么是AI",
                    "chosen": "人工智能是计算机科学的分支",
                    "rejected": "人工智能就是机器人",
                },
                {"instruction": "你好", "chosen": "你好！有什么可以帮助你的？", "rejected": "嗯"},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

            ds = DPODataset(path, tokenizer, max_seq_len=MAX_SEQ_LEN)
            assert len(ds) == 2

            loader = DataLoader(ds, batch_size=2, collate_fn=dpad_collate)
            batch = next(iter(loader))
            assert isinstance(batch, dict)
            assert "chosen_ids" in batch and "rejected_ids" in batch

    def test_dpo_loss(self):
        set_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            data = [
                {
                    "instruction": "什么是AI",
                    "chosen": "人工智能是计算机科学的分支",
                    "rejected": "人工智能就是机器人",
                },
                {"instruction": "你好", "chosen": "你好！有什么可以帮助你的？", "rejected": "嗯"},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

            ds = DPODataset(path, tokenizer, max_seq_len=MAX_SEQ_LEN)
            loader = DataLoader(ds, batch_size=2, collate_fn=dpad_collate)
            batch = next(iter(loader))

            policy = _make_model(device, tokenizer)
            ref = _make_model(device, tokenizer)
            ref.load_state_dict(policy.state_dict())
            for p in ref.parameters():
                p.requires_grad = False

            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            policy.train()
            c_logits, _, _, _ = policy(chosen_ids)
            r_logits, _, _, _ = policy(rejected_ids)
            p_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
            p_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)

            with torch.no_grad():
                c_logits_r, _, _, _ = ref(chosen_ids)
                r_logits_r, _, _, _ = ref(rejected_ids)
            r_cho = compute_log_probs(c_logits_r.float(), chosen_ids, chosen_mask)
            r_rej = compute_log_probs(r_logits_r.float(), rejected_ids, rejected_mask)

            loss = dpo_loss(p_cho, p_rej, r_cho, r_rej, beta=0.1)
            assert not torch.isnan(loss)
            assert loss.item() > 0
