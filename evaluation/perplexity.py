"""困惑度评估。PPL = exp(loss)，越低越好"""

import torch
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@torch.no_grad()
def compute_perplexity(model, data_loader, device='cuda'):
    """计算模型在数据集上的 PPL"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )

        total_loss += loss.item()
        total_tokens += (target_ids != 0).sum().item()  # 不计 padding

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)

    return avg_loss, ppl
