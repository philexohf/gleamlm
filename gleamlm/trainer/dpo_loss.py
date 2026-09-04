"""DPO (Direct Preference Optimization) loss functions.

Provides compute_log_probs, dpo_loss, get_reference_logps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gleamlm.utils.torch_utils import safe_autocast


def compute_log_probs(
    logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-token log probabilities, masked. Returns [B]."""
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs_token = log_probs_all[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (log_probs_token * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    term = (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)
    return -F.logsigmoid(beta * term).mean()


@torch.no_grad()
def get_reference_logps(
    ref_model: torch.nn.Module,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute chosen and rejected log-probs from frozen reference model."""
    ref_model.eval()
    with safe_autocast():
        c_logits, _, _, _ = ref_model(chosen_ids)
        r_logits, _, _, _ = ref_model(rejected_ids)
    ref_cho = compute_log_probs(c_logits.float(), chosen_ids, chosen_mask)
    ref_rej = compute_log_probs(r_logits.float(), rejected_ids, rejected_mask)
    return ref_cho, ref_rej
