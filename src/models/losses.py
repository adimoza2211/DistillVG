from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossOutput:
    total: torch.Tensor
    ce: torch.Tensor
    kl: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    mse: torch.Tensor
    attn: torch.Tensor
    cst: torch.Tensor
    ver: torch.Tensor


def compute_distillation_losses(
    pred_box: torch.Tensor,
    target_box: torch.Tensor,
    verifier_logit: torch.Tensor,
    verifier_target: torch.Tensor,
) -> LossOutput:
    l1 = torch.nn.functional.l1_loss(pred_box, target_box)
    ver = torch.nn.functional.binary_cross_entropy_with_logits(verifier_logit, verifier_target)

    zero = torch.zeros((), device=pred_box.device, dtype=pred_box.dtype)
    total = l1 + ver

    return LossOutput(
        total=total,
        ce=zero,
        kl=zero,
        l1=l1,
        giou=zero,
        mse=zero,
        attn=zero,
        cst=zero,
        ver=ver,
    )
