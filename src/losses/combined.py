from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.losses.consistency import consistency_loss
from src.losses.dwbd import DWBDLoss
from src.losses.verifier_kl import verifier_kl_loss


@dataclass
class GroundingLossOutput:
    dwbd: torch.Tensor
    cst: torch.Tensor
    ver_kl: torch.Tensor


class GroundingLoss(nn.Module):
    """Computes individual distillation loss components.

    Weighting and total computation are handled by the caller (distill.py),
    which applies STAL, ProgLoss, and per-term lambda scaling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dwbd_loss = DWBDLoss()

    def forward(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_soft: torch.Tensor,
        gt_one_hot: torch.Tensor,
        alpha: float,
        verifier_mask: torch.Tensor,
        attention_maps: list[torch.Tensor] | None = None,
    ) -> GroundingLossOutput:
        l_dwbd = self.dwbd_loss(student_logits, teacher_soft, gt_one_hot, alpha)
        l_ver = verifier_kl_loss(student_logits, teacher_soft, verifier_mask)
        l_cst = consistency_loss(attention_maps or [])
        return GroundingLossOutput(dwbd=l_dwbd, cst=l_cst, ver_kl=l_ver)
