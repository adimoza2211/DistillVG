from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.losses.box_losses import box_loss
from src.losses.consistency import consistency_loss
from src.losses.dwbd import DWBDLoss
from src.losses.verifier_kl import verifier_kl_loss


@dataclass
class GroundingLossOutput:
    total: torch.Tensor
    dwbd: torch.Tensor
    cst: torch.Tensor
    ver_kl: torch.Tensor
    box: torch.Tensor


class GroundingLoss(nn.Module):
    def __init__(self, lambda1: float = 0.1, lambda2: float = 1.0, lambda3: float = 5.0) -> None:
        super().__init__()
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.dwbd_loss = DWBDLoss()

    def forward(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_soft: torch.Tensor,
        gt_one_hot: torch.Tensor,
        alpha: float,
        verifier_mask: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        attention_maps: list[torch.Tensor] | None = None,
    ) -> GroundingLossOutput:
        l_dwbd = self.dwbd_loss(student_logits, teacher_soft, gt_one_hot, alpha)
        l_ver = verifier_kl_loss(student_logits, teacher_soft, verifier_mask)
        l_box = box_loss(pred_boxes, gt_boxes)
        l_cst = consistency_loss(attention_maps or [])
        total = l_dwbd + self.lambda1 * l_cst + self.lambda2 * l_ver + self.lambda3 * l_box
        return GroundingLossOutput(total=total, dwbd=l_dwbd, cst=l_cst, ver_kl=l_ver, box=l_box)
