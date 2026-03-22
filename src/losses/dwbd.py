from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_dwbd_alpha(
    epoch_index: int,
    total_epochs: int,
    alpha_max: float = 0.8,
    alpha_min: float = 0.1,
    gamma: float = 2.0,
) -> float:
    if total_epochs <= 1:
        return float(alpha_max)
    progress = max(0.0, min(1.0, epoch_index / float(total_epochs - 1)))
    decay = (1.0 - progress) ** gamma
    return float((alpha_max - alpha_min) * decay + alpha_min)


class DWBDLoss(nn.Module):
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_soft: torch.Tensor,
        gt_one_hot: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        blended_target = alpha * teacher_soft + (1.0 - alpha) * gt_one_hot
        return F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            blended_target.clamp(min=1e-8),
            reduction="batchmean",
        )
