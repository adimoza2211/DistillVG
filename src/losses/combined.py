from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.box_losses import box_loss
from src.losses.consistency import consistency_loss
from src.losses.dwbd import DWBDBoxLoss


@dataclass
class DualBranchLossOutput:
    """Loss components from the dual-branch grounding loss."""
    total: torch.Tensor
    decoder_box: torch.Tensor
    dwbd: torch.Tensor
    alignment: torch.Tensor
    consistency: torch.Tensor


class DualBranchGroundingLoss(nn.Module):
    """Combined loss for SimVG-style dual-branch visual grounding.

    L_total = λ_box × (L1 + GIoU)(decoder_bbox, GT)          # Primary box loss
            + λ_dwbd × DWBD_box(mlp_bbox, decoder_bbox, GT)   # Branch distillation
            + λ_align × MSE(alignment_score, teacher_score)    # Verifier KD
            + λ_cst × ConsistencyLoss(fused_tokens)            # Feature regularization
    """

    def __init__(
        self,
        lambda_box: float = 5.0,
        lambda_dwbd: float = 2.0,
        lambda_align: float = 1.0,
        lambda_cst: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_dwbd = lambda_dwbd
        self.lambda_align = lambda_align
        self.lambda_cst = lambda_cst
        self.dwbd_loss = DWBDBoxLoss()

    def forward(
        self,
        *,
        decoder_bbox: torch.Tensor,
        mlp_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
        alpha: float,
        alignment_score: torch.Tensor | None = None,
        teacher_alignment: torch.Tensor | None = None,
        fused_tokens: torch.Tensor | None = None,
        box_scale: float = 1.0,
    ) -> DualBranchLossOutput:
        """Compute combined dual-branch loss.

        Args:
            decoder_bbox: [B, 4] — Decoder branch predictions.
            mlp_bbox: [B, 4] — MLP branch predictions.
            gt_bbox: [B, 4] — Ground truth boxes.
            alpha: DWBD blending coefficient.
            alignment_score: [B] — Student alignment score (optional).
            teacher_alignment: [B] — Teacher alignment score (optional).
            fused_tokens: [B, N, D] — Fused features for consistency (optional).
            box_scale: Progressive loss scale for box loss.

        Returns:
            DualBranchLossOutput with all loss components.
        """
        device = decoder_bbox.device

        # 1. Primary box loss (L1 + GIoU) on decoder branch.
        l_box = box_loss(decoder_bbox, gt_bbox, reduction="mean")

        # 2. DWBD branch distillation (decoder → MLP).
        l_dwbd = self.dwbd_loss(mlp_bbox, decoder_bbox, gt_bbox, alpha)

        # 3. Alignment loss (verifier KD).
        if alignment_score is not None and teacher_alignment is not None:
            l_align = F.mse_loss(alignment_score, teacher_alignment.to(alignment_score.dtype))
        else:
            l_align = torch.zeros((), device=device, dtype=decoder_bbox.dtype)

        # 4. Consistency loss.
        if fused_tokens is not None:
            l_cst = consistency_loss([fused_tokens])
        else:
            l_cst = torch.zeros((), device=device, dtype=decoder_bbox.dtype)

        # Total.
        total = (
            self.lambda_box * box_scale * l_box
            + self.lambda_dwbd * l_dwbd
            + self.lambda_align * l_align
            + self.lambda_cst * l_cst
        )

        return DualBranchLossOutput(
            total=total,
            decoder_box=l_box,
            dwbd=l_dwbd,
            alignment=l_align,
            consistency=l_cst,
        )


# Backward compatibility alias.
GroundingLoss = DualBranchGroundingLoss
