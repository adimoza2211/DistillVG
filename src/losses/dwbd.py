from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_dwbd_alpha(
    epoch_index: int,
    total_epochs: int,
    alpha_max: float = 0.7,
    alpha_min: float = 0.1,
    gamma: float = 2.0,
) -> float:
    """Compute DWBD alpha schedule.

    Alpha decays from alpha_max to alpha_min over training,
    shifting trust from decoder predictions toward ground truth.
    """
    if total_epochs <= 1:
        return float(alpha_max)
    progress = max(0.0, min(1.0, epoch_index / float(total_epochs - 1)))
    decay = (1.0 - progress) ** gamma
    return float((alpha_max - alpha_min) * decay + alpha_min)


def _paired_generalized_box_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute pairwise GIoU between predicted and target boxes [B, 4] in xyxy format."""
    inter_x1 = torch.maximum(pred[..., 0], target[..., 0])
    inter_y1 = torch.maximum(pred[..., 1], target[..., 1])
    inter_x2 = torch.minimum(pred[..., 2], target[..., 2])
    inter_y2 = torch.minimum(pred[..., 3], target[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)

    pred_area = (pred[..., 2] - pred[..., 0]).clamp_min(0) * (pred[..., 3] - pred[..., 1]).clamp_min(0)
    tgt_area = (target[..., 2] - target[..., 0]).clamp_min(0) * (target[..., 3] - target[..., 1]).clamp_min(0)
    union = (pred_area + tgt_area - inter_area).clamp_min(1e-8)
    iou = inter_area / union

    enc_x1 = torch.minimum(pred[..., 0], target[..., 0])
    enc_y1 = torch.minimum(pred[..., 1], target[..., 1])
    enc_x2 = torch.maximum(pred[..., 2], target[..., 2])
    enc_y2 = torch.maximum(pred[..., 3], target[..., 3])
    enc_area = ((enc_x2 - enc_x1).clamp_min(0) * (enc_y2 - enc_y1).clamp_min(0)).clamp_min(1e-8)

    return iou - ((enc_area - union) / enc_area)


class DWBDBoxLoss(nn.Module):
    """Dynamic Weight-Balance Distillation for box regression.

    Distills from the decoder branch to the MLP branch by blending
    decoder predictions and GT as the MLP's target.
    
    Following SimVG (NeurIPS 2024):
    - alpha dynamically decays during training
    - Early training: MLP learns from decoder (alpha high)
    - Late training: MLP learns from GT directly (alpha low)
    """

    def forward(
        self,
        mlp_bbox: torch.Tensor,
        decoder_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Compute DWBD box distillation loss.

        Args:
            mlp_bbox: [B, 4] — MLP branch predictions (sigmoid, xyxy normalized).
            decoder_bbox: [B, 4] — Decoder branch predictions (detached).
            gt_bbox: [B, 4] — Ground truth boxes (xyxy normalized).
            alpha: Blending coefficient (0 = pure GT, 1 = pure decoder).

        Returns:
            Scalar loss.
        """
        # Blended target: alpha × decoder_pred + (1-alpha) × GT.
        target = alpha * decoder_bbox.detach() + (1.0 - alpha) * gt_bbox

        # L1 + GIoU loss.
        l1 = F.l1_loss(mlp_bbox, target)
        giou = 1.0 - _paired_generalized_box_iou(mlp_bbox, target).mean()

        return l1 + giou


# Backward compatibility alias.
DWBDLoss = DWBDBoxLoss
