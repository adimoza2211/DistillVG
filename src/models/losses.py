from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import generalized_box_iou


@dataclass
class LossOutput:
    total: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    ver: torch.Tensor


def _box_loss(pred_box: torch.Tensor, target_box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_box = pred_box.sigmoid()
    x1 = torch.minimum(pred_box[..., 0], pred_box[..., 2])
    y1 = torch.minimum(pred_box[..., 1], pred_box[..., 3])
    x2 = torch.maximum(pred_box[..., 0], pred_box[..., 2]).clamp_min(x1 + 1e-4)
    y2 = torch.maximum(pred_box[..., 1], pred_box[..., 3]).clamp_min(y1 + 1e-4)
    pred_box = torch.stack([x1, y1, x2, y2], dim=-1)
    l1 = torch.nn.functional.l1_loss(pred_box, target_box)
    giou = 1.0 - generalized_box_iou(pred_box, target_box).diagonal().mean()
    return l1, giou


def compute_distillation_losses(
    pred_box: torch.Tensor,
    target_box: torch.Tensor,
    verifier_logit: torch.Tensor,
    verifier_target: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_giou: float = 1.0,
    lambda_ver: float = 1.0,
) -> LossOutput:
    l1, giou = _box_loss(pred_box, target_box)
    ver = torch.nn.functional.binary_cross_entropy_with_logits(verifier_logit, verifier_target)

    total = (
        lambda_l1 * l1
        + lambda_giou * giou
        + lambda_ver * ver
    )

    return LossOutput(
        total=total,
        l1=l1,
        giou=giou,
        ver=ver,
    )
