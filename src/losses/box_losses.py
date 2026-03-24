from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou


def _normalize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2]).clamp_min(x1 + 1e-4)
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3]).clamp_min(y1 + 1e-4)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    pred_xyxy = _normalize_xyxy(pred_boxes)
    gt_xyxy = _normalize_xyxy(gt_boxes)
    l1 = F.l1_loss(pred_xyxy, gt_xyxy, reduction="none").mean(dim=-1)
    giou = 1.0 - generalized_box_iou(pred_xyxy, gt_xyxy).diagonal()
    loss = l1 + giou

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
