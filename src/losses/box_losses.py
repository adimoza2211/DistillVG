from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2]).clamp_min(x1 + 1e-4)
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3]).clamp_min(y1 + 1e-4)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _paired_generalized_box_iou(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    inter_y1 = torch.maximum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    inter_x2 = torch.minimum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    inter_y2 = torch.minimum(pred_xyxy[..., 3], gt_xyxy[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp_min(0.0) * (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp_min(0.0)
    gt_area = (gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp_min(0.0) * (gt_xyxy[..., 3] - gt_xyxy[..., 1]).clamp_min(0.0)
    union = (pred_area + gt_area - inter_area).clamp_min(1e-8)
    iou = inter_area / union

    enc_x1 = torch.minimum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    enc_y1 = torch.minimum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    enc_x2 = torch.maximum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    enc_y2 = torch.maximum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    enc_area = ((enc_x2 - enc_x1).clamp_min(0.0) * (enc_y2 - enc_y1).clamp_min(0.0)).clamp_min(1e-8)

    return iou - ((enc_area - union) / enc_area)


def box_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    pred_xyxy = _normalize_xyxy(pred_boxes)
    gt_xyxy = _normalize_xyxy(gt_boxes)
    l1 = F.l1_loss(pred_xyxy, gt_xyxy, reduction="none").mean(dim=-1)
    giou = 1.0 - _paired_generalized_box_iou(pred_xyxy, gt_xyxy)
    loss = l1 + giou

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
