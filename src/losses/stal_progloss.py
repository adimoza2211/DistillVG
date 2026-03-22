from __future__ import annotations

import torch


def compute_stal_weight(
    gt_boxes_xyxy: torch.Tensor,
    *,
    area_threshold: float,
    max_boost: float,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return torch.ones((), device=gt_boxes_xyxy.device, dtype=gt_boxes_xyxy.dtype)

    x1 = torch.minimum(gt_boxes_xyxy[..., 0], gt_boxes_xyxy[..., 2])
    y1 = torch.minimum(gt_boxes_xyxy[..., 1], gt_boxes_xyxy[..., 3])
    x2 = torch.maximum(gt_boxes_xyxy[..., 0], gt_boxes_xyxy[..., 2])
    y2 = torch.maximum(gt_boxes_xyxy[..., 1], gt_boxes_xyxy[..., 3])
    areas = ((x2 - x1).clamp_min(1e-6) * (y2 - y1).clamp_min(1e-6)).clamp_min(1e-6)

    threshold = max(float(area_threshold), 1e-6)
    ratio = (threshold / areas).clamp(min=1.0)
    per_sample_weight = ratio.sqrt().clamp(max=max(float(max_boost), 1.0))
    return per_sample_weight.mean()


def compute_progloss_scale(
    *,
    epoch_index: int,
    total_epochs: int,
    start_scale: float,
    end_scale: float,
    power: float,
    enabled: bool,
) -> float:
    if not enabled:
        return 1.0
    if total_epochs <= 1:
        return float(end_scale)

    progress = max(0.0, min(1.0, epoch_index / float(total_epochs - 1)))
    adjusted_progress = progress ** max(float(power), 1e-6)
    return float(start_scale + (end_scale - start_scale) * adjusted_progress)
