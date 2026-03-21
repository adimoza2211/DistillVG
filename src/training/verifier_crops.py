from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_predicted_boxes(pred_box: torch.Tensor) -> torch.Tensor:
    normalized = pred_box.sigmoid()
    x1 = torch.minimum(normalized[..., 0], normalized[..., 2])
    y1 = torch.minimum(normalized[..., 1], normalized[..., 3])
    x2 = torch.maximum(normalized[..., 0], normalized[..., 2]).clamp_min(x1 + 1e-4)
    y2 = torch.maximum(normalized[..., 1], normalized[..., 3]).clamp_min(y1 + 1e-4)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def crop_and_resize_from_boxes(images: torch.Tensor, boxes_xyxy: torch.Tensor, output_size: int) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    crops: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        x1, y1, x2, y2 = boxes_xyxy[batch_index]
        left = int(torch.floor(x1 * width).item())
        top = int(torch.floor(y1 * height).item())
        right = int(torch.ceil(x2 * width).item())
        bottom = int(torch.ceil(y2 * height).item())

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        crop = images[batch_index : batch_index + 1, :, top:bottom, left:right]
        resized = F.interpolate(
            crop,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
        crops.append(resized)

    return torch.cat(crops, dim=0)
