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


def normalize_xyxy_boxes(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes_xyxy[..., 0], boxes_xyxy[..., 2]).clamp(0.0, 1.0)
    y1 = torch.minimum(boxes_xyxy[..., 1], boxes_xyxy[..., 3]).clamp(0.0, 1.0)
    x2 = torch.maximum(boxes_xyxy[..., 0], boxes_xyxy[..., 2]).clamp_min(x1 + 1e-4).clamp(max=1.0)
    y2 = torch.maximum(boxes_xyxy[..., 1], boxes_xyxy[..., 3]).clamp_min(y1 + 1e-4).clamp(max=1.0)
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


def crop_and_resize_from_proposals(images: torch.Tensor, boxes_xyxy: torch.Tensor, output_size: int) -> torch.Tensor:
    batch_size, proposal_count, _ = boxes_xyxy.shape
    flattened_boxes = boxes_xyxy.reshape(batch_size * proposal_count, 4)
    repeated_images = (
        images.unsqueeze(1)
        .expand(-1, proposal_count, -1, -1, -1)
        .reshape(batch_size * proposal_count, images.shape[1], images.shape[2], images.shape[3])
    )
    return crop_and_resize_from_boxes(images=repeated_images, boxes_xyxy=flattened_boxes, output_size=output_size)
