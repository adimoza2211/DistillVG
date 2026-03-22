from __future__ import annotations

import torch


def scalar_to_soft_label(true_logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(true_logits)
    excellent = torch.tensor([0.7, 0.2, 0.08, 0.02], device=true_logits.device, dtype=true_logits.dtype)
    good = torch.tensor([0.1, 0.6, 0.25, 0.05], device=true_logits.device, dtype=true_logits.dtype)
    partial = torch.tensor([0.02, 0.15, 0.6, 0.23], device=true_logits.device, dtype=true_logits.dtype)
    no = torch.tensor([0.01, 0.04, 0.15, 0.80], device=true_logits.device, dtype=true_logits.dtype)

    labels = no.unsqueeze(0).expand(true_logits.shape[0], -1).clone()
    labels = torch.where((probabilities > 0.35).unsqueeze(-1), partial.unsqueeze(0), labels)
    labels = torch.where((probabilities > 0.6).unsqueeze(-1), good.unsqueeze(0), labels)
    labels = torch.where((probabilities > 0.85).unsqueeze(-1), excellent.unsqueeze(0), labels)
    return labels


def draw_highlighted_proposals(
    images: torch.Tensor,
    proposal_boxes: torch.Tensor,
    top_indices: torch.Tensor,
    thickness: int = 3,
) -> torch.Tensor:
    batch_size, _, image_height, image_width = images.shape
    top_k = top_indices.shape[1]
    highlighted: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        for rank in range(top_k):
            proposal_idx = int(top_indices[batch_idx, rank].item())
            x1, y1, x2, y2 = proposal_boxes[batch_idx, proposal_idx]
            left = max(0, min(int(torch.floor(x1 * image_width).item()), image_width - 1))
            right = max(left + 1, min(int(torch.ceil(x2 * image_width).item()), image_width))
            top = max(0, min(int(torch.floor(y1 * image_height).item()), image_height - 1))
            bottom = max(top + 1, min(int(torch.ceil(y2 * image_height).item()), image_height))

            canvas = images[batch_idx].clone()
            t = max(1, thickness)
            canvas[0, top : min(top + t, image_height), left:right] = 1.0
            canvas[1:, top : min(top + t, image_height), left:right] = 0.0

            canvas[0, max(bottom - t, 0) : bottom, left:right] = 1.0
            canvas[1:, max(bottom - t, 0) : bottom, left:right] = 0.0

            canvas[0, top:bottom, left : min(left + t, image_width)] = 1.0
            canvas[1:, top:bottom, left : min(left + t, image_width)] = 0.0

            canvas[0, top:bottom, max(right - t, 0) : right] = 1.0
            canvas[1:, top:bottom, max(right - t, 0) : right] = 0.0

            highlighted.append(canvas.unsqueeze(0))

    return torch.cat(highlighted, dim=0)


def build_verifier_targets(
    stage1_scores: torch.Tensor,
    stage2_soft_labels: torch.Tensor,
    top_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, proposal_count = stage1_scores.shape
    top_k = top_indices.shape[1]

    targets = scalar_to_soft_label(stage1_scores.reshape(-1)).reshape(batch_size, proposal_count, 4)
    mask = (stage1_scores.abs() > 0.5)

    scatter_index = top_indices.unsqueeze(-1).expand(-1, -1, 4)
    targets = targets.scatter(dim=1, index=scatter_index, src=stage2_soft_labels)
    mask = mask.scatter(
        dim=1,
        index=top_indices,
        src=torch.ones(batch_size, top_k, device=mask.device, dtype=mask.dtype),
    )
    return targets, mask
