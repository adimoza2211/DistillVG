from __future__ import annotations

import torch

from src.models.student.proposals import _normalize_xyxy


def test_normalize_xyxy_scales_to_unit_range() -> None:
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 100.0, 200.0]])
    normalized = _normalize_xyxy(boxes, image_width=100, image_height=200)
    assert normalized.shape == (2, 4)
    assert torch.all(normalized >= 0.0)
    assert torch.all(normalized <= 1.0)
    assert torch.allclose(normalized[0], torch.tensor([0.1, 0.1, 0.5, 0.4]))
