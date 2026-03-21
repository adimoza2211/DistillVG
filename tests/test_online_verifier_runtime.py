from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.verifier.prompting import build_verifier_queries
from src.models.verifier.runtime import build_online_verifier
from src.training.verifier_crops import crop_and_resize_from_boxes, normalize_predicted_boxes


def test_normalize_predicted_boxes_enforces_xyxy_order() -> None:
    pred = torch.tensor([[2.0, -1.0, -2.0, 1.0]], dtype=torch.float32)
    boxes = normalize_predicted_boxes(pred)
    assert boxes.shape == (1, 4)
    assert torch.all(boxes[:, 0] <= boxes[:, 2])
    assert torch.all(boxes[:, 1] <= boxes[:, 3])


def test_crop_and_resize_from_boxes_returns_fixed_size() -> None:
    images = torch.randn(2, 3, 32, 32)
    boxes = torch.tensor([[0.1, 0.2, 0.7, 0.9], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    crops = crop_and_resize_from_boxes(images=images, boxes_xyxy=boxes, output_size=24)
    assert crops.shape == (2, 3, 24, 24)


def test_build_online_verifier_mock_is_frozen_and_scores() -> None:
    cfg = OmegaConf.create(
        {
            "backend": "mock",
            "model_id": "OpenGVLab/InternVL3_5-8B",
            "mock_hidden_dim": 16,
            "crop_size": 32,
        }
    )
    verifier = build_online_verifier(cfg=cfg, device=torch.device("cpu"))
    crops = torch.randn(3, 3, 32, 32)
    queries = [
        "Does this image crop match the expression: 'red cup'? Answer only True or False.",
        "Does this image crop match the expression: 'dog next to sofa'? Answer only True or False.",
        "Does this image crop match the expression: 'man with hat'? Answer only True or False.",
    ]
    logits = verifier(crops=crops, queries=queries)

    assert verifier.training is False
    assert all(not parameter.requires_grad for parameter in verifier.parameters())
    assert logits.shape == (3,)


def test_build_verifier_queries_prefers_augmented_phrase() -> None:
    queries = build_verifier_queries(
        base_phrases=["person"],
        augmented_phrase_sets=[("person", "person holding racket", "person near net")],
        use_augmented=True,
        selection_strategy="longest",
    )
    assert len(queries) == 1
    assert "person holding racket" in queries[0]
