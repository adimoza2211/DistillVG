from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.verifier.prompting import build_verifier_queries
from src.models.verifier.runtime import _collect_single_token_ids, _normalize_crop_for_processor, build_online_verifier
from src.models.verifier.two_stage import build_verifier_targets, scalar_to_soft_label
from src.training.verifier_crops import (
    crop_and_resize_from_boxes,
    crop_and_resize_from_proposals,
    normalize_predicted_boxes,
    normalize_xyxy_boxes,
)


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


def test_crop_and_resize_from_proposals_returns_flattened_batch() -> None:
    images = torch.randn(2, 3, 32, 32)
    boxes = torch.tensor(
        [
            [[0.1, 0.2, 0.7, 0.9], [0.0, 0.0, 1.0, 1.0]],
            [[0.2, 0.1, 0.8, 0.6], [0.3, 0.3, 0.6, 0.9]],
        ],
        dtype=torch.float32,
    )
    crops = crop_and_resize_from_proposals(images=images, boxes_xyxy=boxes, output_size=20)
    assert crops.shape == (4, 3, 20, 20)


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
    logits_4 = verifier.score_4class(images=crops, queries=queries)

    assert verifier.training is False
    assert all(not parameter.requires_grad for parameter in verifier.parameters())
    assert logits.shape == (3,)
    assert logits_4.shape == (3, 4)


def test_build_verifier_queries_prefers_augmented_phrase() -> None:
    queries = build_verifier_queries(
        base_phrases=["person"],
        augmented_phrase_sets=[("person", "person holding racket", "person near net")],
        use_augmented=True,
        selection_strategy="longest",
    )
    assert len(queries) == 1
    assert "person holding racket" in queries[0]


def test_normalize_crop_for_processor_rescales_to_unit_range() -> None:
    crop = torch.tensor(
        [[[-2.0, 0.5], [1.5, 4.0]], [[0.0, 0.0], [0.0, 0.0]], [[-1.0, 2.0], [2.0, 3.0]]],
        dtype=torch.float32,
    )
    normalized = _normalize_crop_for_processor(crop)
    assert normalized.min().item() >= 0.0
    assert normalized.max().item() <= 1.0


def test_collect_single_token_ids_filters_multi_token_candidates() -> None:
    class DummyTokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            mapping = {
                "True": [10],
                " true": [11],
                "Yes": [12],
                " false": [20],
                "No": [21],
                "two tokens": [4, 5],
            }
            return mapping.get(text, [99, 100])

    ids = _collect_single_token_ids(DummyTokenizer(), ["two tokens", "True", " true", "Yes"])
    assert ids == [10, 11, 12]


def test_normalize_xyxy_boxes_clamps_and_orders() -> None:
    boxes = torch.tensor([[[0.8, 0.9, 0.1, 0.2]]], dtype=torch.float32)
    normalized = normalize_xyxy_boxes(boxes)
    assert normalized.shape == (1, 1, 4)
    assert normalized[0, 0, 0] <= normalized[0, 0, 2]
    assert normalized[0, 0, 1] <= normalized[0, 0, 3]


def test_two_stage_target_builder_shapes() -> None:
    stage1 = torch.tensor([[0.2, 1.2, -1.5, 0.9]], dtype=torch.float32)
    top_idx = torch.tensor([[1, 3]], dtype=torch.long)
    stage2 = torch.softmax(torch.randn(1, 2, 4), dim=-1)
    targets, mask = build_verifier_targets(stage1_scores=stage1, stage2_soft_labels=stage2, top_indices=top_idx)
    assert targets.shape == (1, 4, 4)
    assert mask.shape == (1, 4)
    assert torch.allclose(targets[0, 1], stage2[0, 0], atol=1e-6)
    assert torch.allclose(targets[0, 3], stage2[0, 1], atol=1e-6)


def test_scalar_to_soft_label_shape() -> None:
    logits = torch.tensor([2.0, 0.0, -2.0], dtype=torch.float32)
    labels = scalar_to_soft_label(logits)
    assert labels.shape == (3, 4)
