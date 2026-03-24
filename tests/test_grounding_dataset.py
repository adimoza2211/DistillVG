from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.grounding import AugmentedGroundingDataset, GroundingRecord, _encode_phrase_to_ids, load_grounding_records


def test_load_grounding_records_from_augmented_payload(tmp_path: Path) -> None:
    payload = [
        (
            "COCO_train2014_000000000009.jpg",
            "1038967.pth",
            [1.08, 187.69, 611.59, 285.84],
            ["phrase one", "phrase two"],
            [("r1", ["rectangle"])],
        )
    ]
    path = tmp_path / "sample_aug.pth"
    torch.save(payload, path)

    records = load_grounding_records(str(tmp_path / "*_aug.pth"))
    assert len(records) == 1
    assert records[0].image_name == "COCO_train2014_000000000009.jpg"
    assert records[0].phrases == ("phrase one", "phrase two")
    assert records[0].source == tmp_path.name


def test_augmented_grounding_dataset_emits_phase1_sample(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_path = image_root / "COCO_train2014_000000000009.jpg"
    Image.new("RGB", (100, 80), color=(128, 64, 32)).save(image_path)

    records = [
        GroundingRecord(
            image_name="COCO_train2014_000000000009.jpg",
            box_xywh=(10.0, 5.0, 30.0, 20.0),
            phrases=("small object", "tiny object"),
            source="synthetic",
        )
    ]
    dataset = AugmentedGroundingDataset(
        records=records,
        image_root=str(image_root),
        resize_long_edge=64,
        padded_square_size=64,
        max_query_length=12,
        text_vocab_size=256,
    )

    sample = dataset[0]
    assert sample.image.shape == (3, 64, 64)
    assert sample.token_ids.shape == (12,)
    assert sample.attention_mask.shape == (12,)
    assert sample.target_box.shape == (4,)
    assert sample.phrase == "small object"
    assert sample.augmented_phrases == ("small object", "tiny object")
    assert torch.all(sample.target_box >= 0.0)
    assert torch.all(sample.target_box <= 1.0)


def test_phrase_encoding_is_deterministic_and_nonzero_for_valid_tokens() -> None:
    phrase = "the small red object"
    ids_a = _encode_phrase_to_ids(phrase=phrase, seq_len=8, vocab_size=256)
    ids_b = _encode_phrase_to_ids(phrase=phrase, seq_len=8, vocab_size=256)

    assert torch.equal(ids_a, ids_b)
    assert torch.all(ids_a[:4] > 0)


def test_resize_long_edge_larger_than_canvas_still_stays_in_bounds(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_path = image_root / "COCO_train2014_000000000010.jpg"
    Image.new("RGB", (320, 160), color=(100, 100, 100)).save(image_path)

    records = [
        GroundingRecord(
            image_name="COCO_train2014_000000000010.jpg",
            box_xywh=(20.0, 10.0, 150.0, 80.0),
            phrases=("object",),
            source="synthetic",
        )
    ]
    dataset = AugmentedGroundingDataset(
        records=records,
        image_root=str(image_root),
        resize_long_edge=512,
        padded_square_size=64,
        max_query_length=12,
        text_vocab_size=256,
    )

    sample = dataset[0]
    assert sample.image.shape == (3, 64, 64)
    assert torch.all(sample.target_box >= 0.0)
    assert torch.all(sample.target_box <= 1.0)
