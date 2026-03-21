from __future__ import annotations

from pathlib import Path

import torch

from src.data.phrase_bank import _extract_phrase_record, load_phrase_bank_from_augmented_pth


def test_extract_phrase_record_prefers_augmented_phrase_slot() -> None:
    item = (
        "COCO_train2014_000000000009.jpg",
        "1038967.pth",
        [1.08, 187.69, 611.59, 285.84],
        [
            "a yellow rectangle bowl with broccoli",
            "a yellow bowl with broccoli",
        ],
        [("r1", ["rectangle"])],
    )
    record = _extract_phrase_record(item)
    assert record is not None
    assert record.base_phrase == "a yellow rectangle bowl with broccoli"
    assert len(record.augmented_phrases) == 2


def test_load_phrase_bank_from_augmented_pth_reads_records(tmp_path: Path) -> None:
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

    records = load_phrase_bank_from_augmented_pth(str(tmp_path / "*_aug.pth"))
    assert len(records) == 1
    assert records[0].base_phrase == "phrase one"
    assert records[0].augmented_phrases == ("phrase one", "phrase two")
