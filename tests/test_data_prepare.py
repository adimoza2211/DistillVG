from __future__ import annotations

from pathlib import Path

from src.data.prepare import default_requirements, validate_raw_data


def test_validate_raw_data_reports_missing_entries(tmp_path: Path) -> None:
    requirements = default_requirements()
    missing = validate_raw_data(raw_dir=tmp_path, requirements=requirements)
    assert "refcoco" in missing
    assert "annotations/train.pth" in missing["refcoco"]


def test_validate_raw_data_passes_when_entries_exist(tmp_path: Path) -> None:
    requirements = default_requirements()
    for requirement in requirements:
        dataset_dir = tmp_path / requirement.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for entry in requirement.required_entries:
            target = dataset_dir / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.endswith(".pth"):
                target.write_bytes(b"stub")
            else:
                target.mkdir(parents=True, exist_ok=True)

    missing = validate_raw_data(raw_dir=tmp_path, requirements=requirements)
    assert missing == {}
