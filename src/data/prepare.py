from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRequirement:
    name: str
    required_entries: tuple[str, ...]


def default_requirements() -> tuple[DatasetRequirement, ...]:
    return (
        DatasetRequirement(
            name="refcoco",
            required_entries=(
                "images",
                "annotations/corpus.pth",
                "annotations/train.pth",
                "annotations/val.pth",
                "annotations/testA.pth",
                "annotations/testB.pth",
            ),
        ),
        DatasetRequirement(
            name="refcoco+",
            required_entries=(
                "images",
                "annotations/corpus.pth",
                "annotations/train.pth",
                "annotations/val.pth",
                "annotations/testA.pth",
                "annotations/testB.pth",
            ),
        ),
        DatasetRequirement(
            name="refcocog",
            required_entries=(
                "images",
                "annotations/corpus.pth",
                "annotations/train.pth",
                "annotations/val.pth",
                "annotations/test.pth",
            ),
        ),
        DatasetRequirement(
            name="referitgame",
            required_entries=(
                "images",
                "annotations/corpus.pth",
                "annotations/train.pth",
                "annotations/val.pth",
                "annotations/test.pth",
            ),
        ),
    )


def write_manifest_template(manifest_path: Path, requirements: tuple[DatasetRequirement, ...]) -> None:
    raw_dir = manifest_path.parent.resolve()
    template = {
        requirement.name: {
            "source_path": str((raw_dir / requirement.name).resolve()),
            "notes": "Must be a path inside this repo under data/raw/<dataset_name>.",
        }
        for requirement in requirements
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(template, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    return payload


def validate_manifest_paths_local(
    raw_dir: Path,
    requirements: tuple[DatasetRequirement, ...],
    manifest: dict[str, dict[str, str]],
) -> dict[str, str]:
    violations: dict[str, str] = {}
    for requirement in requirements:
        entry = manifest.get(requirement.name, {})
        source_path_str = str(entry.get("source_path", "")).strip()
        expected_path = (raw_dir / requirement.name).resolve()

        if not source_path_str:
            violations[requirement.name] = f"source_path missing; expected {expected_path}"
            continue

        source_path = Path(source_path_str).expanduser().resolve()
        if source_path != expected_path:
            violations[requirement.name] = f"source_path must be {expected_path}"

    return violations


def validate_raw_data(
    raw_dir: Path,
    requirements: tuple[DatasetRequirement, ...],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for requirement in requirements:
        dataset_dir = raw_dir / requirement.name
        dataset_missing: list[str] = []

        if not dataset_dir.exists():
            dataset_missing.extend(requirement.required_entries)
        else:
            for item in requirement.required_entries:
                if not (dataset_dir / item).exists():
                    dataset_missing.append(item)

        if dataset_missing:
            missing[requirement.name] = dataset_missing

    return missing
