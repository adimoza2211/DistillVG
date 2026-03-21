from __future__ import annotations

from pathlib import Path

from src.data.prepare import default_requirements, validate_manifest_paths_local


def test_manifest_policy_rejects_external_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "refcoco": {"source_path": "/outside/refcoco"},
        "refcoco+": {"source_path": str((raw_dir / "refcoco+").resolve())},
        "refcocog": {"source_path": str((raw_dir / "refcocog").resolve())},
        "referitgame": {"source_path": str((raw_dir / "referitgame").resolve())},
    }

    violations = validate_manifest_paths_local(raw_dir=raw_dir, requirements=default_requirements(), manifest=manifest)
    assert "refcoco" in violations


def test_manifest_policy_accepts_repo_local_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        requirement.name: {"source_path": str((raw_dir / requirement.name).resolve())}
        for requirement in default_requirements()
    }

    violations = validate_manifest_paths_local(raw_dir=raw_dir, requirements=default_requirements(), manifest=manifest)
    assert violations == {}
