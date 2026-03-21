from __future__ import annotations

from pathlib import Path


def test_core_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src" / "utils",
        root / "src" / "models",
        root / "src" / "data",
        root / "src" / "training",
        root / "scripts",
        root / "configs",
    ]
    for path in required:
        assert path.exists()
