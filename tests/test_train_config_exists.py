from __future__ import annotations

from pathlib import Path


def test_train_config_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "train.yaml"
    assert config_path.exists(), "configs/train.yaml must exist for scripts/train.py"
