from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_train_config_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "train.yaml"
    assert config_path.exists(), "configs/train.yaml must exist for scripts/train.py"


def test_train_config_has_required_data_and_resume_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "train.yaml"
    cfg = OmegaConf.load(config_path)

    assert str(cfg.data.train_records_original_glob).strip() != ""
    assert str(cfg.data.train_records_augmented_glob).strip() != ""
    assert str(cfg.data.val_records_glob).strip() != ""

    assert str(cfg.train.checkpoint_dir).strip() != ""
    assert isinstance(bool(cfg.train.resume), bool)
    assert hasattr(cfg.train, "resume_checkpoint_path")
