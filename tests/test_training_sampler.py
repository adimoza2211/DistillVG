from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.grounding import GroundingRecord
from src.training.trainer import Trainer


def test_compute_sampling_weights_inverse_to_source_frequency() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "experiment": "sampler_test",
            "training": {"device": "cpu"},
            "train": {"mode": "phase1"},
        }
    )
    trainer = Trainer(cfg)

    records = [
        GroundingRecord(
            image_name="a.jpg",
            box_xywh=(0.0, 0.0, 1.0, 1.0),
            phrases=("x",),
            source="set_a",
        ),
        GroundingRecord(
            image_name="b.jpg",
            box_xywh=(0.0, 0.0, 1.0, 1.0),
            phrases=("x",),
            source="set_a",
        ),
        GroundingRecord(
            image_name="c.jpg",
            box_xywh=(0.0, 0.0, 1.0, 1.0),
            phrases=("x",),
            source="set_b",
        ),
    ]

    weights = trainer._compute_sampling_weights(records)

    assert float(weights[0]) == float(weights[1])
    assert float(weights[2]) > float(weights[0])


def test_load_records_from_patterns_combines_and_ignores_empty(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "experiment": "sampler_test",
            "training": {"device": "cpu"},
            "train": {"mode": "phase1"},
        }
    )
    trainer = Trainer(cfg)

    records_a = [GroundingRecord("a.jpg", (0.0, 0.0, 1.0, 1.0), ("a",), "unc")]
    records_b = [GroundingRecord("b.jpg", (0.0, 0.0, 1.0, 1.0), ("b",), "gref")]

    def _fake_loader(pattern: str) -> list[GroundingRecord]:
        if pattern == "pattern_a":
            return records_a
        if pattern == "pattern_b":
            return records_b
        return []

    monkeypatch.setattr("src.training.trainer.load_grounding_records", _fake_loader)

    merged = trainer._load_records_from_patterns(["pattern_a", "", "pattern_b"])
    assert len(merged) == 2
    assert merged[0].image_name == "a.jpg"
    assert merged[1].image_name == "b.jpg"


def test_checkpoint_save_and_resume_roundtrip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "latest_pretrain.pt"
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "experiment": "resume_test",
            "training": {"device": "cpu"},
            "train": {
                "mode": "phase1",
                "resume": True,
                "resume_checkpoint_path": str(checkpoint_path),
                "checkpoint_dir": str(tmp_path),
            },
        }
    )
    trainer = Trainer(cfg)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    trainer._save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        epoch_index=2,
        global_step=17,
        best_val_acc=0.6,
        latest_path=checkpoint_path,
        best_path=tmp_path / "best_pretrain.pt",
        is_best=True,
    )

    resumed_model = torch.nn.Linear(4, 4)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scaler = torch.amp.GradScaler(enabled=False)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1)

    start_epoch, global_step, best_val_acc = trainer._try_resume(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        scheduler=resumed_scheduler,
    )

    assert start_epoch == 3
    assert global_step == 17
    assert best_val_acc == 0.6
