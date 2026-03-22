from __future__ import annotations

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
