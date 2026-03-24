from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.student.model import StudentModel
from src.training.distill import run_phase2_finetune_step
from src.training.trainer import Trainer


def _build_student() -> StudentModel:
    return StudentModel(
        hidden_dim=32,
        vocab_size=128,
        tob_tokens=24,
        proposal_count=20,
        use_yolo26_proposals=False,
        yolo26_model_cfg="",
        yolo26_weights_path="",
        yolo26_conf_threshold=0.001,
        yolo26_iou_threshold=0.95,
    )


def test_run_phase2_finetune_step_returns_finite_losses() -> None:
    model = _build_student()
    batch = {
        "images": torch.randn(2, 3, 64, 64),
        "token_ids": torch.randint(0, 128, (2, 12)),
        "attention_mask": torch.ones(2, 12, dtype=torch.long),
        "target_box": torch.rand(2, 4),
        "phrases": ["person", "dog"],
        "augmented_phrases": [("person", "person wearing hat"), ("dog", "small dog")],
    }
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    losses = run_phase2_finetune_step(
        model=model,
        batch=batch,
        scaler=scaler,
        device=torch.device("cpu"),
        use_amp=False,
        amp_dtype=torch.float16,
        grad_accum_steps=1,
        lambda1=0.1,
        lambda3=5.0,
        vocab_size=128,
        epoch_index=0,
        total_epochs=2,
        stal_enabled=True,
        stal_area_threshold=0.05,
        stal_max_boost=1.5,
        progloss_enabled=True,
        progloss_box_start=0.9,
        progloss_box_end=1.1,
        progloss_power=1.0,
    )
    assert torch.isfinite(losses.total)


def test_run_phase2_finetune_step_uses_augmented_consistency_signal() -> None:
    torch.manual_seed(7)
    model = _build_student()
    batch = {
        "images": torch.randn(2, 3, 64, 64),
        "token_ids": torch.randint(0, 128, (2, 12)),
        "attention_mask": torch.ones(2, 12, dtype=torch.long),
        "target_box": torch.rand(2, 4),
        "phrases": ["person in red", "dog near tree"],
        "augmented_phrases": [
            ("person in red", "person wearing red jacket"),
            ("dog near tree", "small dog beside a tree"),
        ],
    }
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    losses_with_aug = run_phase2_finetune_step(
        model=model,
        batch=batch,
        scaler=scaler,
        device=torch.device("cpu"),
        use_amp=False,
        amp_dtype=torch.float16,
        grad_accum_steps=1,
        lambda1=0.1,
        lambda3=5.0,
        vocab_size=128,
        epoch_index=0,
        total_epochs=2,
        stal_enabled=True,
        stal_area_threshold=0.05,
        stal_max_boost=1.5,
        progloss_enabled=True,
        progloss_box_start=0.9,
        progloss_box_end=1.1,
        progloss_power=1.0,
    )

    batch_no_aug = dict(batch)
    batch_no_aug["augmented_phrases"] = None
    model.zero_grad(set_to_none=True)
    losses_without_aug = run_phase2_finetune_step(
        model=model,
        batch=batch_no_aug,
        scaler=scaler,
        device=torch.device("cpu"),
        use_amp=False,
        amp_dtype=torch.float16,
        grad_accum_steps=1,
        lambda1=0.1,
        lambda3=5.0,
        vocab_size=128,
        epoch_index=0,
        total_epochs=2,
        stal_enabled=True,
        stal_area_threshold=0.05,
        stal_max_boost=1.5,
        progloss_enabled=True,
        progloss_box_start=0.9,
        progloss_box_end=1.1,
        progloss_power=1.0,
    )

    assert losses_with_aug.total > losses_without_aug.total


def test_trainer_rejects_unknown_mode() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "experiment": "x",
            "training": {"device": "cpu"},
            "train": {"mode": "bad_mode"},
        }
    )
    trainer = Trainer(cfg)
    try:
        trainer._train_mode()
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unsupported mode")
