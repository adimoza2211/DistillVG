from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.student.model import StudentModel
from src.training.distill import DistillStepOutput
import src.training.trainer as trainer_module
from src.training.distill import run_phase2_finetune_step
from src.training.trainer import Trainer


def _build_student() -> StudentModel:
    return StudentModel(
        hidden_dim=32,
        fusion_layers=2,
        attention_heads=4,
        vocab_size=128,
        roi_tokens=32,
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


def test_trainer_flushes_residual_grad_accumulation(monkeypatch) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params, lr: float) -> None:
            super().__init__(params, lr=lr)
            self.step_count = 0

        def step(self, closure=None):
            self.step_count += 1
            return super().step(closure)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "experiment": "grad_accum_flush",
            "training": {
                "device": "cpu",
                "batch_size_per_gpu": 1,
                "epochs": 1,
                "text_vocab_size": 128,
            },
            "train": {
                "mode": "phase2_strict",
                "amp": False,
                "amp_dtype": "fp16",
                "grad_accum_steps": 2,
                "grad_clip_norm": 1.0,
                "lambda_l1": 1.0,
                "lambda_giou": 1.0,
                "lambda_ver": 1.0,
                "stal": {"enabled": False, "area_threshold": 0.05, "max_boost": 1.0},
                "progloss": {"enabled": False, "box_start_scale": 1.0, "box_end_scale": 1.0, "power": 1.0},
            },
            "lambda1": 0.1,
            "lambda2": 1.0,
            "lambda3": 1.0,
        }
    )
    trainer = Trainer(cfg)
    model = TinyModel()
    optimizer_holder: dict[str, CountingSGD] = {}

    monkeypatch.setattr(Trainer, "_build_model", lambda self: model.to(self.device))
    monkeypatch.setattr(Trainer, "_build_dataloader", lambda self: [{}, {}, {}])
    monkeypatch.setattr(Trainer, "_apply_epoch_trainability", lambda self, _model, _epoch: None)

    def _build_optimizer_override(self, _model):
        optimizer = CountingSGD(model.parameters(), lr=1e-2)
        optimizer_holder["optimizer"] = optimizer
        return optimizer

    monkeypatch.setattr(Trainer, "_build_optimizer", _build_optimizer_override)

    def _fake_phase2_step(**kwargs):
        fake_loss = (model.weight ** 2).sum()
        kwargs["scaler"].scale(fake_loss / kwargs["grad_accum_steps"]).backward()
        detached = fake_loss.detach()
        return DistillStepOutput(total=detached, l1=detached, giou=detached * 0.0, ver=detached * 0.0)

    monkeypatch.setattr(trainer_module, "run_phase2_finetune_step", _fake_phase2_step)

    trainer.train()

    optimizer = optimizer_holder["optimizer"]
    assert optimizer.step_count == 2
