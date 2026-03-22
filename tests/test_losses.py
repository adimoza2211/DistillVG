from __future__ import annotations

import torch

from src.models.losses import compute_distillation_losses


def test_compute_distillation_losses_returns_finite_values() -> None:
    batch_size = 2
    pred_box = torch.randn(batch_size, 4)
    target_box = torch.rand(batch_size, 4)
    verifier_logit = torch.randn(batch_size, 16)
    verifier_target = torch.rand(batch_size, 16)

    losses = compute_distillation_losses(
        pred_box=pred_box,
        target_box=target_box,
        verifier_logit=verifier_logit,
        verifier_target=verifier_target,
    )

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.giou)
    assert torch.isfinite(losses.ver)
