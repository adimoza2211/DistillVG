from __future__ import annotations

from typing import Any

import torch

from src.models.losses import LossOutput, compute_distillation_losses


def run_distillation_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accum_steps: int,
) -> LossOutput:
    images = batch["images"].to(device, non_blocking=True)
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_box = batch["target_box"].to(device, non_blocking=True)
    verifier_target = batch["verifier_target"].to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        outputs = model(images=images, token_ids=token_ids, attention_mask=attention_mask)
        losses = compute_distillation_losses(
            pred_box=outputs["bbox"],
            target_box=target_box,
            verifier_logit=outputs["verifier_logit"],
            verifier_target=verifier_target,
        )
        scaled_total = losses.total / grad_accum_steps

    scaler.scale(scaled_total).backward()
    return losses
