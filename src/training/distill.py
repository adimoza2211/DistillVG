from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from src.losses.box_losses import box_loss
from src.losses.combined import DualBranchGroundingLoss
from src.losses.dwbd import compute_dwbd_alpha
from src.losses.stal_progloss import compute_progloss_scale, compute_stal_weight


@dataclass
class DistillStepOutput:
    total: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    ver: torch.Tensor


# Tokenize function type: (phrase: str) -> tuple[token_ids, attention_mask]
TokenizeFn = Callable[[str], tuple[torch.Tensor, torch.Tensor]]


def run_distillation_step(
    model: torch.nn.Module,
    online_verifier,  # BaseOnlineVerifier | None
    batch: dict[str, torch.Tensor | list[str] | list[tuple[str, ...]]],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    epoch_index: int,
    total_epochs: int,
    lambda_box: float = 5.0,
    lambda_dwbd: float = 2.0,
    lambda_align: float = 1.0,
    lambda_cst: float = 0.1,
    dwbd_alpha_max: float = 0.7,
    dwbd_alpha_min: float = 0.1,
    dwbd_gamma: float = 2.0,
    stal_enabled: bool = True,
    stal_area_threshold: float = 0.05,
    stal_max_boost: float = 1.5,
    progloss_enabled: bool = True,
    progloss_box_start: float = 0.9,
    progloss_box_end: float = 1.1,
    progloss_power: float = 1.0,
    # Legacy params accepted but ignored for backward compat.
    **_kwargs,
) -> DistillStepOutput:
    """Run one distillation training step for the dual-branch model.

    Computes:
    1. Primary box loss (L1+GIoU) on decoder branch vs GT
    2. DWBD distillation loss (decoder → MLP branch)
    3. Alignment loss (optional, from teacher verifier)
    4. Consistency regularization
    5. STAL weighting for small objects
    6. ProgLoss scaling
    """
    images = batch["images"].to(device, non_blocking=True)
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_box = batch["target_box"].to(device, non_blocking=True)

    # Precomputed teacher alignment scores (optional).
    precomputed_alignment = batch.get("precomputed_alignment")
    teacher_alignment = None
    if precomputed_alignment is not None:
        teacher_alignment = precomputed_alignment.to(device, non_blocking=True)

    # DWBD alpha scheduling.
    alpha = compute_dwbd_alpha(
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        alpha_max=dwbd_alpha_max,
        alpha_min=dwbd_alpha_min,
        gamma=dwbd_gamma,
    )

    # Progressive loss scaling.
    box_scale = compute_progloss_scale(
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        start_scale=progloss_box_start,
        end_scale=progloss_box_end,
        power=progloss_power,
        enabled=progloss_enabled,
    )

    loss_fn = DualBranchGroundingLoss(
        lambda_box=lambda_box,
        lambda_dwbd=lambda_dwbd,
        lambda_align=lambda_align,
        lambda_cst=lambda_cst,
    )

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(
            images=images,
            token_ids=token_ids,
            attention_mask=attention_mask,
        )

        # STAL weighting for small objects.
        stal_weight = compute_stal_weight(
            target_box,
            area_threshold=stal_area_threshold,
            max_boost=stal_max_boost,
            enabled=stal_enabled,
        )

        # Compute per-sample box loss for STAL weighting.
        per_sample_box = box_loss(outputs["decoder_bbox"], target_box, reduction="none")
        weighted_decoder_box = (per_sample_box * stal_weight).sum() / stal_weight.sum().clamp_min(1.0)

        # Compute dual-branch loss.
        loss_out = loss_fn(
            decoder_bbox=outputs["decoder_bbox"],
            mlp_bbox=outputs["mlp_bbox"],
            gt_bbox=target_box,
            alpha=alpha,
            alignment_score=outputs.get("alignment_score"),
            teacher_alignment=teacher_alignment,
            fused_tokens=outputs.get("fused_tokens"),
            box_scale=box_scale,
        )

        # Replace the box component with STAL-weighted version.
        total = (
            loss_fn.lambda_box * box_scale * weighted_decoder_box
            + loss_fn.lambda_dwbd * loss_out.dwbd
            + loss_fn.lambda_align * loss_out.alignment
            + loss_fn.lambda_cst * loss_out.consistency
        )

        scaled_total = total / grad_accum_steps

    scaler.scale(scaled_total).backward()

    return DistillStepOutput(
        total=total.detach(),
        l1=weighted_decoder_box.detach(),
        giou=loss_out.dwbd.detach(),  # Log DWBD in the GIoU slot for monitoring
        ver=loss_out.alignment.detach(),
    )


def run_phase2_finetune_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str] | list[tuple[str, ...]]],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    lambda_box: float = 5.0,
    lambda_dwbd: float = 2.0,
    lambda_cst: float = 0.1,
    tokenize_fn: TokenizeFn | None = None,
    epoch_index: int = 0,
    total_epochs: int = 1,
    stal_enabled: bool = True,
    stal_area_threshold: float = 0.05,
    stal_max_boost: float = 1.5,
    progloss_enabled: bool = True,
    progloss_box_start: float = 0.9,
    progloss_box_end: float = 1.1,
    progloss_power: float = 1.0,
    # Legacy params accepted but ignored.
    **_kwargs,
) -> DistillStepOutput:
    """Phase 2: GT-only fine-tuning with fixed-alpha DWBD between branches."""
    images = batch["images"].to(device, non_blocking=True)
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_box = batch["target_box"].to(device, non_blocking=True)

    box_scale = compute_progloss_scale(
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        start_scale=progloss_box_start,
        end_scale=progloss_box_end,
        power=progloss_power,
        enabled=progloss_enabled,
    )

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(images=images, token_ids=token_ids, attention_mask=attention_mask)

        # Box loss with STAL weighting.
        stal_weight = compute_stal_weight(
            target_box,
            area_threshold=stal_area_threshold,
            max_boost=stal_max_boost,
            enabled=stal_enabled,
        )
        per_sample_box = box_loss(outputs["decoder_bbox"], target_box, reduction="none")
        l_box = (per_sample_box * stal_weight).sum() / stal_weight.sum().clamp_min(1.0)

        # Fixed-alpha DWBD for branch consistency.
        from src.losses.dwbd import DWBDBoxLoss
        dwbd_fn = DWBDBoxLoss()
        l_dwbd = dwbd_fn(outputs["mlp_bbox"], outputs["decoder_bbox"], target_box, alpha=0.3)

        total = lambda_box * box_scale * l_box + lambda_dwbd * l_dwbd
        scaled_total = total / grad_accum_steps

    scaler.scale(scaled_total).backward()

    return DistillStepOutput(
        total=total.detach(),
        l1=l_box.detach(),
        giou=l_dwbd.detach(),
        ver=torch.zeros_like(l_box.detach()),
    )
