from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torchvision.ops import box_iou

from src.losses.box_losses import box_loss
from src.losses.consistency import consistency_loss
from src.losses.dwbd import DWBDLoss, compute_dwbd_alpha
from src.losses.stal_progloss import compute_progloss_scale, compute_stal_weight
from src.losses.verifier_kl import verifier_kl_loss
from src.models.verifier.prompting import build_verifier_queries
from src.models.verifier.prompting import build_stage2_verifier_queries
from src.models.verifier.runtime import BaseOnlineVerifier
from src.models.verifier.two_stage import build_verifier_targets, draw_highlighted_proposals
from src.training.verifier_crops import crop_and_resize_from_proposals, normalize_predicted_boxes, normalize_xyxy_boxes


@dataclass
class DistillStepOutput:
    total: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    ver: torch.Tensor


# Tokenize function type: (phrase: str) -> tuple[token_ids, attention_mask]
TokenizeFn = Callable[[str], tuple[torch.Tensor, torch.Tensor]]


def _select_alternate_phrase(base_phrase: str, candidates: tuple[str, ...]) -> str | None:
    normalized_base = base_phrase.strip()
    for candidate in candidates:
        normalized_candidate = candidate.strip()
        if normalized_candidate and normalized_candidate != normalized_base:
            return normalized_candidate
    return None


def _build_augmented_inputs(
    *,
    phrases: list[str],
    augmented_phrase_sets: list[tuple[str, ...]] | None,
    tokenize_fn: TokenizeFn,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if augmented_phrase_sets is None or len(augmented_phrase_sets) != len(phrases):
        return None

    augmented_token_ids: list[torch.Tensor] = []
    augmented_attention_masks: list[torch.Tensor] = []
    has_any_alternate = False
    for sample_idx, base_phrase in enumerate(phrases):
        chosen_phrase = _select_alternate_phrase(base_phrase, augmented_phrase_sets[sample_idx])
        if chosen_phrase is None:
            chosen_phrase = base_phrase
        else:
            has_any_alternate = True

        token_ids, attention_mask = tokenize_fn(chosen_phrase)
        augmented_token_ids.append(token_ids.unsqueeze(0).to(device))
        augmented_attention_masks.append(attention_mask.unsqueeze(0).to(device))

    if not has_any_alternate:
        return None

    return torch.cat(augmented_token_ids, dim=0), torch.cat(augmented_attention_masks, dim=0)


def run_distillation_step(
    model: torch.nn.Module,
    online_verifier: BaseOnlineVerifier | None,
    batch: dict[str, torch.Tensor | list[str] | list[tuple[str, ...]]],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    verifier_crop_size: int,
    verifier_top_k_proposals: int,
    verifier_stage2_top_k: int,
    use_augmented_verifier_queries: bool,
    verifier_query_selection: str,
    epoch_index: int,
    total_epochs: int,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    dwbd_alpha_max: float,
    dwbd_alpha_min: float,
    dwbd_gamma: float,
    stal_enabled: bool,
    stal_area_threshold: float,
    stal_max_boost: float,
    progloss_enabled: bool,
    progloss_box_start: float,
    progloss_box_end: float,
    progloss_verifier_start: float,
    progloss_verifier_end: float,
    progloss_power: float,
) -> DistillStepOutput:
    images = batch["images"].to(device, non_blocking=True)
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_box = batch["target_box"].to(device, non_blocking=True)
    phrases = list(batch.get("phrases", []))

    # --- Precomputed target support ---
    precomputed_proposals = batch.get("precomputed_proposals")
    precomputed_verifier_targets = batch.get("precomputed_verifier_targets")
    precomputed_verifier_mask = batch.get("precomputed_verifier_mask")

    has_precomputed = (
        precomputed_verifier_targets is not None
        and precomputed_verifier_mask is not None
    )

    if precomputed_proposals is not None:
        precomputed_proposals = precomputed_proposals.to(device, non_blocking=True)
    if has_precomputed:
        precomputed_verifier_targets = precomputed_verifier_targets.to(device, non_blocking=True)
        precomputed_verifier_mask = precomputed_verifier_mask.to(device, non_blocking=True)

    dwbd_loss_fn = DWBDLoss()

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(
            images=images,
            token_ids=token_ids,
            attention_mask=attention_mask,
            precomputed_proposals=precomputed_proposals,
        )
        consistency_maps: list[torch.Tensor] = [outputs["fused_tokens"]]

        if bool(getattr(model, "use_yolo26_proposals", False)):
            proposal_boxes = normalize_xyxy_boxes(outputs["proposal_boxes"].detach())
        else:
            proposal_boxes = normalize_predicted_boxes(outputs["proposal_boxes"].detach())

        proposal_count = proposal_boxes.shape[1]
        stage1_k = max(1, min(verifier_top_k_proposals, proposal_count))
        proposal_boxes_stage1 = proposal_boxes[:, :stage1_k, :]

        if has_precomputed:
            # --- Use precomputed verifier targets ---
            verifier_targets = precomputed_verifier_targets
            verifier_mask = precomputed_verifier_mask
        else:
            # --- Online verifier path ---
            if online_verifier is None:
                raise RuntimeError(
                    "No online verifier and no precomputed targets. "
                    "Either set verifier.backend or provide precomputed_cache_path."
                )

            crops = crop_and_resize_from_proposals(
                images=images.detach(),
                boxes_xyxy=proposal_boxes_stage1,
                output_size=verifier_crop_size,
            )

            with torch.no_grad():
                # FIX: augmented_phrase_sets is not available in the batch
                # (augmented phrases are already expanded into separate samples).
                # Pass None so build_verifier_queries uses base phrases only.
                verifier_queries = build_verifier_queries(
                    base_phrases=phrases,
                    augmented_phrase_sets=None,
                    use_augmented=use_augmented_verifier_queries,
                    selection_strategy=verifier_query_selection,
                )
                repeated_queries = [query for query in verifier_queries for _ in range(stage1_k)]
                stage1_scores = online_verifier(
                    crops=crops,
                    queries=repeated_queries,
                )
                stage1_scores = stage1_scores.reshape(images.shape[0], stage1_k)

                stage2_top_k = max(1, min(verifier_stage2_top_k, stage1_k))
                top_stage2_indices = stage1_scores.topk(stage2_top_k, dim=-1).indices

                # FIX: Draw red rectangles on images before stage2 scoring.
                # The stage2 prompt references "the red rectangle" — without this,
                # the verifier sees raw images and the prompt is nonsensical.
                highlighted_images = draw_highlighted_proposals(
                    images=images,
                    proposal_boxes=proposal_boxes_stage1,
                    top_indices=top_stage2_indices,
                )

                stage2_queries = build_stage2_verifier_queries(verifier_queries, top_k=stage2_top_k)
                stage2_logits = online_verifier.score_4class(
                    images=highlighted_images,
                    queries=stage2_queries,
                )
                stage2_soft = torch.softmax(stage2_logits, dim=-1).reshape(images.shape[0], stage2_top_k, 4)

                verifier_targets, verifier_mask = build_verifier_targets(
                    stage1_scores=stage1_scores,
                    stage2_soft_labels=stage2_soft,
                    top_indices=top_stage2_indices,
                )

        student_logits = outputs["proposal_verifier_logits"][:, :stage1_k, :]
        target_boxes_expanded = target_box.unsqueeze(1).expand(-1, stage1_k, -1)
        ious = box_iou(
            proposal_boxes_stage1.reshape(-1, 4),
            target_boxes_expanded.reshape(-1, 4),
        )
        ious = ious.diagonal().reshape(images.shape[0], stage1_k)

        gt_one_hot = torch.zeros(images.shape[0], stage1_k, 4, device=images.device, dtype=student_logits.dtype)
        gt_one_hot[..., 3] = 1.0
        gt_one_hot = torch.where((ious >= 0.3).unsqueeze(-1), torch.tensor([0.0, 0.0, 1.0, 0.0], device=images.device, dtype=student_logits.dtype), gt_one_hot)
        gt_one_hot = torch.where((ious >= 0.5).unsqueeze(-1), torch.tensor([0.0, 1.0, 0.0, 0.0], device=images.device, dtype=student_logits.dtype), gt_one_hot)
        gt_one_hot = torch.where((ious >= 0.7).unsqueeze(-1), torch.tensor([1.0, 0.0, 0.0, 0.0], device=images.device, dtype=student_logits.dtype), gt_one_hot)

        alpha = compute_dwbd_alpha(
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            alpha_max=dwbd_alpha_max,
            alpha_min=dwbd_alpha_min,
            gamma=dwbd_gamma,
        )

        # --- Compute individual losses directly ---
        l_dwbd = dwbd_loss_fn(
            student_logits,
            verifier_targets.to(student_logits.dtype),
            gt_one_hot,
            alpha,
        )
        l_ver_kl = verifier_kl_loss(
            student_logits,
            verifier_targets.to(student_logits.dtype),
            verifier_mask,
        )
        l_cst = consistency_loss(consistency_maps)

        stal_weight = compute_stal_weight(
            target_box,
            area_threshold=stal_area_threshold,
            max_boost=stal_max_boost,
            enabled=stal_enabled,
        )
        box_scale = compute_progloss_scale(
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            start_scale=progloss_box_start,
            end_scale=progloss_box_end,
            power=progloss_power,
            enabled=progloss_enabled,
        )
        verifier_scale = compute_progloss_scale(
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            start_scale=progloss_verifier_start,
            end_scale=progloss_verifier_end,
            power=progloss_power,
            enabled=progloss_enabled,
        )

        per_sample_box = box_loss(outputs["bbox"], target_box, reduction="none")
        weighted_box = (per_sample_box * stal_weight).sum() / stal_weight.sum().clamp_min(1.0)

        total = (
            l_dwbd
            + lambda1 * l_cst
            + lambda2 * verifier_scale * l_ver_kl
            + lambda3 * box_scale * weighted_box
        )
        scaled_total = total / grad_accum_steps

    scaler.scale(scaled_total).backward()
    return DistillStepOutput(
        total=total.detach(),
        l1=weighted_box.detach(),
        giou=torch.zeros_like(weighted_box.detach()),
        ver=l_ver_kl.detach(),
    )


def run_phase2_finetune_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str] | list[tuple[str, ...]]],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    lambda1: float,
    lambda3: float,
    tokenize_fn: TokenizeFn,
    epoch_index: int,
    total_epochs: int,
    stal_enabled: bool,
    stal_area_threshold: float,
    stal_max_boost: float,
    progloss_enabled: bool,
    progloss_box_start: float,
    progloss_box_end: float,
    progloss_power: float,
) -> DistillStepOutput:
    images = batch["images"].to(device, non_blocking=True)
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_box = batch["target_box"].to(device, non_blocking=True)
    phrases = list(batch.get("phrases", []))
    augmented_phrases = batch.get("augmented_phrases")
    augmented_phrase_sets = list(augmented_phrases) if augmented_phrases is not None else None

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(images=images, token_ids=token_ids, attention_mask=attention_mask)
        per_sample_box = box_loss(outputs["bbox"], target_box, reduction="none")

        l_cst = torch.zeros((), device=images.device, dtype=outputs["bbox"].dtype)
        maybe_augmented_inputs = _build_augmented_inputs(
            phrases=phrases,
            augmented_phrase_sets=augmented_phrase_sets,
            tokenize_fn=tokenize_fn,
            device=device,
        )
        if maybe_augmented_inputs is not None:
            aug_token_ids_tensor, aug_attention_mask_tensor = maybe_augmented_inputs
            aug_outputs = model(images=images, token_ids=aug_token_ids_tensor, attention_mask=aug_attention_mask_tensor)
            l_cst = consistency_loss([outputs["fused_tokens"], aug_outputs["fused_tokens"]])

        stal_weight = compute_stal_weight(
            target_box,
            area_threshold=stal_area_threshold,
            max_boost=stal_max_boost,
            enabled=stal_enabled,
        )
        box_scale = compute_progloss_scale(
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            start_scale=progloss_box_start,
            end_scale=progloss_box_end,
            power=progloss_power,
            enabled=progloss_enabled,
        )

        l_box = (per_sample_box * stal_weight).sum() / stal_weight.sum().clamp_min(1.0)

        total = lambda3 * box_scale * l_box + lambda1 * l_cst
        scaled_total = total / grad_accum_steps

    scaler.scale(scaled_total).backward()
    return DistillStepOutput(
        total=total.detach(),
        l1=l_box.detach(),
        giou=torch.zeros_like(l_box.detach()),
        ver=torch.zeros_like(l_box.detach()),
    )
