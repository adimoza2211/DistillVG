"""Precompute YOLO proposals + InternVL verifier targets offline.

Produces a cache file that the Trainer loads to skip online verifier calls
during training, reducing per-step time from ~20s to ~0.05s.

Usage:
    python scripts/precompute_targets.py [hydra overrides]

Example:
    python scripts/precompute_targets.py \
        verifier.backend=internvl \
        training.available_gpu_ids=0

The cache is saved to the path specified by train.precomputed_cache_path.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

# Bootstrap project root onto sys.path.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.grounding import (
    AugmentedGroundingDataset,
    GroundingRecord,
    load_grounding_records,
)
from src.models.student.proposals import YOLO26ProposalGenerator
from src.models.verifier.prompting import build_verifier_queries, build_stage2_verifier_queries
from src.models.verifier.runtime import BaseOnlineVerifier, build_online_verifier
from src.models.verifier.two_stage import build_verifier_targets, draw_highlighted_proposals
from src.training.verifier_crops import crop_and_resize_from_proposals, normalize_xyxy_boxes
from src.utils.logging import get_logger

logger = get_logger("distillvg.precompute")


def _configure_visible_gpus(cfg: DictConfig) -> None:
    available_gpu_ids = str(getattr(cfg.training, "available_gpu_ids", "")).strip()
    if available_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu_ids


def _resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(cfg.training.device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_all_records(cfg: DictConfig) -> list[GroundingRecord]:
    """Load training records from configured glob patterns."""
    records: list[GroundingRecord] = []
    for pattern_key in ("train_records_original_glob", "train_records_augmented_glob"):
        pattern = str(getattr(cfg.data, pattern_key, "")).strip()
        if pattern:
            records.extend(load_grounding_records(pattern=pattern))
    return records


def _build_yolo_generator(cfg: DictConfig) -> YOLO26ProposalGenerator:
    pg = cfg.model.proposal_generator
    return YOLO26ProposalGenerator.from_config(
        max_proposals=int(cfg.model.proposal_count),
        conf_threshold=float(pg.yolo26_conf_threshold),
        iou_threshold=float(pg.yolo26_iou_threshold),
        model_cfg=str(pg.yolo26_model_cfg),
        weights_path=str(pg.yolo26_weights_path),
    )


def _preprocess_image_for_yolo(
    dataset: AugmentedGroundingDataset,
    record: GroundingRecord,
) -> torch.Tensor:
    """Load and preprocess an image identically to how the training loop sees it."""
    from PIL import Image

    image_path = dataset._resolve_image_path(record)
    image = Image.open(image_path).convert("RGB")
    image_tensor, _, _, _ = dataset._resize_and_pad_image(image)
    return image_tensor


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    _configure_visible_gpus(cfg)
    device = _resolve_device(cfg)

    cache_path = str(getattr(cfg.train, "precomputed_cache_path", "")).strip()
    if not cache_path:
        cache_path = "outputs/precomputed_cache.pt"
        logger.info("No precomputed_cache_path configured; defaulting to %s", cache_path)

    # --- Load records ---
    records = _load_all_records(cfg)
    if not records:
        raise RuntimeError("No training records found.")
    logger.info("Loaded %d training records.", len(records))

    # --- Build dataset (for image loading/preprocessing only) ---
    dataset = AugmentedGroundingDataset(
        records=records,
        image_root=str(cfg.verifier.image_root),
        resize_long_edge=int(cfg.training.resize_long_edge),
        padded_square_size=int(cfg.training.padded_square_size),
        max_query_length=int(cfg.training.max_query_length),
        text_vocab_size=int(cfg.training.text_vocab_size),
        source_image_roots={
            str(k): str(v) for k, v in getattr(cfg.verifier, "source_image_roots", {}).items()
        },
        source_image_styles={
            str(k): str(v) for k, v in getattr(cfg.verifier, "source_image_styles", {}).items()
        },
    )

    # --- Build YOLO generator ---
    use_yolo = bool(cfg.model.proposal_generator.use_yolo26)
    yolo_gen: YOLO26ProposalGenerator | None = None
    if use_yolo:
        yolo_gen = _build_yolo_generator(cfg)
        logger.info("YOLO26 proposal generator loaded.")

    # --- Build online verifier ---
    online_verifier = build_online_verifier(cfg=cfg.verifier, device=device)
    logger.info("Online verifier (%s) loaded.", cfg.verifier.backend)

    # --- Config ---
    top_k_proposals = int(cfg.verifier.top_k_proposals)
    stage2_top_k = int(cfg.verifier.stage2_top_k)
    crop_size = int(cfg.verifier.crop_size)
    proposal_count = int(cfg.model.proposal_count)

    # --- Deduplicate images ---
    image_to_record_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        image_to_record_indices.setdefault(record.image_name, []).append(idx)

    unique_images = list(image_to_record_indices.keys())
    logger.info(
        "Processing %d unique images across %d records.",
        len(unique_images),
        len(records),
    )

    # --- Output storage ---
    proposals_cache: dict[str, torch.Tensor] = {}
    targets_cache: dict[tuple[str, str], torch.Tensor] = {}
    masks_cache: dict[tuple[str, str], torch.Tensor] = {}

    # --- Try to resume from partial cache ---
    partial_cache_path = Path(cache_path + ".partial")
    if partial_cache_path.exists():
        logger.info("Found partial cache at %s — resuming.", partial_cache_path)
        partial = torch.load(partial_cache_path, weights_only=False)
        proposals_cache = partial.get("proposals", {})
        targets_cache = partial.get("targets", {})
        masks_cache = partial.get("masks", {})
        logger.info(
            "Resumed with %d proposal sets, %d target sets.",
            len(proposals_cache),
            len(targets_cache),
        )

    start_time = time.time()
    save_interval = 500  # Save partial cache every N images.

    for img_idx, image_name in enumerate(tqdm(unique_images, desc="Precomputing")):
        # Skip if already computed for ALL records of this image.
        record_indices = image_to_record_indices[image_name]
        all_done = image_name in proposals_cache and all(
            (records[ri].image_name, records[ri].phrases[0]) in targets_cache
            for ri in record_indices
        )
        if all_done:
            continue

        # --- Load and preprocess image ---
        representative_record = records[record_indices[0]]
        try:
            image_tensor = _preprocess_image_for_yolo(dataset, representative_record)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s — skipping.", image_name, exc)
            continue

        image_batch = image_tensor.unsqueeze(0).to(device)

        # --- YOLO proposals ---
        if image_name not in proposals_cache:
            if yolo_gen is not None:
                with torch.no_grad():
                    yolo_proposals = yolo_gen.generate(image_batch)
                proposals_cache[image_name] = yolo_proposals.squeeze(0).cpu()
            else:
                # Learned proposals — can't precompute without the model.
                proposals_cache[image_name] = torch.zeros(proposal_count, 4)

        proposal_tensor = proposals_cache[image_name].unsqueeze(0).to(device)
        # Normalize YOLO proposals.
        proposal_boxes = normalize_xyxy_boxes(proposal_tensor)
        stage1_k = max(1, min(top_k_proposals, proposal_boxes.shape[1]))
        proposal_boxes_stage1 = proposal_boxes[:, :stage1_k, :]

        # --- Crop proposals ---
        crops = crop_and_resize_from_proposals(
            images=image_batch.detach(),
            boxes_xyxy=proposal_boxes_stage1,
            output_size=crop_size,
        )

        # --- Process each phrase for this image ---
        for record_idx in record_indices:
            record = records[record_idx]
            phrase = record.phrases[0]
            cache_key = (image_name, phrase)

            if cache_key in targets_cache:
                continue

            with torch.no_grad():
                # Stage 1: binary scoring of crops.
                verifier_queries = build_verifier_queries(
                    base_phrases=[phrase],
                    augmented_phrase_sets=None,
                    use_augmented=False,
                    selection_strategy="first",
                )
                repeated_queries = [q for q in verifier_queries for _ in range(stage1_k)]

                stage1_scores = online_verifier(
                    crops=crops,
                    queries=repeated_queries,
                )
                stage1_scores = stage1_scores.reshape(1, stage1_k)

                # Stage 2: 4-class scoring on highlighted images.
                s2_k = max(1, min(stage2_top_k, stage1_k))
                top_stage2_indices = stage1_scores.topk(s2_k, dim=-1).indices

                highlighted_images = draw_highlighted_proposals(
                    images=image_batch,
                    proposal_boxes=proposal_boxes_stage1,
                    top_indices=top_stage2_indices,
                )

                stage2_queries = build_stage2_verifier_queries(verifier_queries, top_k=s2_k)
                stage2_logits = online_verifier.score_4class(
                    images=highlighted_images,
                    queries=stage2_queries,
                )
                stage2_soft = torch.softmax(stage2_logits, dim=-1).reshape(1, s2_k, 4)

                verifier_targets, verifier_mask = build_verifier_targets(
                    stage1_scores=stage1_scores,
                    stage2_soft_labels=stage2_soft,
                    top_indices=top_stage2_indices,
                )

                targets_cache[cache_key] = verifier_targets.squeeze(0).cpu()
                masks_cache[cache_key] = verifier_mask.squeeze(0).cpu()

        # --- Periodic partial save ---
        if (img_idx + 1) % save_interval == 0:
            _save_cache(partial_cache_path, proposals_cache, targets_cache, masks_cache)
            logger.info(
                "Partial save at image %d/%d (%.1f%%).",
                img_idx + 1,
                len(unique_images),
                100.0 * (img_idx + 1) / len(unique_images),
            )

    # --- Final save ---
    output_path = Path(cache_path)
    _save_cache(output_path, proposals_cache, targets_cache, masks_cache)

    # Clean up partial file.
    if partial_cache_path.exists():
        partial_cache_path.unlink()

    elapsed = time.time() - start_time
    logger.info(
        "Precomputation complete in %.1fs. Saved %d proposal sets and %d target sets to %s",
        elapsed,
        len(proposals_cache),
        len(targets_cache),
        output_path,
    )


def _save_cache(
    path: Path,
    proposals: dict[str, torch.Tensor],
    targets: dict[tuple[str, str], torch.Tensor],
    masks: dict[tuple[str, str], torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"proposals": proposals, "targets": targets, "masks": masks},
        path,
    )


if __name__ == "__main__":
    main()
