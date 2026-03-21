# Student Model — Component Specs

## Overview
~33–36M total parameters. ~18–22M trainable at peak (after text encoder unfreezing at epoch 6).
All components project to a shared dimension `d = 384`.

## Visual Backbone — `src/models/student/backbone.py`
- **Model**: YOLOv8-small (ultralytics)
- **Params**: ~11M, **frozen** throughout training
- **Output**: FPN feature maps at strides 8, 16, 32
- **Proposals**: Top-200 class-agnostic proposals via NMS
- **RoI-Align**: 7×7 crops (49 spatial cells per proposal) → projected to d=384
- **Why 7×7**: Triples spatial resolution vs 4×4; parameter-free; critical for left/right/center disambiguation on RefCOCO testA/testB

## Text Encoder — `src/models/student/text_encoder.py`
- **Model**: MobileCLIP-S1 text branch
- **Params**: ~15M
- **Training schedule**: Frozen epochs 1–5; unfrozen at 0.1× base LR from epoch 6
- **Output**: d=384 text embedding

## Fusion Transformer — `src/models/student/fusion.py`
- **Layers**: 4 transformer layers
- **Dimension**: d=384, 6 attention heads
- **Params**: ~7M
- **Token Blurring (ToB)**: Applied after layer 2; reduces active visual tokens 200 → ~150 via similarity-weighted merging (not hard pruning). Retaining 150 (not 130) preserves spatial reference frame.
- **Input**: 200 RoI-projected visual tokens + text token sequence
- **Why merge-not-drop**: Hard token pruning destroys spatial reference frame; merging preserves it
- **Efficiency interface requirements**:
	- Attention backend and checkpointing must be config-driven
	- Token count ceilings (`200` input, `~150` post-ToB) are hard constraints for memory predictability
	- Module must remain AMP-safe and DDP-safe

## Re-ranking Verifier Head — `src/models/student/verifier.py`
- **Type**: Lightweight cross-attention module
- **Params**: ~0.4M
- **Input**: Top-k candidate RoI features + pooled expression embedding
- **Output**: Scalar plausibility score per proposal
- **Usage**: Re-weights box coordinate predictions at inference; trained with `L_ver` (BCE)
- **Especially effective**: On ReferItGame where spatial expressions are ambiguous

## System-level compute envelope
- Preserve training-time feasibility under AMP + gradient accumulation baseline
- Avoid architectural modifications that violate <30 ms deployment target by default
- Keep additions modular and switchable through Hydra for ablation control

## Knowledge Distillation Novelties
1. **Backbone-teacher structural alignment**: YOLOv8 C2f maps are structurally compatible with teacher spatial features → feature-MSE is semantically meaningful
2. **Depth-matched attention distillation**: Student's 4 fusion layers supervised against grouped averages of teacher's multi-layer cross-attention maps (prevents geometric misalignment from asymmetric depth)
3. **SelfEQ as regulariser** (`L_cst`): Enforces expression-invariant attention patterns at zero inference cost
4. **Token Blurring**: Merge-not-drop for grounding spatial preservation
5. **Re-ranking verifier**: Post-fusion plausibility scoring without increasing backbone/text encoder footprint
