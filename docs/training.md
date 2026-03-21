# Training Guide

Kickoff and run order: `docs/quickstart.md`
Efficiency policy details: `docs/efficiency_playbook.md`
Scene graph augmentation policy: `docs/scene_graph_augmentation.md`

## Phases Summary
| Phase | Script | GPU estimate |
|---|---|---|
| 0 — Scene graph augmentation | `scripts/generate_paraphrases.py` | CPU / small GPU |
| 1 — Teacher precomputation | `scripts/precompute_teacher.py` | ~1 A100-day |
| 2 — Student training | `scripts/train.py` | ~4–5 A100-days |
| 3 — Compression + QAT | `scripts/compress.py` | ~1 A100-day |

## Training Schedule (Phase 2)
- Epochs 1–5: text encoder frozen, backbone frozen, only fusion transformer + verifier trainable
- Epoch 6+: text encoder unfrozen at 0.1× base LR
- Token Blurring active throughout (200 → 150 after fusion layer 2)
- Joint pretraining on all 4 datasets, then per-dataset fine-tuning

## Current Implemented Baseline (2026-03)
- A runnable synthetic Phase 2 baseline is implemented to validate training plumbing:
  - Hydra config loading via `configs/train.yaml`
  - AMP + `GradScaler`
  - Gradient accumulation and gradient clipping
  - Modular student path (`backbone`, `text_encoder`, `fusion`, `verifier`, `bbox_head`)
  - Synthetic grounding dataset + collate function for smoke training
- This baseline is an execution harness only; production dataset/model integrations are the next step.

Smoke run:
```bash
conda activate distill
python scripts/train.py
```

## Efficiency Defaults (required)
- AMP enabled by default (`autocast` + `GradScaler`)
- Gradient accumulation enabled to reach target effective batch size under memory limits
- Gradient clipping required (`clip_grad_norm_`) for distillation stability
- Activation checkpointing supported for fusion layers (enable when memory-bound)
- Dataloader performance knobs exposed in config (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`)
- Runtime knobs exposed via Hydra (compile/distributed/precision/checkpointing)

Recommended initial config envelope:
- Precision: FP16 AMP on A100; BF16 only when hardware+stack are validated
- Accumulation: start with 2–4 steps and tune by throughput vs convergence
- Compile: use guarded enable flag (`torch.compile`) with fallback path
- Distributed: DDP optional, enabled only after single-GPU baseline stability

## Loss Weights (configure in `configs/train/losses.yaml`)
```yaml
lambda_kl: 1.0      # KL distillation loss
lambda_l1: 5.0      # L1 box regression
lambda_giou: 2.0    # GIoU loss
lambda_mse: 1.0     # Feature MSE distillation
lambda_attn: 1.0    # Attention map distillation
lambda_cst: 0.5     # SelfEQ consistency loss
lambda_ver: 1.0     # Verifier head BCE loss
kl_temp_start: 4.0  # KL temperature (annealed over training)
kl_temp_end: 1.0
```
These are initial values; ablation studies will determine final values.

## Wandb Logging
- Log every epoch: Accuracy@0.5 per split (refcoco_val, refcoco_testA, refcoco_testB, refcoco+_val, refcocog_val, referit_test)
- Log all individual loss components
- Log recall@200 gate values at start of training
- Log effective batch size, precision mode, gradient scale, peak GPU memory, and epoch wall-time

## Checkpointing
- Save to `outputs/checkpoints/` — best by RefCOCO val Accuracy@0.5
- Include: model state_dict, optimizer, epoch, all metric scores, config snapshot

## Reproducibility Contract
- Persist seed, Hydra config snapshot, and git commit hash in each checkpoint
- Keep teacher in deterministic inference path (`eval` + `no_grad`) for all runs
- Ensure all toggles that affect numerics are config-tracked (precision, compile, checkpointing)

## Phase 3 — Compression
```bash
python scripts/compress.py \
  checkpoint=outputs/checkpoints/best.pth \
  pruning.ffn_ratio=0.2 \
  pruning.finetune_epochs=5 \
  qat.epochs=5 \
  qat.calib_samples=15000
```
Calibration corpus: 15,000 samples stratified across expression-length buckets (short/medium/long).
