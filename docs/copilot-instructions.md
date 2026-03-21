# Lightweight Visual Grounding via Progressive Knowledge Distillation

## Project Overview
A research ML codebase implementing a **single-stage progressive knowledge distillation** framework that compresses Grounding DINO (Swin-L, ~200M params) into a ~33–36M param student model targeting <30 ms end-to-end latency on Snapdragon 8 Gen 3 / Apple A17 NPU. Target publication: **BMVC 2026** (Abstract: 22 May, Paper: 29 May).

Task: Visual grounding — given image `I` and text expression `E`, predict bounding box `b ∈ R^4`. Metric: **Accuracy@0.5** (fraction of predictions with IoU ≥ 0.5 vs ground truth).

## Implementation Strategy (required)
- Use a **hybrid implementation path**:
	- Reuse mature DL repo patterns for infrastructure (configs, loops, logging, launchers, evaluation harness)
	- Implement this project's research-specific logic from scratch (distillation losses, attention alignment, SelfEQ, ToB, verifier behavior)
- Avoid both extremes:
	- no one-repo hard fork as the whole project base
	- no full greenfield reinvention of solved infrastructure

## Tech Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.x (CUDA), torchvision
- **Backbone**: YOLOv8-small (ultralytics) — frozen, ~11M params
- **Text Encoder**: MobileCLIP-S1 text branch — frozen epochs 1–5, then fine-tuned at 0.1× LR, ~15M params
- **Teacher**: Grounding DINO (Swin-L) — inference-only, never backpropagated through
- **Data Storage**: HDF5 (h5py) for teacher feature shards (~80–120 GB FP16)
- **Augmentation / Scene Graphs**: Offline substitution of VLM-paraphrasing with rigid multi-key matched Scene Graph relationships from `_graphs_.pth`.
- **Compression**: ONNX → TensorRT (Jetson), CoreML (Apple Silicon), ONNX Runtime Mobile (ARM Android)
- **Quantisation**: INT8/INT4 QAT via PyTorch FX or brevitas
- **Experiment Tracking**: Weights & Biases (wandb)
- **Config Management**: Hydra (YAML configs under `configs/`)
- **Testing**: pytest

## Project Structure
```
project-root/
├── .github/
│   ├── copilot-instructions.md   ← You are here
│   └── instructions/             ← Scoped instructions per subsystem
├── AGENTS.md                     ← Universal agent instructions (same content)
├── configs/                      ← Hydra YAML configs
│   ├── data/
│   ├── model/
│   └── train/
├── data/
│   ├── raw/                      ← RefCOCO, RefCOCO+, RefCOCOg, ReferItGame
│   ├── scene_graphs/             ← JSON/Feature extracts from graphs
│   └── teacher_cache/            ← HDF5 shards (FP16 teacher features)
├── src/
│   ├── data/                     ← Dataset classes, loaders, augmentation
│   ├── models/
│   │   ├── teacher/              ← Grounding DINO wrapper (inference only)
│   │   ├── student/              ← Full student architecture
│   │   │   ├── backbone.py       ← YOLOv8-small FPN wrapper
│   │   │   ├── text_encoder.py   ← MobileCLIP-S1 text branch
│   │   │   ├── fusion.py         ← 4-layer fusion transformer + ToB merging
│   │   │   └── verifier.py       ← Re-ranking cross-attention head
│   │   └── losses.py             ← All loss functions (CE, KL, L1, GIoU, MSE, Attn, SelfEQ, Verifier)
│   ├── training/
│   │   ├── trainer.py
│   │   └── distill.py
│   ├── compression/
│   │   ├── pruning.py            ← L1-norm FFN pruning + gradient-norm attention head pruning
│   │   └── qat.py                ← INT8/INT4 quantisation-aware training
│   └── export/
│       ├── export_onnx.py
│       ├── export_trt.py
│       └── export_coreml.py
├── scripts/
│   ├── precompute_teacher.py     ← Phase 1: cache teacher features to HDF5
│   ├── extract_scene_graphs.py   ← Phase 0: offline scene graph structure extraction
│   ├── train.py                  ← Main training entry point
│   ├── compress.py               ← Phase 3: pruning + QAT
│   └── evaluate.py               ← Benchmark on all 4 datasets
├── tests/
├── notebooks/
├── docs/                         ← Extended component and design docs
└── outputs/                      ← checkpoints, logs, exports
```

## Coding Guidelines
- Use type hints everywhere (`from __future__ import annotations`)
- Configs via Hydra; never hardcode paths or hyperparameters inline
- All dataset splits (RefCOCO testA/testB, RefCOCOg val/test, ReferItGame test) must be evaluated in the same eval run
- Teacher is **always** in `torch.no_grad()` and `model.eval()` — never modify teacher weights
- Loss weights λ1–λ7 and KL temperature `T` are ablation hyperparameters — keep them in `configs/train/`
- Use `wandb.log()` for all scalar metrics every epoch; log Accuracy@0.5 per split
- HDF5 shards are keyed by dataset split + sample index; document shard schema in `docs/teacher_cache.md`
- Before student training starts, assert recall@200 gate: ≥97% RefCOCO, ≥94% ReferItGame
- All new modules must have a unit test in `tests/`

## Efficiency & Reproducibility Requirements
- Training must default to AMP (`autocast` + `GradScaler`) unless explicitly disabled via config
- Effective batch size must be configurable using gradient accumulation
- Gradient clipping, optional activation checkpointing, and dataloader throughput tuning are required config surfaces
- Precision/distributed/compile/checkpointing behavior must be toggled via Hydra, never via scattered inline constants
- Minimum run metadata to log each epoch: per-split Accuracy@0.5, loss components, effective batch size, peak GPU memory, epoch time, precision mode, gradient scale
- Seed and experiment config snapshot must be persisted in checkpoints and wandb metadata

## Phase 0 Scene Graph Policy
- Phase 0 is deterministic scene-graph augmentation, not VLM paraphrasing.
- Exactly 3 augmented queries per train/val sample.
- Use multi-key matching on `(image, phrase)` and `(image, mask_id)` when aligning `_graphs_.pth` records.

- We do not use VLM paraphrasing anymore. We strictly use Scene Graph relationships loaded via multi-key `(image, phrase)|(image, mask_id)` correlation.
- Ensure `0` poisoned records by leveraging deterministic lookup matrices validated against original bounding boxes.

## Key Commands
```bash
# Environment
pip install -r requirements.txt

# Phase 0: scene-graph augmentation
python scripts/generate_paraphrases.py dataset=refcocog

# Phase 1: precompute teacher features
python scripts/precompute_teacher.py dataset=all

# Phase 2: train student
python scripts/train.py experiment=student_full

# Phase 3: compress
python scripts/compress.py checkpoint=outputs/checkpoints/best.pth

# Evaluate
python scripts/evaluate.py checkpoint=outputs/checkpoints/compressed.pth

# Run tests
pytest tests/ -v
```

Kickoff reference: `docs/quickstart.md`

## Critical Constraints
- Latency target: **<30 ms** end-to-end on Snapdragon 8 Gen 3 / Apple A17 NPU (hard constraint)
- Parameter budget: ~33–36M total, ~18–22M trainable at peak — soft constraint
- INT8-quantized final model: **≈30 MB** on disk
- Never train teacher; never include teacher in export artifacts
- Scene graph context: perfectly matched structural graphs containing `nodes` and `edges`

## IMPORTANT: AFTER EVERY SIGNIFICANT CHANGE YOU MAKE, YOU NEED TO UPDATE THE APPROPRIATE md files