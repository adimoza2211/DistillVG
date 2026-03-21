# Architecture Reference

Start here for setup/run order: `docs/quickstart.md`

## Pipeline Phases

### Phase 0 — Data & Augmentation
- Joint pretraining on all 4 benchmarks: RefCOCO, RefCOCO+, RefCOCOg, ReferItGame
- Offline structural augmentation replaces VLM-paraphrasing by securely indexing target-centered *Scene Graph Annotations*:
  - Deterministic alignment guarantees 0 poisoned records using dual multi-key matching `(image, phrase)` | `(image, mask_id)`.
  - Extracted properties include object attributes (e.g., "distant", "volcanic"), neighbor relationships, and directional edges involving the `root` (the grounding query target).
- Scene Graph rewrites or structured fields act as consistent variants of the grounding phrase.
- User-facing entry point: `scripts/generate_paraphrases.py` (compatibility wrapper around the scene-graph augmentation engine)
- Final per-dataset fine-tuning after joint pretraining

### Phase 1 — Teacher Feature Precomputation
- Teacher: Grounding DINO (Swin-L), always `eval()` + `no_grad()`
- Cached per sample (HDF5 FP16, ~80–120 GB total across 4 datasets):
  - Top-200 proposal boxes + logits
  - Per-proposal features (dim 256)
  - Cross-attention maps over expression tokens
  - Multimodal pooled embedding
- Recall@200 gate: ≥97% RefCOCO, ≥94% ReferItGame must pass before Phase 2

### Phase 2 — Student Training
See `docs/student_model.md` for detailed component specs.

Loss: `L = L_CE + λ1·L_KL(T) + λ2·L_L1 + λ3·L_GIoU + λ4·L_MSE + λ5·L_attn + λ6·L_cst + λ7·L_ver`

- `L_cst`: SelfEQ consistency loss — MSE between cross-attention maps from original vs augmented expression
- `L_ver`: Binary CE on verifier head plausibility scores (positive: IoU ≥ 0.5 with GT)
- λ1–λ7 and KL temperature T are treated as primary ablation hyperparameters

Efficiency baseline for Phase 2:
- AMP with GradScaler
- Gradient accumulation for effective batch scaling
- Optional activation checkpointing on fusion blocks when memory constrained
- High-throughput dataloading and reproducible logging
- Config-first controls for precision/distributed/compile/checkpointing

### Phase 3 — Compression & Export
**Pruning:**
- Fusion transformer FFN: 20% channel pruning by L1-norm ranking
- Attention heads: 1 head removed per layer by gradient-norm ranking
- 5 fine-tuning epochs post-pruning

**QAT:**
- Mixed-precision INT8/INT4 quantisation-aware training, 5 epochs
- Calibration corpus: 15,000 samples stratified across expression-length buckets

**Export targets:**
- ONNX → TensorRT (Jetson)
- CoreML (Apple Silicon)
- ONNX Runtime Mobile (ARM Android)

## Target Performance

| Model Stage | RefCOCO val | RefCOCO+ val | RefCOCOg val | ReferItGame test |
|---|---|---|---|---|
| Teacher (G-DINO) | ~92% | ~88% | ~86% | ~78% |
| Student (pre-compress) | 90–93% | 85–89% | 82–86% | 73–77% |
| Final Compressed (INT8) | 89–91% | 84–87% | 81–85% | 72–76% |
| Final model size | — | — | — | ~30 MB (INT8) |
| Latency target | <30 ms | <30 ms | <30 ms | <30 ms |

Hardware targets: Snapdragon 8 Gen 3, Apple A17 NPU (measured on physical hardware).

Related docs:
- `docs/training.md`
- `docs/student_model.md`
- `docs/compression_export.md`
