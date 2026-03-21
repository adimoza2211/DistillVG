# Quickstart — DistillVG

## Goal
Get from empty environment to Phase 0–3 execution with consistent configs, reproducible runs, and documented checkpoints.

## 1) Startup Strategy (must follow)
Use a **hybrid** path:
- Reuse mature infrastructure patterns (Hydra config composition, runners, logging, evaluation harness)
- Implement project novelties from scratch (distillation objectives, SelfEQ, ToB behavior, verifier logic)

Do not hard-fork one repo as whole base. Do not reinvent solved infra from scratch.

## 2) Environment
```bash
conda activate distill
pip install -r requirements.txt
python scripts/organize_raw_data.py
python scripts/prepare_data.py
```

Minimum assumptions:
- Python 3.10+
- PyTorch 2.x CUDA environment
- Sufficient local storage for teacher cache (~80–120 GB FP16 HDF5)

## 3) Data Layout Check
Expected layout:
- `data/raw/refcoco/`
- `data/raw/refcoco+/`
- `data/raw/refcocog/`
- `data/raw/referitgame/`

See `docs/datasets.md` for schema and split details.

## 4) Run Pipeline in Order
```bash
conda activate distill
 python scripts/generate_paraphrases.py dataset=refcocog
python scripts/precompute_teacher.py dataset=all
python scripts/train.py experiment=student_full
python scripts/compress.py checkpoint=outputs/checkpoints/best.pth
python scripts/evaluate.py checkpoint=outputs/checkpoints/compressed.pth
pytest tests/ -v
```

## 5) Mandatory Runtime Policies
- Teacher always `eval()` + `torch.no_grad()`
- No hardcoded hyperparameters or paths; use Hydra
- Train with AMP + GradScaler by default
- Use gradient accumulation for effective batch scaling
- Log all required split metrics in one evaluation pass

## 6) Phase 0 Augmentation
Phase 0 now uses deterministic scene-graph augmentation instead of VLM paraphrasing.

Supported workflow:
- `scripts/generate_paraphrases.py` as the user-facing entry point for `_aug.pth` generation
- `scripts/extract_scene_graphs.py` as the lower-level augmentation engine

Output contract:
- Exactly 3 augmented queries per train/val sample
- Scene-graph-derived phrase variants only
- Deterministic lookup via `(image, phrase)` and `(image, mask_id)` alignment

## 7) Where to read next
- Architecture details: `docs/architecture.md`
- Training defaults and logging: `docs/training.md`
- Compression/export path: `docs/compression_export.md`
- Current status tracker: `docs/repo_state.md`
- Efficiency policy deep dive: `docs/efficiency_playbook.md`
- Scene graph augmentation policy: `docs/scene_graph_augmentation.md`
