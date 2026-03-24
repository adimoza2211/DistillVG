# DistillVG

Compact visual grounding with online verifier-guided distillation.

This README is the current-state document. It records what is actually in the repo right now, what is legacy, and what the intended runtime flow is supposed to become.

## Canonical Docs
- [docs/pipeline_plan.md](docs/pipeline_plan.md)
- [.github/copilot-instructions.md](.github/copilot-instructions.md)

## Current State

The repository is currently in a transition state.

What is already established:
- Deterministic scene-graph augmentation is the active Phase 0 direction.
- The student training entrypoint exists at [scripts/train.py](scripts/train.py).
- The distillation trainer lives in [src/training/trainer.py](src/training/trainer.py).
- The active Phase-1/Phase-2 loss stack is now under [src/losses/](src/losses/) with combined loss composition in [src/losses/combined.py](src/losses/combined.py).
- The student implementation is split across [src/models/student/model.py](src/models/student/model.py), [src/models/student/backbone.py](src/models/student/backbone.py), [src/models/student/text_encoder.py](src/models/student/text_encoder.py), [src/models/student/fusion.py](src/models/student/fusion.py), and [src/models/student/verifier.py](src/models/student/verifier.py).
- Fusion depth and attention head count are now consumed from Hydra config (`model.fusion_layers`, `model.attention_heads`) and directly control student capacity.
- Pre-fusion visual token budget is now consumed from Hydra config (`model.roi_tokens`) and applied in the student forward path.
- Phase 1 has started: the trainer now builds a frozen online verifier module and computes verifier supervision inside each training step.
- The distillation step now converts student proposal boxes into online crops and scores top-K proposals with a frozen verifier path.
- Verifier backend is now configured InternVL-first in Hydra, with explicit 4-bit loading scaffolding.
- Verifier query text now pulls augmentation candidates from `_aug.pth` files and constructs True/False crop-matching prompts.
- The phrase-bank parser is now aligned to the real `_aug.pth` tuple contract (paraphrases from slot index `3`).
- The student now emits proposal-level boxes and proposal-level plausibility logits used for verifier-supervised ranking.
- Proposal generation now uses an Ultralytics YOLO26-backed path for high-recall top-200 proposals.
- Two-stage verifier utilities are now present (Stage-1 crop scoring + Stage-2 highlighted full-image 4-class scoring helpers).
- The trainer dataloader now uses real image-backed `_aug.pth` records and no longer depends on the synthetic training scaffold.
- The trainer dataloader now balances source splits using a `WeightedRandomSampler` built from `_aug.pth` sources.
- The Phase 1 loss path is now aligned to live verifier supervision plus box losses (legacy placeholder feature-distillation terms removed).
- Consistency loss now runs from student-only augmented-query forwards using fused-token representations.
- Phase 2 has started with DWBD dynamic loss balancing and MuSGD optimizer support.
- Phase 2 strict verifier-unloaded fine-tuning is now fully wired as `train.mode=phase2_strict` using only `L_box + L_cst`.
- STAL small-target weighting and ProgLoss progressive scaling are now wired into both phase1 and phase2 loss composition via Hydra config.
- The repo already has a Hydra-based configuration layout under [configs/](configs/).
- There are working scene-graph extraction and integrity-check scripts: [scripts/extract_scene_graphs.py](scripts/extract_scene_graphs.py) and [scripts/verify_augmentation_integrity.py](scripts/verify_augmentation_integrity.py).

What is still legacy or inconsistent with the target design:
- Training pipeline is launch-ready for Phase 1 and Phase 2.
- Remaining work is Phase 3 only: compression, export, and deployment artifact validation.

What the target runtime is supposed to be:
- A YOLO26-small student produces proposal boxes in the forward pass.
- Those boxes are used immediately to crop the original image.
- A frozen InternVL 3.5-8B verifier, loaded in 4-bit quantization, scores those crops in the same training step.
- The verifier stays in VRAM, frozen, under `eval()` and `no_grad()`.
- The loss uses verifier logits as supervision for the student.
- Only the student is updated by backpropagation.

What the student is supposed to look like:
- Visual backbone: a lightweight YOLO26-small style encoder that produces proposal-aligned visual tokens.
- Text branch: a MobileCLIP-S1 style text encoder for the grounding expression.
- Fusion: a small fusion block that mixes visual tokens with text context.
- Box regression: a head that turns fused features into the grounding box.
- Plausibility scoring: a student-side verifier head that predicts how well a proposal matches the phrase.
- Deployment target: the student is the only model that should survive export and compression.

What this means operationally:
- There is no persistent cache in the intended online setup.
- There is no offline HDF5 pseudo-label store in the target pipeline.
- Candidate boxes are generated and consumed inside the same step.
- Any reuse is in-memory within the batch/step, not a disk-backed artifact.
- Local debugging can still use a mock verifier via Hydra override: `verifier.backend=mock`.

## Current Repo Layout

The docs have been intentionally collapsed to three categories:
- Current state: this file
- Pipeline plan: [docs/pipeline_plan.md](docs/pipeline_plan.md)
- Coding instructions: [.github/copilot-instructions.md](.github/copilot-instructions.md)

Relevant code areas:
- [configs/](configs/) for Hydra configuration
- [scripts/](scripts/) for entrypoints
- [src/data/](src/data/) for augmentation and dataset plumbing
- [src/models/](src/models/) for student, verifier runtime, and loss code
- [src/training/](src/training/) for the training loop and distillation step
- [tests/](tests/) for unit coverage

## Active vs Legacy Files

Active or still-relevant files:
- [scripts/train.py](scripts/train.py)
- [scripts/extract_scene_graphs.py](scripts/extract_scene_graphs.py)
- [scripts/verify_augmentation_integrity.py](scripts/verify_augmentation_integrity.py)
- [src/training/trainer.py](src/training/trainer.py)
- [src/models/losses.py](src/models/losses.py)
- [src/models/student/](src/models/student/)

Legacy files that should not define the target architecture anymore:
- None at the moment; the cache-oriented code was pruned in this pass.

## Immediate Next Work
- Implement Phase 3 compression parity (structured pruning + QAT).
- Implement and validate Phase 3 export parity (ONNX/TensorRT/CoreML student-only artifact).

## Hyperparameters
- The authoritative optimizer/schedule/loss/verifier/compression hyperparameter spec is maintained in [docs/pipeline_plan.md](docs/pipeline_plan.md) under **Hyperparameter Specification (Authoritative)**.

## Start Training
- Minimal smoke run (fast sanity check):
	- `conda activate distill && python scripts/train.py verifier.backend=mock model.proposal_generator.use_yolo26=false training.epochs=1 training.max_records=8 training.micro_batch_size=2 data.num_workers=0 data.persistent_workers=false train.mode=phase1`
- Full Phase 1 run (online InternVL verifier, YOLO26 proposals):
	- `conda activate distill && python scripts/train.py`
- Phase 2 strict fine-tuning run (verifier unloaded):
	- `conda activate distill && python scripts/train.py train.mode=phase2_strict`

## Short Version

The repo is no longer being documented as a cache-first pipeline.
The intended design is online distillation with simultaneous student and verifier execution.
Phase 1 online verifier-guided distillation is now implemented in-repo with real `_aug.pth` records, proposal-level top-K verifier scoring, and live verifier-supervised losses.
