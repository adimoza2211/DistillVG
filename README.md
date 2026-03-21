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
- The student loss stack is in [src/models/losses.py](src/models/losses.py).
- The student implementation is split across [src/models/student/model.py](src/models/student/model.py), [src/models/student/backbone.py](src/models/student/backbone.py), [src/models/student/text_encoder.py](src/models/student/text_encoder.py), [src/models/student/fusion.py](src/models/student/fusion.py), and [src/models/student/verifier.py](src/models/student/verifier.py).
- The current training code is still a synthetic scaffold; the online verifier step is not wired into the trainer yet.
- The repo already has a Hydra-based configuration layout under [configs/](configs/).
- There are working scene-graph extraction and integrity-check scripts: [scripts/extract_scene_graphs.py](scripts/extract_scene_graphs.py) and [scripts/verify_augmentation_integrity.py](scripts/verify_augmentation_integrity.py).

What is still legacy or inconsistent with the target design:
- The old cache precompute and teacher-wrapper code have been removed.
- The trainer still needs to be rewired from the synthetic scaffold to the live verifier path.
- The student-side wiring and losses still need to be connected to the actual online verifier execution.

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

## Current Repo Layout

The docs have been intentionally collapsed to three categories:
- Current state: this file
- Pipeline plan: [docs/pipeline_plan.md](docs/pipeline_plan.md)
- Coding instructions: [.github/copilot-instructions.md](.github/copilot-instructions.md)

Relevant code areas:
- [configs/](configs/) for Hydra configuration
- [scripts/](scripts/) for entrypoints
- [src/data/](src/data/) for augmentation and dataset plumbing
- [src/models/](src/models/) for student, verifier, and loss code
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
- Rewire training so verifier scoring happens online in the same forward pass.
- Make the verifier loading path 4-bit and VRAM-resident.
- Replace the synthetic trainer scaffold with the real live-verifier path.
- Keep pruning dead code as each replaced subsystem becomes obsolete.

## Short Version

The repo is no longer being documented as a cache-first pipeline.
The intended design is online distillation with simultaneous student and verifier execution.
The current code still contains a synthetic training scaffold that must be replaced by the live verifier path.
