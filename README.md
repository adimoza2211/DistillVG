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
- Phase 1 has started: the trainer now builds a frozen online verifier module and computes verifier supervision inside each training step.
- The distillation step now converts student-predicted boxes into online crops and scores them with a frozen verifier path.
- Verifier backend is now configured InternVL-first in Hydra, with explicit 4-bit loading scaffolding.
- Verifier query text now pulls augmentation candidates from `_aug.pth` files and constructs True/False crop-matching prompts.
- The phrase-bank parser is now aligned to the real `_aug.pth` tuple contract (paraphrases from slot index `3`).
- The repo already has a Hydra-based configuration layout under [configs/](configs/).
- There are working scene-graph extraction and integrity-check scripts: [scripts/extract_scene_graphs.py](scripts/extract_scene_graphs.py) and [scripts/verify_augmentation_integrity.py](scripts/verify_augmentation_integrity.py).

What is still legacy or inconsistent with the target design:
- The active dataloader path is still synthetic and not yet connected to the real grounding dataset pipeline.
- The InternVL backend path is scaffolded for 4-bit loading, but generation-logit extraction for True/False supervision is still pending.
- The training objective still contains legacy placeholder loss inputs that should be replaced with real proposal-level verifier supervision.

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
- Replace synthetic data loading with real grounding minibatches that carry verifier-ready text prompts.
- Complete the InternVL scoring bridge for crop+prompt True/False logit extraction.
- Upgrade from single-box online scoring to proposal-set scoring and ranking supervision.
- Remove legacy placeholder distillation inputs once proposal-level verifier logits are active.

## Short Version

The repo is no longer being documented as a cache-first pipeline.
The intended design is online distillation with simultaneous student and verifier execution.
The current code now includes an initial online verifier scaffold in the training step, but still needs real-data and InternVL scoring completion to match the full target runtime contract.
