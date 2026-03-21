# Copilot Instructions

## Purpose
This file is the agent-visible coding instruction surface for DistillVG.

The target system is online distillation. Do not introduce persistent verifier caches or teacher-feature dumps.

## Canonical Runtime Contract

- The student and frozen verifier are both resident in VRAM during training.
- The student produces proposal boxes first.
- The verifier scores the cropped proposals immediately in the same step.
- Losses use verifier logits as online supervision.
- Backpropagation updates only the student.
- The verifier is always frozen, always in `eval()`, and always under `torch.no_grad()`.
- The verifier should be loaded in 4-bit quantization to fit the VRAM budget.

## Naming Rules

- Call the frozen vision-language model the verifier.
- Do not call it a teacher in new code or new docs.
- Do not use cache-first language for the training path.
- If a legacy script or module still uses old naming, treat that as backlog to remove.

## Code Generation Rules

- Use type hints everywhere.
- Preserve the Hydra configuration style already used in the repo.
- Keep training logic in [src/training/](../src/training/).
- Keep model components split by responsibility under [src/models/](../src/models/).
- Keep script entrypoints in [scripts/](../scripts/).
- Add or update unit tests for every new module or changed behavior.
- For running stuff in cli, always do conda activate distill first
- Use modern pytorch features like Amp or any other modern deep learning concepts that will decrease inference or training time WITHOUT any sacrifice to performance
- As you start solidifying components, keep removing fallbacks to ensure correctness. 

## Architecture Constraints

- Do not add HDF5 pseudo-label caches.
- Do not add disk-backed verifier features.
- Do not train the verifier.
- Do not export the verifier.
- Keep the student lean enough for the edge deployment target.
- Keep the verifier loading path explicit, quantized, and frozen.

## Preferred Implementation Shape

- Student backbone: YOLO26-small style visual encoder that produces proposal-aligned tokens for the current image.
- Text encoder: MobileCLIP-S1 style encoder for the grounding expression.
- Student fusion: lightweight token-mixing block that combines visual tokens with text context and feeds both box regression and plausibility scoring.
- Student heads: one bbox regression head plus one plausibility/verifier head in the student model.
- Verifier: InternVL 3.5-8B, frozen, 4-bit.
- Training loop: online crop generation, online verifier scoring, immediate loss computation.
- Compression: structured pruning, QAT, export only for the student.

## Current Implementation Reality

- The trainer is still a synthetic scaffold.
- The online verifier execution path is the target, not yet the live code path.
- New code should move the trainer and student toward the runtime contract above without reintroducing cache-oriented plumbing.

## Documentation Rules

- Keep the repo documented with only three categories:
  - current state in [README.md](../README.md)
  - pipeline plan in [docs/pipeline_plan.md](../docs/pipeline_plan.md)
  - coding instructions in this file
- If the architecture changes, update those three files first.
- Do not create new markdown docs unless the user explicitly asks for them.

## Practical Editing Rules

- Prefer minimal, local changes.
- Remove stale cache terminology when you touch nearby code.
- Keep the current-state README honest about legacy scaffolding that still exists.
- If you implement the online verifier path, update tests and the README in the same change.
