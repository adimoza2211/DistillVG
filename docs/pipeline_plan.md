# Pipeline Plan

## Purpose
This document is the authoritative pipeline plan for the online distillation setup.

The key design decision is: there is no persistent cache in the training loop.
Both the student and the frozen verifier reside in VRAM during training, and the verifier is executed online in the same step that produces the student proposals.

## Target Flow

For one training step:
1. The student model processes the input image and expression.
2. YOLO26-small produces noisy proposal boxes.
3. The original image is cropped at those proposal coordinates.
4. The frozen InternVL 3.5-8B verifier scores each crop against the expression.
5. The verifier logits are treated as supervision targets for the student’s plausibility head and/or proposal ranking logic.
6. The backward pass updates only the student parameters.

## Student Specification

The student is the trainable model that we are compressing and eventually exporting.

Current intended structure:
- Backbone: YOLO26-small style visual encoder, used to generate proposal boxes and visual tokens.
- Text branch: MobileCLIP-S1 text encoder, producing token embeddings for the grounding expression.
- Fusion block: a lightweight fusion transformer that mixes proposal-aligned visual tokens with text context.
- Box head: a regression head that predicts the final grounding box coordinates.
- Plausibility head: a verifier head that scores whether a proposal or final prediction matches the expression.

Implementation-level shape used in this repo:
- [src/models/student/model.py](../src/models/student/model.py) defines the composite student module.
- [src/models/student/backbone.py](../src/models/student/backbone.py) provides the visual token generator.
- [src/models/student/text_encoder.py](../src/models/student/text_encoder.py) provides the text token encoder.
- [src/models/student/fusion.py](../src/models/student/fusion.py) provides the token-mixing block.
- [src/models/student/verifier.py](../src/models/student/verifier.py) provides the student-side plausibility scorer.

Important constraints on the student:
- It is the only model updated by backpropagation.
- It must stay lean enough for the edge deployment target.
- It should be designed around the 30 ms latency budget, not around maximum parameter count.
- It is allowed to evolve as implementation matures, but the high-level components above should remain.

## Why There Is No Cache

The verifier is too large to train, but it is still small enough to run in VRAM when quantized.

Instead of writing verifier outputs to disk:
- keep the verifier frozen and loaded in 4-bit quantization
- keep the student trainable
- compute verifier scores live per batch
- use those scores immediately for loss computation

This avoids a stale snapshot problem where proposal boxes or pseudo-labels would diverge from the student that is actively learning.

## Phase Breakdown

### Phase 0: Deterministic Data Preparation
- Extract scene graphs offline.
- Keep the augmentation deterministic.
- Use the scene-graph-based phrase augmentation path only.
- Validate the augmented records before training.

### Phase 1: Online Verifier-Guided Distillation
- Load the frozen InternVL verifier in 4-bit mode.
- Load the YOLO26 student and the rest of the trainable modules.
- For each minibatch, generate proposals, crop the image, score crops with the verifier, and compute losses immediately.
- Do not serialize verifier outputs as a training artifact.
- Do not introduce HDF5 or other persistent pseudo-label caches.

Current code reality:
- The trainer still runs a synthetic grounding scaffold.
- The live online verifier pass is the target design, not yet the implemented training loop.
- The student modules already exist, but the wiring between student proposals and verifier scoring is still pending.

### Student-to-Verifier Supervision Contract
- The student produces candidate boxes and internal plausibility features.
- The verifier evaluates the crop-text pair immediately.
- The verifier logit is the online supervision signal, not a stored label.
- The student’s own plausibility head learns to approximate the verifier signal while still respecting ground-truth box losses.
- Only the student’s parameters participate in gradient updates.

### Phase 2: Student Optimization
- Update only trainable student parameters.
- Use the verifier signal together with ground-truth supervision and the distillation losses.
- Keep the verifier in `eval()` and `torch.no_grad()`.

### Phase 3: Compression and Export
- Prune the student where required.
- Apply quantization-aware training.
- Export the student to deployment targets.
- Never export the verifier.

## Explicit Non-Goals
- No teacher-feature cache.
- No verifier pseudo-label HDF5 shard pipeline.
- No offline precompute step for verifier outputs.
- No training of the verifier.
- No export artifact containing the verifier.

## Implementation Backlog
- Replace the synthetic trainer scaffold with real image crops and verifier scoring.
- Make the verifier loading path explicit and quantized.
- Keep the student loss code aligned with live verifier logits rather than stored records.
