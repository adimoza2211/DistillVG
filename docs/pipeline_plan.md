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
- Do not introduce HDF5 or other persistent pseudo-label cache

Current code reality:
- The trainer still runs a synthetic grounding scaffold.
- The live online verifier pass is the target design, not yet the implemented training loop.
- The student modules already exist, but the wiring between student proposals and verifier scoring is still pending.

### Student-to-Verifier Supervision Contract(VALOR)
- The student produces candidate boxes and internal plausibility features.
- The verifier evaluates the crop-text pair immediately.
- The verifier logit is the online supervision signal, not a stored label.
- The student’s own plausibility head learns to approximate the verifier signal while still respecting ground-truth box losses.
- Only the student’s parameters participate in gradient updates.

### Exact Implementation Details
Instead of mathematically unstable feature-level distillation from massive 200M+ parameter models, we adopt a verifier-based distillation approach inspired by the VALOR framework and Zero-Shot Box Verification techniques.3 We utilize the highly capable, memory-efficient InternVL 3.5-8B (https://huggingface.co/OpenGVLab/InternVL3_5-8B) model (quantised), which easily fits within our 30GB VRAM budget.
High-Recall Proposal Generation: The student backbone generates the top-200 candidate bounding boxes with intentionally lowered confidence thresholds to ensure maximum recall.
Box-Wise Cropping: The high-resolution image is cropped using these bounding box coordinates to isolate specific target regions.
VLM Verification Query: We feed these image crops alongside the Llama-generated paraphrases (including explicitly confusing hard-negative relational inversions) into the frozen InternVL 3.5-8B model. We prompt the VLM with a strict True/False query regarding whether the crop matches the text.
Logit Extraction: Instead of raw text output, we extract the continuous logit distribution of the "True" token, providing a highly refined, scalar plausibility score.5 This completely bypasses the need to cache high-dimensional tensors and inherently eliminates cross-box interference.

The student is designed to maximise accuracy within the latency constraint:
Visual backbone: YOLO26-small (frozen, $\approx9.5M$ parameters). YOLO26 features a native end-to-end NMS-free inference architecture, guaranteeing strict, deterministic adherence to our sub-30 ms latency budget by eliminating post-processing bottlenecks.6 We extract the top-200 proposals using RoI-Align with $7\times7$ crops, projected to $d=384$.
Text encoder: MobileCLIP-S1 text branch (frozen for epochs 1-5, then fine-tuned), projecting to $d=384$.
Fusion transformer: 4 layers, $d=384$, 6 heads ($\approx7M$ parameters), with Token Blurring (ToB) merging applied after layer 2 to reduce active visual tokens from 200 to ~150.
Re-ranking verifier head: A lightweight cross-attention module ($\approx0.4M$ parameters).


### Implement Dynamic Weight-Balance Distillation (DWBD)
 During your training phase, use the DWBD technique introduced in the SimVG architecture. Because there is a massive parameter gap between your 7M parameter fusion transformer and your 0.4M parameter verifier head, DWBD dynamically adjusts the distillation weights across epochs to prevent the smaller student head from underfitting the teacher's complex features early on.

### Training Objective & Optimization
 We optimize the network using the MuSGD optimizer, a hybrid of SGD and Muon, which ensures highly stable and rapid convergence within our strict 1-month timeframe. The loss function utilizes Dynamic Weight-Balance Distillation (DWBD) from the SimVG framework. Because there is a massive capacity mismatch between the 7M parameter fusion transformer and the 0.4M parameter verifier head, DWBD actively adjusts the weights assigned to the VLM teacher's logits and the ground-truth labels across epochs, preventing the smaller branch from underfitting early on. Finally, we integrate YOLO26's Small-Target-Aware Label Assignment (STAL) and Progressive Loss Balancing (ProgLoss) to handle small, dense referents effectively.

### Phase 2: Student Optimization
- Update only trainable student parameters.
- Use the verifier signal together with ground-truth supervision and the distillation losses.
- Keep the verifier in `eval()` and `torch.no_grad()`.

### Phase 3: Compression and Export
- Prune the student where required.
- Apply quantization-aware training.
- Export the student to deployment targets.
- Never export the verifier.

  We apply structured pruning of the fusion transformer (20% FFN channel pruning), followed by 5 epochs of mixed-precision INT8 Quantization-Aware Training (QAT). Crucially, YOLO26 removes the Distribution Focal Loss (DFL) module, making its regression pipeline highly hardware-friendly. This ensures that our export via ONNX to TensorRT, CoreML, and Android NNAPI suffers minimal quantization degradation, maintaining high fidelity across FP16 and INT8 precincts.
  Key Technical Novelties
  NMS-Free Edge-Optimized Backbone: Utilizing YOLO26-small completely eliminates the Non-Maximum Suppression bottleneck, providing deterministic, constant-time inference critical for the <30 ms mobile NPU budget.
  Verifier-Based Logit Distillation: Bypassing computationally heavy feature distillation by utilizing InternVL 3.5-8B as a zero-shot box verifier, transforming complex spatial alignment into a clean, scalar logit reward signal that fits into a 30GB VRAM footprint.
  Dynamic Weight-Balance Distillation (DWBD): Implementing dynamic loss weighting to bridge the massive capacity gap

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
