# DistillVG Pipeline Plan (Phase Finalize)

*Note: This document reflects the authoritative structure implemented in the DistillVG repository. All mock components, testing suites, and fallbacks have been eliminated based on the latest directives.*

## System Architecture

The overarching goal is to achieve sub-30ms mobile inference (Apple Neural Engine / CoreML / ONNX) while retaining state-of-the-art visual grounding from massive VLM teachers. The system structure is strictly divided into Student pathways and Teacher (Verifier) pathways during Phase 1 training, decoupling into a pure Student topology for later stages.

### 1. The Student Topology
The student model must be highly compact but conceptually rich:
- **Visual Encoder:** Ultralytics YOLO26 frozen backbone. The backbone runs inference on the source image, emitting hierarchical feature maps (FPN levels P3, P4, P5). Simultaneously, the YOLO head generates dense region proposals. We use `torchvision.ops.roi_align` to construct fixed-size localized visual tokens (`roi_tokens`) for every candidate bounding box from these frozen features.
- **Language Encoder:** Apple's MobileCLIP-S1. The input string is passed through a BPE tokenizer (sequence limit: 77 tokens) and then encoded by the MobileCLIP Text-Transformer. The unprojected contextual sequence (`[B, 77, 512]`) is extracted. 
- **Projection Layers:** Both the pooled RoI visual tokens and the text sequence tokens are fed through trainable $d_{orig} \rightarrow d_{hidden}$ linear projections to align representation spaces.
- **Fusion Transformer:** The projected text-visual sequences are concatenated. A computationally tight 6-layer, 8-head Transformer Encoder cross-attends the modes. 
- **Detection Heads:** Specialized shallow MLPs translate the sequence items back into regression outputs (tight bbox coordinates) and classification outputs (verifier agreement probability).

### 2. The Teacher Verifier Topology
The teacher acts strictly as an online or offline target-generator.
- **InternVL Backend**: Uses the `InternVLOnlineVerifier`.
- **Two-Stage Prompting**: 
    1. **Coarse Crop Evaluation**: Crops each YOLO proposal and compares standard image-text alignment.
    2. **Fine Highlight Evaluation**: Takes the top-K proposals, draws a structural "red rectangle" over the source image, and prompts the VLM with a 4-class categorical structure to decisively rank the region's semantic relevance to the raw query.

## Data Processing
- **Text:** Direct runtime execution of the CLIP BPE Tokenizer (`_tokenize_fn`), avoiding all hash-based embedding tables and fast-vocabulary fallbacks.
- **Visuals:** Standard contiguous Torch tensors normalized and passed asynchronously via `AugmentedGroundingDataset`.
- **Synthetics / Mocks:** Obliterated. The pipeline requires real or precomputed COCO/RefCOCO data structures. 

## Training Epoch Schema

### Phase 1: Distillation
During Stage 1, the Teacher's knowledge is actively queried (or loaded via `precomputed_cache_path`). The Student evaluates the loss surfaces strictly via KD distributions (DWBD, Progloss, Kl-Divergence, Consistency maps). The base backbone and text-transformer are generally frozen, heavily punishing only the fusion and projection blocks.

### Phase 2: Strict
The Verifier is purged from VM allocations. Ground Truth regression logic assumes absolute authority. Remaining un-frozen components align firmly to localized detection benchmarks.

## Execution
*Refer to the `README.md` for standard operation. As a standing rule, operations mandate active execution contexts via `conda activate distill`.*
