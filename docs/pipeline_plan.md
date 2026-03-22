```markdown
# Pipeline Plan — Lightweight Visual Grounding via Progressive Knowledge Distillation

## Document Status
Authoritative implementation reference. All architecture, data, training, and export decisions
are documented here. This file is the primary agentic context for Copilot-assisted development.
When implementation details conflict with any other file, this document takes precedence.

---

## Big Picture

### What This System Does
Given an image `I` and a free-form natural-language expression `E`, the system predicts a
bounding box `b̂ ∈ ℝ⁴` (x_center, y_center, w, h in COCO normalized format) that localizes
the referent of `E` in `I`. Performance is evaluated by Accuracy@0.5 (fraction of predictions
with IoU ≥ 0.5 against ground truth).

### Core Design Philosophy
A large frozen VLM (InternVL 3.5-8B, 4-bit quantized) acts as an online teacher. A compact
student (~33–36M params) learns to replicate the teacher's spatial reasoning under a strict
<30 ms end-to-end latency budget on mobile NPUs. There is no offline pseudo-label cache.
All teacher signals are computed live per batch.

### System Components at a Glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TRAINING TIME ONLY                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Frozen InternVL 3.5-8B (4-bit, eval mode, torch.no_grad())     │   │
│  │  ├─ Stage 1: scores top-200 crops   → scalar True logit         │   │
│  │  └─ Stage 2: scores top-5 full-img  → 4-class soft distribution │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                           │  online supervision signal (no cache)       │
│  ┌────────────────────────▼────────────────────────────────────────┐   │
│  │                    STUDENT (trainable)                           │   │
│  │  YOLO26-small backbone  →  top-200 proposals + FPN tokens       │   │
│  │  MobileCLIP-S1 text     →  text token embeddings                │   │
│  │  Fusion Transformer     →  cross-modal token mixing             │   │
│  │  Box Head               →  final box regression                 │   │
│  │  Verifier Head (4-cls)  →  [Excellent, Good, Partial, No] dist  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                           │
                    (Phase 3 export)
                           │
┌──────────────────────────▼──────────────────────────────────────────────┐
│            DEPLOYMENT ARTIFACT (student only, INT8, ~30 MB)             │
│  ONNX → TensorRT (Jetson) / CoreML (Apple Silicon) / ONNX RT (Android)  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Phases

| Phase | Name | What Happens |
|-------|------|--------------|
| 0 | Deterministic Data Prep | Scene graph extraction, phrase augmentation, JSON serialization |
| 1 | Online Verifier-Guided Distillation | Live two-stage teacher scoring, DWBD + KL loss training |
| 2 | Student Fine-Tuning | Ground-truth-only refinement with frozen verifier head |
| 3 | Compression & Export | Structured pruning → QAT → ONNX/TensorRT/CoreML export |

### Key Design Invariants (never violate these)
- **No persistent verifier cache.** Verifier outputs are never written to disk as training
  artifacts. No HDF5, no `.pt` shard, no npy dump of pseudo-labels.
- **Verifier is always frozen.** `internvl.eval()` + `torch.no_grad()` for all forward passes.
- **Only student parameters receive gradients.**
- **Latency is the hard constraint.** Architecture decisions are made to satisfy <30 ms
  end-to-end on Snapdragon 8 Gen 3 / Apple A17 / Jetson Orin. Parameter count is secondary.
- **Verifier is never exported.** The deployment artifact contains the student only.

---

## Implementation Status Snapshot (Done vs Pending)

Status legend:
- ✅ Present in repo
- 🟡 Partial / scaffolded
- ❌ Not implemented yet

### Phase 0
- ✅ Deterministic `_aug.pth` records are consumed by training dataset code.
- ✅ Record schema used by the training loader is documented below (tuple contract + consumed fields).

### Phase 1 (Core Runtime Contract)
- ✅ Frozen verifier loaded in 4-bit and run under no-grad/eval in training path.
- ✅ Online crop scoring in-step (no persistent verifier cache).
- ✅ Proposal-level top-K crop scoring is wired.
- ✅ Real image-backed `_aug.pth` training records are used (synthetic scaffold removed from active path).
- ✅ YOLO26-backed proposal generator path is present.
- ✅ Two-stage verifier utilities are implemented (Stage-1 crop scoring + Stage-2 highlighted full-image 4-class scoring path).
- ✅ Verifier head exposes 4-class logits and ordinal scoring utility.
- ✅ Planned `src/losses/*` module split (`verifier_kl.py`, `dwbd.py`, `consistency.py`, `box_losses.py`, `combined.py`) is now in place.

### Phase 2
- ✅ DWBD scheduling utilities and MuSGD optimizer support are introduced.
- ✅ Full Phase-2 recipe from this doc is wired as a standalone mode (`train.mode=phase2_strict`) with verifier unloaded and `L_box + L_cst` only.

### Phase 3
- 🟡 Compression/export scaffolds exist, but this exact plan’s pruning/QAT/export sequence is not fully implemented and validated end-to-end.
---

## Repository Layout

```


---

## Phase 0: Deterministic Data Preparation

### Purpose
Build a clean, deterministic augmented dataset before any training begins. All randomness
is seeded and fixed. Phase 1 reads from the augmented records; it never re-runs augmentation.

### Inputs
- Raw annotation JSONs for RefCOCO, RefCOCO+, RefCOCOg, ReferItGame (COCO format).
- Images from COCO 2014 train/val splits + ReferItGame image pool.

### Outputs
- Per-split augmented record files: `data/annotation/{split}_aug.pth`
- Each record is a tuple-like payload consumed from `*_aug.pth`:
  - `item[0]`: `image_name` (`str`) — relative filename of the image.
  - `item[1]`: auxiliary graph/reference payload (ignored by training loader).
  - `item[2]`: `box_xywh` (`List[float]`, len=4) — absolute image-space box.
  - `item[3]`: `phrases` (`List[str]`) — original expression + paraphrases.
  - `item[4+]`: optional augmentation metadata (ignored by active training loader).
- Normalization and filtering contract in active loader:
  - Empty/whitespace phrases are dropped.
  - Records missing valid `image_name`, `box_xywh`, or phrase list are skipped.
  - Loaded records are converted to internal `GroundingRecord(image_name, box_xywh, phrases, source)`.


## Phase 1: Online Verifier-Guided Distillation

### Overview
This is the main training phase. The frozen InternVL verifier provides live supervision
signals; the student learns to produce accurate proposals and rank them correctly.

---

### Component: Dataset and DataLoader (`src/data/dataset.py`)

**Class:** `GroundingDataset(torch.utils.data.Dataset)`


**`collate_fn`:**
- Variable-resolution images are padded to the batch's max H, W with `pad_value=0`.
- `image_raw` is kept at native resolution for cropping accuracy (do not normalize this).
- `all_queries` is a list-of-lists; pad to the max number of queries per sample in the batch.

**Sampling:**
- Concatenate all four datasets into a single `ConcatDataset`.
- Use `WeightedRandomSampler` with weights inversely proportional to dataset size to
  prevent RefCOCO from dominating (RefCOCOg and ReferItGame have fewer samples).

---

### Component: YOLO26-Small Backbone (`src/models/student/backbone.py`)

**Class:** `YOLO26SmallBackbone(nn.Module)`

**Purpose:** Produces multi-scale FPN feature maps and top-K class-agnostic proposals.

**Key properties:**
- ~9.5M parameters, **always frozen** during all training phases.
- NMS-free by design (uses TOOD/TaskAligned head without DFL, no post-processing).
- Outputs FPN feature maps at strides 8, 16, 32 (P3, P4, P5).
- Proposals are class-agnostic; we intentionally lower the objectness threshold to 0.01
  to generate high-recall top-200 candidates.

**Forward signature:**
```python
def forward(self, images: torch.Tensor) -> BackboneOutput:
    # images: [B, 3, H, W]
    # returns:
    #   .proposals:    [B, 200, 4]  # normalized cx,cy,w,h, sorted by objectness score desc
    #   .proposal_scores: [B, 200]  # raw objectness logits (pre-sigmoid)
    #   .fpn_features: Dict[str, Tensor]  # {"p3": [B,C,H/8,W/8], "p4": ..., "p5": ...}
```

**RoI-Align for proposal tokens:**
- Apply `torchvision.ops.roi_align` with `output_size=(7,7)` on P3/P4/P5 features.
- For each proposal, select the FPN level by proposal area (P3: area<32², P4: 32²–96², P5: >96²).
- Flatten 7×7 spatial grid → 49 tokens per proposal, then apply a linear projection to d=384.
- After projection: `roi_tokens: [B, 200, 384]` (one pooled token per proposal from the
  mean-pooled 7×7 grid; do NOT keep 49 tokens per proposal — this would be 9800 tokens/sample).

**Implementation note:**
- Load checkpoint from official YOLO26-small weights.
- Wrap in `backbone.eval()` + register `backbone.requires_grad_(False)` at model init.
- Do not apply BN running-stat updates (already handled by `eval()` mode).

---

### Component: MobileCLIP-S1 Text Encoder (`src/models/student/text_encoder.py`)

**Class:** `MobileCLIPS1TextEncoder(nn.Module)`

**Purpose:** Encodes the grounding expression into a sequence of token embeddings and a pooled
sentence embedding, both at d=384.

**Key properties:**
- ~15M parameters.
- **Frozen for epochs 1–5.** From epoch 6 onward, fine-tuned at `0.1 × base_lr`.
- Input: tokenized expression (BPE, max 77 tokens, CLIP tokenizer).
- Output: `(token_embeddings: [B, 77, 384], pooled: [B, 384])`.

**Freezing schedule implementation (`src/training/trainer.py`):**
```python
def on_epoch_start(self, epoch: int):
    if epoch < 6:
        self.student.text_encoder.requires_grad_(False)
    else:
        self.student.text_encoder.requires_grad_(True)
        # ensure LR group for text_encoder uses 0.1x scale
        for pg in self.optimizer.param_groups:
            if pg["name"] == "text_encoder":
                pg["lr"] = self.base_lr * 0.1
```

**Projection:**
- If MobileCLIP-S1 native dim ≠ 384, add a `nn.Linear(native_dim, 384)` projection layer
  (this projection IS trainable from epoch 1).

---

### Component: Fusion Transformer (`src/models/student/fusion.py`)

**Class:** `FusionTransformer(nn.Module)`

**Architecture:**
- 4 transformer layers, each with:
  - Cross-attention: visual tokens attend to text tokens (Q=visual, K=V=text)
  - Self-attention on visual tokens
  - FFN with hidden_dim = 4 × 384 = 1536
  - Pre-LN normalization
- d_model = 384, n_heads = 6, head_dim = 64
- ~7M parameters, **fully trainable**.

**Token Blurring (ToB) — applied after Layer 2:**
- Implements similarity-weighted token merging (NOT dropping) from Token Merging (ToMe).
- After layer 2, merge visually similar proposal tokens from 200 → ~150:
  ```python
  # After layer 2 forward pass:
  # proposal_tokens: [B, 200, 384]
  r = 50  # tokens to merge (200 → 150)
  merged_tokens, merge_map = token_blur(proposal_tokens, r=r, mode="mean")
  # merged_tokens: [B, 150, 384]
  # merge_map: [B, 200] → indices into merged_tokens (for spatial recovery at box head)
  ```
- **Critical:** the `merge_map` must be stored and passed to the box head to recover
  which merged token corresponds to which original proposal box. Merging uses cosine
  similarity; do NOT use random merging.
- Spatial grounding note: do NOT drop tokens. Merged tokens carry a weighted mean of
  the spatial positions of their constituents.

**Forward signature:**
```python
def forward(
    self,
    roi_tokens: torch.Tensor,      # [B, 200, 384]
    text_tokens: torch.Tensor,     # [B, 77, 384]
    text_pooled: torch.Tensor,     # [B, 384]
) -> FusionOutput:
    # returns:
    #   .visual_tokens:  [B, 150, 384]  # post-ToB merged tokens
    #   .merge_map:      [B, 200]       # original→merged index mapping
    #   .layer_attns:    List[[B,6,150,77]]  # per-layer cross-attn (for L_cst)
```

---

### Component: Verifier Head (`src/models/student/verifier.py`)

**Class:** `VerifierHead(nn.Module)`

**Purpose:** Scores each proposal token against the expression; outputs a 4-class soft
distribution `[Excellent, Good, Partial, No]` per proposal.

**Architecture:**
- Cross-attention: proposal tokens (Q) attend to pooled text embedding (K=V broadcasted).
- 2-layer MLP on the cross-attention output.
- Output: `[B, K, 4]` logits → softmax → 4-class distribution.
- ~0.4M parameters, fully trainable.

**Forward signature:**
```python
def forward(
    self,
    visual_tokens: torch.Tensor,   # [B, K, 384]  (K ≤ 150 post-ToB, or top-k subset)
    text_pooled: torch.Tensor,     # [B, 384]
) -> torch.Tensor:                 # [B, K, 4]  — raw logits (apply softmax for probabilities)
```

**At inference:** the final box selection score is the ordinal expectation:
```python
probs = softmax(logits, dim=-1)           # [B, K, 4]
ordinal_weights = tensor([3., 2., 1., 0.])
score = (probs * ordinal_weights).sum(-1) # [B, K]  ∈ [dl.acm](https://dl.acm.org/doi/abs/10.1109/TCSVT.2024.3407785)
```
Select the proposal with the highest score as the final prediction.

**At training:** supervised by KL divergence against Stage-2 InternVL 4-class soft labels
(see `src/losses/verifier_kl.py`).

---

### Component: Box Regression Head (`src/models/student/box_head.py`)

**Class:** `BoxRegressionHead(nn.Module)`

**Purpose:** Predicts refined box deltas for each proposal; produces final [cx, cy, w, h].

**Architecture:**
- 3-layer MLP: `384 → 256 → 128 → 4`
- Applied independently to each proposal token.
- Outputs YOLO-style deltas `(dx, dy, dw, dh)` relative to the proposal box.
- Apply `sigmoid(dx), sigmoid(dy)` for center offsets; `exp(dw), exp(dh)` for scale.

**Proposal-to-token mapping:**
- Box head operates on the TOP-K proposals after verifier head scoring (K=5 during
  the Stage-2 re-scoring phase; all 150 during loss computation).
- Use `merge_map` from FusionTransformer to recover original proposal coordinates
  for merged tokens (weighted mean of constituent proposal boxes).

---

### Component: Composite Student Model (`src/models/student/model.py`)

**Class:** `StudentGrounder(nn.Module)`

**Full forward pass:**
```python
def forward(
    self,
    images: torch.Tensor,           # [B, 3, H, W]
    queries: List[str],             # length B
) -> StudentOutput:
    # 1. Text encoding
    text_tokens, text_pooled = self.text_encoder(queries)   # [B,77,384], [B,384]

    # 2. Visual proposals + FPN tokens
    backbone_out = self.backbone(images)
    # backbone_out.proposals: [B, 200, 4]
    # backbone_out.roi_tokens: [B, 200, 384]

    # 3. Fusion
    fusion_out = self.fusion(
        backbone_out.roi_tokens, text_tokens, text_pooled
    )
    # fusion_out.visual_tokens: [B, 150, 384]
    # fusion_out.merge_map: [B, 200]
    # fusion_out.layer_attns: List of [B,6,150,77]

    # 4. Verifier head (all 150 tokens, for loss)
    verifier_logits = self.verifier_head(
        fusion_out.visual_tokens, text_pooled
    )  # [B, 150, 4]

    # 5. Box regression (all 150 tokens, for loss)
    box_deltas = self.box_head(fusion_out.visual_tokens)  # [B, 150, 4]

    return StudentOutput(
        proposals=backbone_out.proposals,           # [B, 200, 4]
        proposal_scores=backbone_out.proposal_scores,
        merge_map=fusion_out.merge_map,             # [B, 200]
        visual_tokens=fusion_out.visual_tokens,     # [B, 150, 384]
        layer_attns=fusion_out.layer_attns,
        verifier_logits=verifier_logits,            # [B, 150, 4]
        box_deltas=box_deltas,                      # [B, 150, 4]
        text_pooled=text_pooled,                    # [B, 384]
    )
```

---

### Component: Two-Stage InternVL Verifier (`src/models/teacher/internvl_verifier.py`)

**Class:** `TwoStageVerifier(nn.Module)` — **always in `eval()`, always in `torch.no_grad()`**

**Initialization:**
```python
model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL3_5-8B",
    torch_dtype=torch.float16,
    load_in_4bit=True,              # bitsandbytes 4-bit quantization
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
```

**VRAM budget:** InternVL 3.5-8B in 4-bit ≈ 5–6 GB. Student + optimizer states ≈ 20–22 GB.
Total ≈ 26–28 GB, within 30 GB budget.

---

#### Stage 1: Broad Crop Scoring (all 200 proposals)

**Input per sample:** image `I` [H, W, 3 uint8], proposals `boxes` [200, 4] (abs xyxy),
expression `E` (str).

**Crop procedure:**
```python
def crop_proposals(image_raw: np.ndarray, boxes_xyxy: np.ndarray) -> List[PIL.Image]:
    H, W = image_raw.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy:
        # clamp to image bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        crop = image_raw[y1:y2, x1:x2]
        crops.append(PIL.Image.fromarray(crop))
    return crops
```

**Prompt template (Stage 1):**
```
Does the highlighted region in this image match the expression "{E}"?
Answer with a single word: True or False.
```

**Logit extraction:**
- Run a single forward pass with the crop as the image input.
- Extract the raw logit of the `True` token from the final vocabulary distribution
  (before softmax): `true_logit = lm_logits[0, -1, true_token_id]`.
- Do NOT sample or decode; operate purely on logits.
- Return: `stage1_scores: [B, 200]` (float32, raw True logits).

**Batching for efficiency:**
- Batch crops from all samples in the minibatch: total 200×B crops.
- Process in sub-batches of 64 crops to avoid OOM.
- Store results indexed by `(sample_idx, proposal_idx)`.

---

#### Stage 2: Full-Image Contextual Re-Scoring (top-5 proposals)

**Trigger:** After Stage 1 scores are computed, select the top-5 proposals per sample
by `stage1_scores`.

**Full-image highlighting procedure:**
```python
def draw_highlighted_box(
    image_raw: np.ndarray,
    box_xyxy: np.ndarray,
    color: Tuple[int,int,int] = (255, 0, 0),
    thickness: int = 3,
) -> PIL.Image:
    img = PIL.Image.fromarray(image_raw.copy())
    draw = PIL.ImageDraw.Draw(img)
    x1, y1, x2, y2 = box_xyxy.astype(int)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    return img
```

- For each of the top-5 proposals, the **full original image** is passed with the box
  drawn as a colored rectangle. The image is NOT cropped.
- This preserves global spatial context needed to evaluate relational expressions.

**Prompt template (Stage 2):**
```
The red rectangle in this image highlights a specific region.
How well does the highlighted region match the expression "{E}"?
Answer with exactly one word from: Excellent, Good, Partial, No.
```

**4-class logit extraction:**
```python
ANSWER_TOKENS = ["Excellent", "Good", "Partial", "No"]
answer_token_ids = [tokenizer.encode(t, add_special_tokens=False) for t in ANSWER_TOKENS]

# For each Stage-2 query:
lm_logits = model(...).logits[0, -1, :]        # [vocab_size]
answer_logits = lm_logits[answer_token_ids]     # [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_MoVE-KD_Knowledge_Distillation_for_VLMs_with_Mixture_of_Visual_Encoders_CVPR_2025_paper.html)
soft_labels = F.softmax(answer_logits, dim=-1)  #  — this is the pseudo-label [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_MoVE-KD_Knowledge_Distillation_for_VLMs_with_Mixture_of_Visual_Encoders_CVPR_2025_paper.html)
```

**Output:** `stage2_soft_labels: [B, 5, 4]` — one 4-class distribution per top-5 proposal
per sample.

**Mapping back to 150-token space:**
- The top-5 proposals from Stage 1 are identified by original proposal indices `[B, 5]`.
- Use `merge_map` to find which merged token each Stage-2-scored proposal corresponds to.
- Construct `verifier_targets: [B, 150, 4]` where:
  - The 5 matched merged-token positions carry their Stage-2 soft labels.
  - All other positions carry the uniform distribution `[0.25, 0.25, 0.25, 0.25]` (no signal).
  - The 195 non-top-5 positions carry a pseudo-label derived from Stage-1 score:
    convert scalar `true_logit` to a 4-class distribution via a calibration mapping
    (see `src/models/teacher/internvl_verifier.py::scalar_to_soft_label()`).

**`scalar_to_soft_label` calibration:**
```python
def scalar_to_soft_label(true_logit: float) -> np.ndarray:
    # Maps scalar True logit to a 4-class distribution
    # Thresholds tuned on RefCOCO val; adjust after initial runs
    p = sigmoid(true_logit)
    if p > 0.85:   return [0.7, 0.2, 0.08, 0.02]   # Excellent-biased
    elif p > 0.6:  return [0.1, 0.6, 0.25, 0.05]   # Good-biased
    elif p > 0.35: return [0.02, 0.15, 0.6, 0.23]  # Partial-biased
    else:          return [0.01, 0.04, 0.15, 0.80]  # No-biased
```

---

### Component: Loss Functions

#### `src/losses/verifier_kl.py` — `L_ver`

**KL Divergence over 4-class soft labels:**
```python
def verifier_kl_loss(
    student_logits: torch.Tensor,    # [B, K, 4]
    teacher_soft_labels: torch.Tensor, # [B, K, 4]
    mask: torch.Tensor,              # [B, K] bool — True where teacher label is meaningful
    temperature: float = 2.0,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = teacher_soft_labels.clamp(min=1e-8)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)  # [B, K]
    return (kl * mask.float()).sum() / mask.float().sum()
```

- Temperature `τ=2.0` softens both distributions; tune as a hyperparameter.
- `mask` is True for Stage-2 positions (top-5) and positions where Stage-1 score is
  sufficiently high (|true_logit| > 0.5); False for uninformative near-zero logits.

#### `src/losses/dwbd.py` — `L_DWBD`

**Dynamic Weight-Balance Distillation** (from SimVG):
- Maintains a dynamic scalar `α(t)` that weights the teacher logit component vs.
  ground-truth label component across training epochs.
- `α(t) = α_max × (1 - t/T_max)^γ` where `t` is current epoch, `T_max` is total epochs,
  `γ=2.0` by default.
- At early epochs: high `α` → student relies heavily on teacher distribution.
- At late epochs: low `α` → student relies more on hard ground-truth labels.

```python
class DWBDLoss(nn.Module):
    def forward(
        self,
        student_logits: torch.Tensor,   # [B, K, 4] student verifier head logits
        teacher_soft: torch.Tensor,     # [B, K, 4] teacher 4-class labels
        gt_one_hot: torch.Tensor,       # [B, K, 4] hard GT (Excellent if IoU>0.7, No otherwise)
        alpha: float,                   # current epoch's α(t)
    ) -> torch.Tensor:
        blended_target = alpha * teacher_soft + (1 - alpha) * gt_one_hot
        return F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            blended_target.clamp(min=1e-8),
            reduction="batchmean"
        )
```

**GT one-hot construction:**
- For each proposal, compute IoU against ground-truth box.
- IoU ≥ 0.7 → class 0 (Excellent); IoU ∈ [0.5, 0.7) → class 1 (Good);
  IoU ∈ [0.3, 0.5) → class 2 (Partial); IoU < 0.3 → class 3 (No).

#### `src/losses/consistency.py` — `L_cst`

**Augmented Query Consistency Loss:**
- For a sample with N augmented queries (including original), run N forward passes through
  the text encoder and fusion transformer (student side only; verifier NOT called again).
- Extract cross-attention maps `A_i ∈ [B, 6, 150, 77]` for each query i from the fusion
  transformer's `layer_attns`.
- Compute pairwise cosine distance between cross-attention maps of different queries:
  ```python
  L_cst = mean over (i,j) pairs: 1 - cosine_sim(flatten(A_i), flatten(A_j))
  ```
- This enforces that the fusion transformer's spatial attention pattern is invariant
  to surface-form query variations, since all augmented queries ground to the same box.
- **Do NOT run the verifier for L_cst computation.** Only the student's attention maps.

#### `src/losses/box_losses.py` — Box Regression

```python
def box_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    # pred_boxes: [B, 4] — final predicted box (top-1 after verifier scoring)
    # gt_boxes:   [B, 4] — ground truth, normalized cx,cy,w,h
    l1 = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")
    giou = generalized_box_iou_loss(pred_boxes, gt_boxes)  # torchvision.ops
    return l1 + giou
```

#### `src/losses/combined.py` — `GroundingLoss`

**Total loss:**
```
L = L_DWBD + λ₁·L_cst + λ₂·L_ver_kl + λ₃·L_box
```

**Default λ values (tune via ablation):**
- `λ₁ = 0.1` (consistency)
- `λ₂ = 1.0` (verifier KL — primary distillation signal)
- `λ₃ = 5.0` (box regression — primary task loss)

---

### Component: Optimizer (`src/training/optimizer.py`)

**MuSGD** — hybrid of SGD and Muon (Nesterov momentum with Orthogonal update projection):

```python
param_groups = [
    {"name": "backbone",      "params": backbone_params,      "lr": 0.0,      "weight_decay": 0.0},
    {"name": "text_encoder",  "params": text_enc_params,      "lr": 0.0,      "weight_decay": 0.01},
    {"name": "fusion",        "params": fusion_params,        "lr": base_lr,  "weight_decay": 0.05},
    {"name": "verifier_head", "params": verifier_head_params, "lr": base_lr,  "weight_decay": 0.05},
    {"name": "box_head",      "params": box_head_params,      "lr": base_lr,  "weight_decay": 0.05},
    {"name": "projection",    "params": projection_params,    "lr": base_lr,  "weight_decay": 0.01},
]
# base_lr = 3e-4
# Backbone LR = 0 (frozen). Text encoder unfreezes at epoch 6 at 0.1 × base_lr.
```

**LR Schedule (`src/training/scheduler.py`):**
- Warmup: linear from `base_lr × 0.1` to `base_lr` over first 500 steps.
- Cosine decay from `base_lr` to `base_lr × 0.01` over remaining steps.

---

### Component: Training Loop (`src/training/trainer.py`)

**One training step — complete pseudocode:**
```python
def training_step(self, batch: Dict) -> torch.Tensor:
    images      = batch["image"].cuda()         # [B, 3, H, W]
    images_raw  = batch["image_raw"]            # [B, H_orig, W_orig, 3] uint8, CPU
    queries     = batch["query"]                # List[str], length B
    all_queries = batch["all_queries"]          # List[List[str]]
    gt_boxes    = batch["bbox_gt"].cuda()       # [B, 4]

    # ── 1. Student forward (main query) ──────────────────────────────────
    student_out = self.student(images, queries)
    # student_out.proposals: [B, 200, 4]
    # student_out.verifier_logits: [B, 150, 4]
    # student_out.box_deltas: [B, 150, 4]
    # student_out.merge_map: [B, 200]
    # student_out.layer_attns: List[[B,6,150,77]]

    # ── 2. Decode proposals → final box per merged token ─────────────────
    decoded_boxes = decode_boxes(student_out.proposals,
                                 student_out.box_deltas,
                                 student_out.merge_map)   # [B, 150, 4]

    # ── 3. Two-stage verifier scoring (teacher, no_grad) ─────────────────
    with torch.no_grad():
        # Stage 1: crop all 200 proposals
        boxes_xyxy = cxcywh_to_xyxy(student_out.proposals, images.shape)  # [B, 200, 4] abs
        stage1_scores = self.verifier.stage1_score_crops(
            images_raw, boxes_xyxy, queries
        )  # [B, 200]

        # Stage 2: re-score top-5 with full image + highlighted box
        top5_indices = stage1_scores.topk(5, dim=-1).indices  # [B, 5]
        stage2_soft_labels = self.verifier.stage2_score_fullimage(
            images_raw, boxes_xyxy, top5_indices, queries
        )  # [B, 5, 4]

        # Build verifier targets for all 150 merged tokens
        verifier_targets, verifier_mask = build_verifier_targets(
            stage1_scores, stage2_soft_labels,
            top5_indices, student_out.merge_map,
            decoded_boxes, gt_boxes
        )  # [B, 150, 4], [B, 150]

        # Build GT one-hot for DWBD
        proposal_ious = box_iou(decoded_boxes.view(-1,4),
                                gt_boxes.unsqueeze(1).expand_as(decoded_boxes).view(-1,4))
        proposal_ious = proposal_ious.view(B, 150)
        gt_one_hot = iou_to_4class_onehot(proposal_ious)  # [B, 150, 4]

        # DWBD alpha
        alpha = self.dwbd_scheduler.alpha(self.current_epoch)

    # ── 4. Compute losses ────────────────────────────────────────────────
    # L_DWBD
    l_dwbd = self.loss.dwbd(
        student_out.verifier_logits, verifier_targets, gt_one_hot, alpha
    )

    # L_ver (KL on top-5 positions only)
    l_ver = self.loss.verifier_kl(
        student_out.verifier_logits, verifier_targets, verifier_mask
    )

    # L_cst (re-run fusion for augmented queries, student only)
    l_cst = self.compute_consistency_loss(
        images, all_queries, student_out.layer_attns
    )

    # L_box (select top-1 prediction by verifier score)
    verifier_probs = F.softmax(student_out.verifier_logits, dim=-1)
    ordinal = torch.tensor([3.,2.,1.,0.], device=verifier_probs.device)
    ranking_scores = (verifier_probs * ordinal).sum(-1)  # [B, 150]
    top1_idx = ranking_scores.argmax(dim=-1)             # [B]
    pred_boxes = decoded_boxes[torch.arange(B), top1_idx]  # [B, 4]
    l_box = self.loss.box(pred_boxes, gt_boxes)

    total_loss = (l_dwbd
                  + self.lambda1 * l_cst
                  + self.lambda2 * l_ver
                  + self.lambda3 * l_box)

    # ── 5. Backward (student only) ───────────────────────────────────────
    self.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
    self.optimizer.step()
    self.scheduler.step()

    return total_loss
```

**Memory management:**
- After the verifier forward passes, call `torch.cuda.empty_cache()` if VRAM pressure is high.
- Use `torch.cuda.amp.autocast()` for the student forward pass (FP16) to save VRAM.
- Keep `images_raw` on CPU; only move crops to GPU one sub-batch at a time during Stage 1.

---

### Training Configuration (`configs/phase1.yaml`)

```yaml
training:
  total_epochs: 30
  batch_size: 16         # per GPU; use gradient accumulation steps=4 for effective BS=64
  base_lr: 3e-4
  weight_decay: 0.05
  warmup_steps: 500
  grad_clip_norm: 1.0
  mixed_precision: true  # AMP for student; verifier always FP16 from bitsandbytes

  lambda1: 0.1   # L_cst
  lambda2: 1.0   # L_ver KL
  lambda3: 5.0   # L_box

  dwbd:
    alpha_max: 0.8
    gamma: 2.0

  verifier_kl:
    temperature: 2.0

  text_encoder_freeze_epochs: 5
  text_encoder_lr_scale: 0.1

  token_blur:
    merge_ratio: 0.25     # merge 25% of 200 tokens → 150 tokens
    similarity: "cosine"

  stage1:
    num_proposals: 200
    objectness_threshold: 0.01
    sub_batch_size: 64    # crops per verifier sub-batch

  stage2:
    top_k: 5
    box_color: 
    box_thickness: 3

  verifier_model: "OpenGVLab/InternVL3_5-8B"
  verifier_load_in_4bit: true
```

### Hyperparameter Specification (Authoritative)

#### Optimizer: MuSGD

| Parameter | Value | Reasoning |
|---|---|---|
| `base_lr` | `3e-4` | Standard for transformer fine-tuning from a strong pretrained init; higher than AdamW defaults because MuSGD's orthogonal projection dampens effective step size |
| `momentum` | `0.95` | Slightly higher than SGD default (0.9); Muon's orthogonal update already reduces oscillation, so heavier momentum helps on sparse gradient steps |
| `weight_decay` | `0.05` (fusion/heads), `0.01` (projections), `0.0` (frozen) | Decoupled weight decay; heavier on the fusion transformer (7M params, high capacity risk) |
| `grad_clip_norm` | `1.0` | Hard cap; critical here because verifier KL loss can produce large gradient spikes when teacher labels shift significantly between proposals |
| `eps` | `1e-8` | Default; do not change unless you observe NaN losses |

**Per param-group LR table:**

| Group | Epochs 1–5 | Epochs 6–30 |
|---|---|---|
| `backbone` (YOLO26) | `0.0` (frozen) | `0.0` (frozen) |
| `text_encoder` (MobileCLIP) | `0.0` (frozen) | `3e-5` (`0.1 × base_lr`) |
| `text_projection` | `3e-4` | `3e-4` |
| `fusion_transformer` | `3e-4` | `3e-4` |
| `verifier_head` | `3e-4` | `3e-4` |
| `box_head` | `3e-4` | `3e-4` |

#### LR Schedule

| Parameter | Value | Reasoning |
|---|---|---|
| Warmup type | Linear | Avoids large KL loss spikes at init when verifier labels are misaligned with random student weights |
| Warmup steps | `500` | ~2 epochs at BS=64 on 1.5M samples; enough to stabilize fusion transformer attention before verifier distillation dominates |
| Warmup start LR | `3e-5` (`0.1 × base_lr`) | |
| Decay type | Cosine | |
| Decay end LR | `3e-6` (`0.01 × base_lr`) | |
| Total steps | `~703k` (30 epochs × 1.5M / BS 64) | |

#### Loss Weights

These are the most sensitive hyperparameters in the system. The ratio between `λ_box` and `λ_ver` matters more than their absolute values.

| Parameter | Starting Value | Sweep Range | Priority |
|---|---|---|---|
| `λ₁` (L_cst consistency) | `0.1` | `[0.05, 0.1, 0.2]` | Low — mostly affects robustness, not peak accuracy |
| `λ₂` (L_ver KL) | `1.0` | `[0.5, 1.0, 2.0]` | Medium |
| `λ₃` (L_box GIoU+L1) | `5.0` | `[3.0, 5.0, 8.0]` | **High — primary task signal** |
| `L_DWBD` weight | `1.0` (implicit) | Don't sweep until λ₃ is stable | — |

**Ratio rule of thumb:** `λ₃ / λ₂ ≈ 5` keeps box regression dominant. If validation IoU stagnates early, increase `λ₃` to `8.0`. If verifier head plateaus at random accuracy, increase `λ₂` to `2.0`.

#### DWBD (Dynamic Weight-Balance Distillation)

| Parameter | Value | Reasoning |
|---|---|---|
| `α_max` | `0.8` | At epoch 0: 80% verifier signal, 20% GT hard labels. Prevents early overfitting to noisy GT boxes |
| `α_min` | `0.1` | At final epoch: mostly GT-driven; verifier signal is residual stabilizer |
| `γ` (decay rate) | `2.0` | Quadratic decay — verifier influence drops steeply in mid-training when student is most plastic |
| Schedule formula | `α(t) = (α_max − α_min) × (1 − t/T)^γ + α_min` | |
| `T` | `30` (total epochs) | |

**GT one-hot IoU thresholds for DWBD:**

| IoU Range | Class Label |
|---|---|
| ≥ 0.7 | `Excellent` (0) |
| [0.5, 0.7) | `Good` (1) |
| [0.3, 0.5) | `Partial` (2) |
| < 0.3 | `No` (3) |

#### KL Divergence Temperature

| Parameter | Value | Reasoning |
|---|---|---|
| `τ` (temperature) | `2.0` | Standard for forward KL in classification distillation; softens peaked verifier distributions enough for the student to learn from near-miss proposals |
| Sweep | `[1.0, 2.0, 4.0]` | Run on RefCOCOg val only (hardest benchmark for soft supervision) |

> Rule: if the verifier head learns to always predict `No` (degenerate collapse), increase `τ` to `4.0`. If training is stable but accuracy is below 88% on RefCOCO val by epoch 10, decrease `τ` to `1.5`.

#### Token Blurring (ToB / ToMe)

| Parameter | Value | Reasoning |
|---|---|---|
| `r` (tokens to merge) | `50` (200 → 150) | 25% merge ratio; practical ToMe guidance suggests minimal quality degradation for moderate merge ratios |
| Merge position | After Fusion Layer 2 of 4 | Middle of the stack: early layers need full token set to build cross-modal alignment; late layers benefit from reduced compute |
| Similarity metric | Cosine on key vectors | More stable than raw feature cosine; use `K` projections from the self-attention layer |
| Merge mode | Weighted mean (NOT drop) | Spatial grounding requires all regions to remain representable; dropping destroys spatial reference frame |
| `r` at inference | `50` (same as training) | Must match training to avoid distribution shift in verifier head inputs |

#### Two-Stage Verifier

**Stage 1 (crop scoring):**

| Parameter | Value |
|---|---|
| Num proposals | `200` |
| Objectness threshold | `0.01` (intentionally low for high recall) |
| Sub-batch size | `64` crops per verifier forward pass |
| Crop min-size | `16 × 16 px` (discard degenerate crops smaller than this) |

**Stage 2 (full-image re-scoring):**

| Parameter | Value | Reasoning |
|---|---|---|
| `top_k` | `5` | Balances relational context coverage vs. verifier VRAM cost; sweep `[3, 5, 10]` on RefCOCOg val |
| Box highlight color | `RGB(255, 0, 0)` | Red — maximally distinct from COCO scene colors |
| Box thickness | `3 px` | Visible at typical COCO image sizes (640px); not so thick it occludes the referent |
| Prompt 4 classes | `[Excellent, Good, Partial, No]` | In that exact order to match token ID mapping |

**`scalar_to_soft_label` thresholds (Stage 1 → 4-class mapping):**

| Sigmoid(true_logit) | Soft label `[E, G, P, N]` |
|---|---|
| > 0.85 | `[0.70, 0.20, 0.08, 0.02]` |
| (0.60, 0.85] | `[0.10, 0.60, 0.25, 0.05]` |
| (0.35, 0.60] | `[0.02, 0.15, 0.60, 0.23]` |
| ≤ 0.35 | `[0.01, 0.04, 0.15, 0.80]` |

Calibrate these thresholds after epoch 1 by plotting the Stage-1 score distribution on RefCOCO val. The 0.35/0.60/0.85 splits assume roughly uniform logit spread; adjust inward if scores cluster near 0.

#### Batch Size and Gradient Accumulation

| Parameter | Value | Reasoning |
|---|---|---|
| Per-GPU batch size | `4` | InternVL 4-bit + student AMP leaves ~6–8 GB headroom; 200 crops × 4 samples = 800 crops per Stage-1 call |
| Gradient accumulation steps | `16` | Effective batch size = `4 × 16 = 64` |
| Effective batch size | `64` | Large enough for stable `L_cst` estimates across augmented queries |

#### Regularization

| Parameter | Value | Applied To |
|---|---|---|
| Dropout in fusion transformer | `0.1` | Attention + FFN layers |
| Dropout in verifier head | `0.1` | MLP layers |
| Dropout in box head | `0.0` | Box regression is hurt by dropout; omit it |
| Label smoothing (GT one-hot) | `0.05` | Prevents overconfidence on GT Excellent class during DWBD |
| Stochastic depth (LayerDrop) | `0.1` | Applied to fusion transformer layers |

#### Phase 3: Compression Hyperparameters

**Pruning:**

| Parameter | Value |
|---|---|
| FFN pruning ratio | `20%` (1536 → ~1229 neurons) |
| Ranking criterion | L1-norm of weight rows |
| Heads to prune | 1 per layer (6 → 5 heads) |
| Head ranking | Taylor importance score (1 forward pass on val set) |
| Post-prune fine-tuning epochs | `5` |
| Post-prune LR | `3e-5` (`0.1 × base_lr`) |

**QAT:**

| Parameter | Value |
|---|---|
| QAT epochs | `5` |
| Quantization config (ARM) | `qnnpack` |
| Quantization config (x86/dev) | `x86` |
| Backbone | INT8 weights + INT8 activations |
| Fusion transformer | INT8 weights + INT8 activations |
| Text encoder | **FP16** (CLIP text encoders degrade under INT8) |
| Verifier head | INT8 |
| Box head | INT8 |

#### Ablation Priority Order

Run ablations in this sequence to avoid confounding:

1. `λ₃` sweep (`3.0, 5.0, 8.0`) — box regression dominance is the single biggest accuracy lever
2. `τ` sweep (`1.0, 2.0, 4.0`) — affects RefCOCOg and ReferItGame most
3. `top_k` Stage-2 sweep (`3, 5, 10`) — compute vs. relational accuracy tradeoff
4. `r` merge ratio sweep (`25%, 30%, 35%`) — latency vs. accuracy on RefCOCO+ val
5. `α_max` sweep (`0.6, 0.8, 0.9`) — only after loss balance is stable

---

## Phase 2: Student Fine-Tuning

- After Phase 1 completes, run 5 additional epochs with the verifier **unloaded from VRAM**
  to free memory for higher batch sizes.
- Use only `L_box + L_cst` (no distillation terms).
- All student components trainable (including MobileCLIP text encoder).
- Purpose: close the remaining gap between teacher-supervised accuracy and GT accuracy.

---

## Phase 3: Compression and Export

### Structured Pruning (`src/compression/pruning.py`)

**Target:** Fusion transformer only (backbone and text encoder not pruned).

**FFN pruning (20% channels):**
```python
for layer in fusion_transformer.layers:
    # Rank FFN neurons by L1 norm of weight rows
    ffn_norms = layer.ffn.fc1.weight.abs().sum(dim=1)  # 
    n_prune = int(0.20 * 1536)   # prune 307 neurons → 1229
    prune_indices = ffn_norms.argsort()[:n_prune]
    prune_ffn_channels(layer.ffn, prune_indices)
```

**Attention head pruning (1 head per layer → 6→5 heads):**
```python
for layer in fusion_transformer.layers:
    # Rank heads by importance score (Taylor expansion, 1 pass over val set)
    head_importance = compute_head_importance(layer, val_loader)
    prune_head(layer, head_importance.argmin())
```

Post-pruning: fine-tune for 5 epochs at `base_lr × 0.1` with `L_box + L_cst` only.

### Quantization-Aware Training (`src/compression/qat.py`)

- 5 QAT epochs using PyTorch FX graph mode quantization.
- Student backbone: INT8 weights + INT8 activations.
- Fusion transformer: INT8 weights + INT8 activations.
- Verifier head and box head: INT8.
- Text encoder: FP16 (CLIP text encoder quantizes poorly; keep FP16 for MobileCLIP).
- Use `torch.ao.quantization.get_default_qat_qconfig("x86")` for x86 dev; swap for
  `"qnnpack"` for ARM export.

### Export Targets (`src/export/`)

| Target | Tool | File | Notes |
|--------|------|------|-------|
| Nvidia Jetson | `torch.onnx.export` → TensorRT | `student.engine` | FP16+INT8, TRT 10.x |
| Apple Silicon | `coremltools.convert` | `student.mlpackage` | ANE-optimized, Core ML 7 |
| ARM Android | ONNX Runtime Mobile | `student.ort` | NNAPI execution provider |

**Export invariant:** The exported artifact contains ONLY the student. The InternVL verifier
is never exported, never included in deployment artifacts.

**ONNX export:**
```python
torch.onnx.export(
    student_model,
    (dummy_image, dummy_query_tokens),
    "student.onnx",
    opset_version=17,
    input_names=["image", "query_tokens"],
    output_names=["pred_box", "verifier_scores"],
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
        "query_tokens": {0: "batch"},
    }
)
```

**Expected final artifact:** ~30 MB (INT8), ~3–4 GFLOPs, <30 ms end-to-end on
Snapdragon 8 Gen 3 / Apple A17 NPU.

---

## VALOR Supervision Contract (Summary)

| Contract Item | Implementation Location |
|---|---|
| Student produces candidate boxes | `backbone.py::YOLO26SmallBackbone.forward()` |
| Stage-1 crop scoring (200 proposals) | `internvl_verifier.py::stage1_score_crops()` |
| Stage-2 full-image scoring (top-5) | `internvl_verifier.py::stage2_score_fullimage()` |
| 4-class soft label construction | `internvl_verifier.py::build_verifier_targets()` |
| KL supervision of verifier head | `verifier_kl.py::verifier_kl_loss()` |
| DWBD blending of teacher+GT | `dwbd.py::DWBDLoss.forward()` |
| Ordinal scoring at inference | `verifier.py::VerifierHead.score()` |
| Only student receives gradients | `trainer.py::training_step()` (verifier always `no_grad`) |

---

## Explicit Non-Goals

- No HDF5 or any persistent pseudo-label cache.
- No offline precompute of verifier outputs as training artifacts.
- No training of InternVL (verifier is frozen at all times).
- No export of verifier in any deployment artifact.
- No DFL module (YOLO26 removes it; do not re-introduce it).
- No token dropping in ToB (only similarity-weighted merging).

---

## Implementation Completion Log

- [x] **Phase 1 parity: loss stack split**
  - Create `src/losses/verifier_kl.py`, `src/losses/dwbd.py`, `src/losses/consistency.py`, `src/losses/box_losses.py`, `src/losses/combined.py`.
  - Add tests for each module and keep existing training loop passing.

- [x] **Phase 1 parity: verifier-head 4-class contract**
  - Ensure verifier head exposes `[B, K, 4]` logits and ordinal scoring utility.
  - Add tests for shape and ordinal-score behavior.

- [x] **Phase 2 strict mode**
  - Added explicit verifier-unloaded fine-tuning mode using only `L_box + L_cst`.

- [x] **Phase 1 parity: two-stage verifier path**
  - Implemented Stage-2 full-image highlighted-box scoring utilities for top-k proposals.
  - Added target-building utilities mapping stage outputs to proposal-token supervision targets.

- [x] **Phase 1 parity: training loop alignment**
  - Distillation now computes DWBD + verifier KL + box loss via `GroundingLoss`-based path while keeping verifier frozen.

- [x] **Phase 1 parity: consistency activation**
  - Consistency loss is now computed from student fused-token representations using augmented queries in student-only forward passes (no extra verifier calls).

- [x] **Phase 1 parity: dataset balancing sampler**
  - Training dataloader now applies source-balanced `WeightedRandomSampler` weights from `_aug.pth` source splits.

- [x] **Training objective parity: STAL + ProgLoss integration**
  - Added small-target-aware weighting (STAL) and progressive loss scaling (ProgLoss) hooks in phase1 and phase2 loss composition, configurable via Hydra.

