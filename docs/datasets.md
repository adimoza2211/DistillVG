# Datasets & Data Pipeline

## Benchmarks
| Dataset | Splits evaluated | Notes |
|---|---|---|
| RefCOCO | val, testA, testB | testA=people, testB=objects |
| RefCOCO+ | val, testA, testB | No absolute location words |
| RefCOCOg | val, test | Longer relational expressions |
| ReferItGame | test | Ambiguous spatial expressions |

All numbers reported as **Accuracy@0.5** (IoU ≥ 0.5 with GT box).

## Directory Layout
```
data/
├── raw/
│   ├── refcoco/
│   │   ├── images -> data/datasets/other/images/mscoco/images/train2014
│   │   └── annotations/
│   │       ├── corpus.pth
│   │       ├── train.pth
│   │       ├── val.pth
│   │       ├── testA.pth
│   │       └── testB.pth
│   ├── refcoco+/
│   │   ├── images -> data/datasets/other/images/mscoco/images/train2014
│   │   └── annotations/
│   │       ├── corpus.pth
│   │       ├── train.pth
│   │       ├── val.pth
│   │       ├── testA.pth
│   │       └── testB.pth
│   ├── refcocog/
│   │   ├── images -> data/datasets/other/images/mscoco/images/train2014
│   │   └── annotations/
│   │       ├── corpus.pth
│   │       ├── train.pth
│   │       ├── val.pth
│   │       └── test.pth
│   └── referitgame/
│       ├── images -> data/datasets/referit/images
│       └── annotations/
│           ├── corpus.pth
│           ├── train.pth
│           ├── val.pth
│           └── test.pth
├── scene_graphs/               ← Scene Graph extracted contextual relationships
│   ├── refcoco_train_sg.json
│   ├── refcocog_train_sg.json  ← relational semantics extracted from .pth
│   └── ...
└── teacher_cache/
    ├── refcoco_train.h5
    ├── refcoco_val.h5
    └── ...               ← schema documented in docs/teacher_cache.md
```

## Dataset Preparation Gate (`scripts/prepare_data.py`)
Run before Phase 0/1/2:
```bash
conda activate distill
python scripts/organize_raw_data.py
python scripts/prepare_data.py
```

What it does:
- Creates `data/raw/dataset_manifest.json` template if missing
- Enforces `source_path` for each dataset to be **inside this repo** under `data/raw/<dataset_name>`
- Validates required entries for each dataset:
  - `images/`
  - dataset-specific `.pth` files under `annotations/`

The script exits non-zero until all required datasets and annotations are present.

Datasets must be physically present in this repo under:
- `data/raw/refcoco`
- `data/raw/refcoco+`
- `data/raw/refcocog`
- `data/raw/referitgame`

## Scene Graph Context Integration (`scripts/extract_scene_graphs.py`)
- We replace earlier VLM/LLM paraphrase approaches with deterministic Scene Graph relationship extractions.
- Graph files exist for all datasets and splits (e.g., `referit_graphs_test.pth`).
- The scene graph is always centered on the referent/target object, providing crucial context:
  - **Node 0 (root)**: The target object of the grounding query with attributes.
  - **Neighbour Nodes**: Surrounding objects with their respective attributes.
  - **Edges**: Directional relational context between nodes (e.g., `[root] -> over -> [clouds]`).
- Matching strategy for guaranteed correctness and avoiding data-poisoning:
  - We do *not* rely on sequential 1:1 zip-pairing or global absolute indexing due to split/corpus index variations.
  - We use dual multi-key matching to align dataset records to scene graph dictionaries strictly on:
    1. `(image_filename, normalized_phrase)`
    2. `(image_filename, mask_id)`
  - A verification script (`scripts/verify_scene_graphs.py`) proves perfect coverage and strict integrity across all subsets.

Current implementation note (2026-03):
- The pipeline now explicitly loads relational subgraph features or textually linearized relationships directly from `.pth` dictionaries inside `data/annotations/{dataset}/` and maps them back via our verified dual key technique. No VLM paraphrasing is required.

## Teacher Cache Schema (`data/teacher_cache/*.h5`)
HDF5 structure per shard file:
```
/{split}/{sample_id}/
    proposals        float16 [200, 4]    # top-200 box coords (x1,y1,x2,y2)
    logits           float16 [200]       # class-agnostic scores
    proposal_feats   float16 [200, 256]  # per-proposal features
    cross_attn       float16 [L, H, 200, T]  # cross-attention maps (layers×heads×props×tokens)
    pooled_embed     float16 [512]       # multimodal pooled embedding
```

## Recall Gate
Must be verified before starting Phase 2 training (checked in `scripts/train.py`):
- RefCOCO: recall@200 ≥ **97%**
- RefCOCO+: recall@200 ≥ **97%**
- RefCOCOg: recall@200 ≥ **94%**
- ReferItGame: recall@200 ≥ **94%**

Assert with `src/data/recall_gate.py` and log to wandb.
