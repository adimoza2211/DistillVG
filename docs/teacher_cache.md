# Teacher Cache Schema

## Purpose
Defines the HDF5 schema used by Phase 1 (`scripts/precompute_teacher.py`) and consumed by Phase 2 training.

## Location
Teacher cache shards are stored under:
- `data/teacher_cache/`

Typical files:
- `refcoco_train.h5`
- `refcoco_val.h5`
- `refcoco+_train.h5`
- `refcocog_train.h5`
- `referitgame_train.h5`

## HDF5 Key Structure
Per sample key:
```
/{split}/{sample_id}/
    proposals        float16 [200, 4]
    logits           float16 [200]
    proposal_feats   float16 [200, 256]
    cross_attn       float16 [L, H, 200, T]
    pooled_embed     float16 [512]
```

## Contracts
- Teacher is always inference-only (`eval` + `no_grad`)
- Cached values are FP16
- `sample_id` keys must align with dataset annotations and paraphrase records
- Missing keys must fail loudly before training starts

## Pre-Training Gate
Before Phase 2 begins, verify recall@200 thresholds:
- RefCOCO: ≥97%
- RefCOCO+: ≥97%
- RefCOCOg: ≥94%
- ReferItGame: ≥94%

## Validation Checklist
- File exists for each required split
- Required tensors present for every sample
- Tensor shapes match schema
- Dtypes are FP16
- Recall gate report is logged
