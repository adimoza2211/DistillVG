# Repo State (Living Document)

> **Update this file whenever you complete a phase milestone, change architecture decisions, or hit a blocker.**
> This is the agent's first read to understand where the project is right now.

Primary references:
- `docs/quickstart.md`
- `docs/efficiency_playbook.md`
- `docs/paraphrase_model_selection.md`

## Current Status
- [ ] Phase 0: Scene graph augmentation
- [ ] Phase 1: Teacher feature precomputation
- [ ] Phase 2: Student training
- [ ] Phase 3: Compression + QAT
- [ ] Evaluation on all splits
- [ ] ONNX/CoreML/TRT export
- [ ] Paper draft

## Current Focus (as of 2026-03-20)
- Documentation-first kickoff completed for implementation readiness
- Hybrid bootstrap policy selected (reuse infra patterns, build research logic custom)
- Scene-graph augmentation is the active Phase 0 path; VLM paraphrasing is retired
- Phase 0 wrapper script now exists as `scripts/generate_paraphrases.py`
- Phase 2 synthetic training harness implemented (Hydra + AMP + grad accumulation + clipping)
- Phase 0 schema-validated scene-graph augmentation baseline implemented (rule-based scaffold)
- Unit tests currently passing (`5 passed`)
- Dataset preparation gate implemented via `scripts/prepare_data.py` + `data/raw/dataset_manifest.json`
- Canonical in-repo raw dataset layout materialized via `scripts/organize_raw_data.py` (COCO + ReferIt images and split `.pth` annotations)
- Repository hygiene updated: `.gitignore` now excludes raw dataset payloads, teacher cache, outputs, checkpoints, the local dataset manifest, and cache/build artifacts

## Active Branch / Last Major Change
- Branch: _to be filled when git branch naming is finalized_
- Last major change: added dataset/annotation validation gate and manifest-driven linking before pipeline execution
- Last major change: reorganized datasets into canonical `data/raw/<dataset>/{images,annotations}` structure and updated validation/tests/docs

## Known Issues / TODOs
- [ ] Confirm SegVG citation details (venue + author list) — marked `[SegVG citation to be confirmed]` in proposal
- [ ] Validate parameter counts against actual model cards during implementation
- [ ] Select λ1–λ7 and KL temperature schedule via ablation studies
- [ ] Determine compressed model size after pruning sensitivity characterisation
- [ ] Obtain physical hardware (Snapdragon 8 Gen 3 device + Apple A17 device) for latency benchmarking

## Key Decisions Log
| Date | Decision | Rationale |
|---|---|---|
| 2026-03 | 7×7 RoI-Align crops (not 4×4) | 3× spatial resolution for left/right/center; parameter-free |
| 2026-03 | Retain 150 active tokens (not 130) post-ToB | Preserves spatial context at negligible cost |
| 2026-03 | Latency as hard constraint, params as soft | Deployment reality on mobile silicon |
| 2026-03 | Backbone-teacher structural alignment via YOLOv8 | Makes feature-MSE distillation semantically valid |
| 2026-03 | Hybrid repo startup strategy | Avoids monolithic fork while preventing greenfield infra overhead |
| 2026-03 | Scene-graph augmentation replaced VLM paraphrasing | Deterministic `_graphs_.pth` lookups avoid poisoned records |
| 2026-03 | Phase 0 wrapper standardized to `scripts/generate_paraphrases.py` | Keeps the command surface stable while delegating to the scene-graph engine |

## Compute Budget
| Task | Estimate |
|---|---|
| Teacher precomputation | ~1 A100-day |
| Student training | ~4–5 A100-days |
| Compression + QAT | ~1 A100-day |
| **Total** | **~6–7 A100-days** |

## Deadlines
| Milestone | Date |
|---|---|
| BMVC 2026 Abstract | 22 May 2026 |
| BMVC 2026 Paper | 29 May 2026 |
