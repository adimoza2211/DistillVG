# Agent Global Instructions

- Architecture: Knowledge distillation (Grounding DINO to YOLOv8-small + MobileCLIP-S1)
- No longer using VLM models or LLMs for data augmentation. Paraphrasing is replaced by `_aug.pth` datasets that encode offline relational context using structurally parsed scene graphs (`_graphs_.pth`).
- Training must stick to exactly 3 augmented queries per train/val sample.
- Target metric is Accuracy@0.5 (evaluated jointly over all RefCOCO and ReferItGame test/val splits).
- Phase 0 user-facing command: `python scripts/generate_paraphrases.py dataset=refcocog`.
- See `docs/copilot-instructions.md` for primary setup.
