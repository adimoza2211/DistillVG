# DistillVG

Lightweight visual grounding via progressive knowledge distillation.

## Project Structure
- `configs/`: Hydra configuration files
- `src/`: Source package (models, data, training, utils)
- `scripts/`: Entrypoints for each pipeline phase
- `tests/`: Unit tests
- `data/`: Raw data, paraphrases, and teacher cache
- `outputs/`: Checkpoints, logs, and exports

## Quick Start
```bash
conda activate distill
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train.py experiment=student_full
```
