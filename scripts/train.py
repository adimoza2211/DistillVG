from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.training.trainer import Trainer


def _configure_visible_gpus(cfg: DictConfig) -> None:
    training_cfg = cfg.training
    available_gpu_ids = str(getattr(training_cfg, "available_gpu_ids", "")).strip()
    if not available_gpu_ids:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu_ids


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    _configure_visible_gpus(cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
