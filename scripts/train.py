from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.training.trainer import Trainer


def _configure_visible_gpus(cfg: DictConfig) -> None:
    training_cfg = cfg.training
    gpu_ids = str(getattr(training_cfg, "gpu_ids", "")).strip()
    num_gpus = int(getattr(training_cfg, "num_gpus", 0))

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return

    if num_gpus > 0:
        requested_ids = ",".join(str(index) for index in range(num_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_ids


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    _configure_visible_gpus(cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
