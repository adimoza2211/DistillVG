from __future__ import annotations

import hydra
from omegaconf import DictConfig

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
