from __future__ import annotations

import time

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.collate import collate_synthetic_batch
from src.data.synthetic import SyntheticGroundingDataset
from src.models.student.model import StudentModel
from src.training.distill import run_distillation_step
from src.utils.logging import get_logger
from src.utils.seed import set_seed


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger("distillvg.trainer")
        self.device = self._resolve_device()
        set_seed(int(self.cfg.seed), deterministic=False)

    def _resolve_device(self) -> torch.device:
        requested = str(self.cfg.training.device)
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _build_dataloader(self) -> DataLoader:
        training_cfg = self.cfg.training
        samples = int(training_cfg.steps_per_epoch) * int(training_cfg.micro_batch_size)
        dataset = SyntheticGroundingDataset(
            num_samples=samples,
            image_size=int(training_cfg.image_size),
            seq_len=int(training_cfg.seq_len),
            vocab_size=int(training_cfg.vocab_size),
        )
        return DataLoader(
            dataset,
            batch_size=int(training_cfg.micro_batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.persistent_workers),
            collate_fn=collate_synthetic_batch,
        )

    def _build_model(self) -> StudentModel:
        training_cfg = self.cfg.training
        model = StudentModel(
            hidden_dim=int(training_cfg.hidden_dim),
            vocab_size=int(training_cfg.vocab_size),
            tob_tokens=int(self.cfg.model.tob_tokens),
        )
        return model.to(self.device)


    def train(self) -> None:
        model = self._build_model()
        dataloader = self._build_dataloader()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )

        use_amp = bool(self.cfg.train.amp) and self.device.type == "cuda"
        grad_accum_steps = int(self.cfg.train.grad_accum_steps)
        grad_clip_norm = float(self.cfg.train.grad_clip_norm)
        scaler = torch.amp.GradScaler(device=self.device.type, enabled=use_amp)

        effective_batch_size = int(self.cfg.training.micro_batch_size) * grad_accum_steps
        self.logger.info(
            "Starting train | experiment=%s | device=%s | amp=%s | grad_accum=%d | effective_batch=%d",
            self.cfg.experiment,
            self.device,
            use_amp,
            grad_accum_steps,
            effective_batch_size,
        )

        model.train()
        start_time = time.time()
        global_step = 0

        for epoch in range(int(self.cfg.training.epochs)):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for step, batch in enumerate(dataloader, start=1):
                losses = run_distillation_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=self.device,
                    use_amp=use_amp,
                    grad_accum_steps=grad_accum_steps,
                )

                if step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running_loss += float(losses.total.detach().cpu().item())

            avg_loss = running_loss / max(len(dataloader), 1)
            self.logger.info("Epoch %d | avg_loss=%.6f | opt_steps=%d", epoch + 1, avg_loss, global_step)

        total_time = time.time() - start_time
        self.logger.info("Training finished in %.2fs", total_time)
