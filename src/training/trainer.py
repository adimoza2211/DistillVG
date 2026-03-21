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

    def _loss_weights(self) -> dict[str, float]:
        return {
            "lambda_kl": float(self.cfg.lambda_kl),
            "lambda_l1": float(self.cfg.lambda_l1),
            "lambda_giou": float(self.cfg.lambda_giou),
            "lambda_mse": float(self.cfg.lambda_mse),
            "lambda_attn": float(self.cfg.lambda_attn),
            "lambda_cst": float(self.cfg.lambda_cst),
            "lambda_ver": float(self.cfg.lambda_ver),
            "kl_temperature": float(self.cfg.kl_temp_start),
        }
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
            drop_last=False,
        )

    def _build_model(self) -> StudentModel:
        training_cfg = self.cfg.training
        model = StudentModel(
            hidden_dim=int(training_cfg.hidden_dim),
            vocab_size=int(training_cfg.vocab_size),
            tob_tokens=int(self.cfg.model.tob_tokens),
        )
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.text_encoder.parameters():
            parameter.requires_grad = False
        return model.to(self.device)

    def _build_optimizer(self, model: StudentModel) -> torch.optim.Optimizer:
        training_cfg = self.cfg.training
        base_lr = float(training_cfg.lr)
        text_lr = base_lr * 0.1
        trainable_groups = [
            {"params": list(model.fusion.parameters()) + list(model.bbox_head.parameters()) + list(model.verifier.parameters()), "lr": base_lr},
            {"params": list(model.text_encoder.parameters()), "lr": text_lr},
        ]
        return torch.optim.AdamW(trainable_groups, weight_decay=float(training_cfg.weight_decay))

    def _apply_epoch_trainability(self, model: StudentModel, epoch_index: int) -> None:
        model.backbone.eval()
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

        if epoch_index < 5:
            model.text_encoder.eval()
            for parameter in model.text_encoder.parameters():
                parameter.requires_grad = False
        else:
            model.text_encoder.train()
            for parameter in model.text_encoder.parameters():
                parameter.requires_grad = True

        model.fusion.train()
        model.bbox_head.train()
        model.verifier.train()

    @staticmethod
    def _epoch_metrics_template() -> dict[str, float]:
        return {
            "total": 0.0,
            "ce": 0.0,
            "kl": 0.0,
            "l1": 0.0,
            "giou": 0.0,
            "mse": 0.0,
            "attn": 0.0,
            "cst": 0.0,
            "ver": 0.0,
        }

    def train(self) -> None:
        model = self._build_model()
        dataloader = self._build_dataloader()
        optimizer = self._build_optimizer(model)

        use_amp = bool(self.cfg.train.amp) and self.device.type == "cuda"
        grad_accum_steps = int(self.cfg.train.grad_accum_steps)
        grad_clip_norm = float(self.cfg.train.grad_clip_norm)
        scaler = torch.amp.GradScaler(device=self.device.type, enabled=use_amp)
        loss_weights = self._loss_weights()

        effective_batch_size = int(self.cfg.training.micro_batch_size) * grad_accum_steps
        self.logger.info(
            "Starting train | experiment=%s | device=%s | amp=%s | grad_accum=%d | effective_batch=%d | synthetic_scaffold=%s",
            self.cfg.experiment,
            self.device,
            use_amp,
            grad_accum_steps,
            effective_batch_size,
            True,
        )

        model.train()
        start_time = time.time()
        global_step = 0

        for epoch in range(int(self.cfg.training.epochs)):
            self._apply_epoch_trainability(model, epoch)
            optimizer.zero_grad(set_to_none=True)
            metrics = self._epoch_metrics_template()

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            for step, batch in enumerate(dataloader, start=1):
                losses = run_distillation_step(
                    model=model,
                    batch=batch,
                    scaler=scaler,
                    device=self.device,
                    use_amp=use_amp,
                    grad_accum_steps=grad_accum_steps,
                    loss_weights=loss_weights,
                )

                if step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                metrics["total"] += float(losses.total.detach().cpu().item())
                metrics["ce"] += float(losses.ce.detach().cpu().item())
                metrics["kl"] += float(losses.kl.detach().cpu().item())
                metrics["l1"] += float(losses.l1.detach().cpu().item())
                metrics["giou"] += float(losses.giou.detach().cpu().item())
                metrics["mse"] += float(losses.mse.detach().cpu().item())
                metrics["attn"] += float(losses.attn.detach().cpu().item())
                metrics["cst"] += float(losses.cst.detach().cpu().item())
                metrics["ver"] += float(losses.ver.detach().cpu().item())

            denom = max(len(dataloader), 1)
            avg_loss = metrics["total"] / denom
            peak_gpu_memory = 0.0
            if self.device.type == "cuda":
                peak_gpu_memory = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

            self.logger.info(
                "Epoch %d | avg_loss=%.6f | ce=%.6f | kl=%.6f | l1=%.6f | giou=%.6f | mse=%.6f | attn=%.6f | cst=%.6f | ver=%.6f | opt_steps=%d | peak_gpu_mem_mb=%.2f | precision=%s",
                epoch + 1,
                avg_loss,
                metrics["ce"] / denom,
                metrics["kl"] / denom,
                metrics["l1"] / denom,
                metrics["giou"] / denom,
                metrics["mse"] / denom,
                metrics["attn"] / denom,
                metrics["cst"] / denom,
                metrics["ver"] / denom,
                global_step,
                peak_gpu_memory,
                "amp" if use_amp else "fp32",
            )

        total_time = time.time() - start_time
        self.logger.info("Training finished in %.2fs", total_time)
