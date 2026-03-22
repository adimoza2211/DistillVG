from __future__ import annotations

import time
from collections import Counter

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.collate import collate_grounding_batch
from src.data.grounding import AugmentedGroundingDataset, GroundingRecord, load_grounding_records
from src.models.student.model import StudentModel
from src.training.optimizers import MuSGD
from src.models.verifier.runtime import BaseOnlineVerifier, build_online_verifier
from src.training.distill import run_distillation_step, run_phase2_finetune_step
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
            "lambda_l1": float(self.cfg.train.lambda_l1),
            "lambda_giou": float(self.cfg.train.lambda_giou),
            "lambda_ver": float(self.cfg.train.lambda_ver),
        }

    def _train_mode(self) -> str:
        mode = str(getattr(self.cfg.train, "mode", "phase1")).strip().lower()
        if mode not in {"phase1", "phase2_strict"}:
            raise ValueError(f"Unsupported training mode: {mode}")
        return mode

    @staticmethod
    def _compute_sampling_weights(records: list[GroundingRecord]) -> torch.Tensor:
        source_counts = Counter(record.source for record in records)
        weights = [1.0 / float(source_counts[record.source]) for record in records]
        return torch.tensor(weights, dtype=torch.double)

    def _build_dataloader(self) -> DataLoader:
        training_cfg = self.cfg.training
        phrase_bank_pattern = str(self.cfg.verifier.augmented_phrase_glob)
        records = load_grounding_records(pattern=phrase_bank_pattern)
        if not records:
            raise RuntimeError(
                f"No augmented grounding records matched pattern: {phrase_bank_pattern}. "
                "Phase 1 requires real _aug.pth-backed training data."
            )

        max_records = int(getattr(training_cfg, "max_records", 0))
        if max_records > 0:
            records = records[:max_records]

        self.logger.info("Loaded %d grounding records from pattern: %s", len(records), phrase_bank_pattern)

        dataset = AugmentedGroundingDataset(
            records=records,
            image_root=str(self.cfg.verifier.image_root),
            image_size=int(training_cfg.image_size),
            seq_len=int(training_cfg.seq_len),
            vocab_size=int(training_cfg.vocab_size),
        )
        sample_weights = self._compute_sampling_weights(records)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=int(training_cfg.micro_batch_size),
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.persistent_workers),
            collate_fn=collate_grounding_batch,
            drop_last=False,
        )

    def _build_model(self) -> StudentModel:
        training_cfg = self.cfg.training
        model = StudentModel(
            hidden_dim=int(training_cfg.hidden_dim),
            vocab_size=int(training_cfg.vocab_size),
            tob_tokens=int(self.cfg.model.tob_tokens),
            proposal_count=int(self.cfg.model.proposal_count),
            use_yolo26_proposals=bool(self.cfg.model.proposal_generator.use_yolo26),
            yolo26_model_cfg=str(self.cfg.model.proposal_generator.yolo26_model_cfg),
            yolo26_weights_path=str(self.cfg.model.proposal_generator.yolo26_weights_path),
            yolo26_conf_threshold=float(self.cfg.model.proposal_generator.yolo26_conf_threshold),
            yolo26_iou_threshold=float(self.cfg.model.proposal_generator.yolo26_iou_threshold),
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
        mode = self._train_mode()
        if mode == "phase2_strict":
            trainable_groups: list[dict[str, object]] = [
                {"params": list(model.backbone.parameters()), "lr": base_lr * 0.1},
                {
                    "params": list(model.fusion.parameters())
                    + list(model.bbox_head.parameters())
                    + list(model.proposal_bbox_head.parameters())
                    + list(model.verifier.parameters()),
                    "lr": base_lr,
                },
                {"params": list(model.text_encoder.parameters()), "lr": base_lr},
            ]
        else:
            trainable_groups = [
            {
                "params": list(model.fusion.parameters())
                + list(model.bbox_head.parameters())
                + list(model.proposal_bbox_head.parameters())
                + list(model.verifier.parameters()),
                "lr": base_lr,
            },
            {"params": list(model.text_encoder.parameters()), "lr": text_lr},
            ]
        optimizer_name = str(self.cfg.train.optimizer).lower()
        if optimizer_name == "musgd":
            optimizer = MuSGD(
                trainable_groups,
                lr=base_lr,
                momentum=float(self.cfg.train.musgd.momentum),
                weight_decay=float(training_cfg.weight_decay),
                muon_beta=float(self.cfg.train.musgd.muon_beta),
                eps=float(self.cfg.train.musgd.eps),
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(trainable_groups, weight_decay=float(training_cfg.weight_decay))
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.train.optimizer}")

        return optimizer

    def _build_online_verifier(self) -> BaseOnlineVerifier:
        return build_online_verifier(cfg=self.cfg.verifier, device=self.device)

    def _apply_epoch_trainability(self, model: StudentModel, epoch_index: int) -> None:
        mode = self._train_mode()
        if mode == "phase2_strict":
            model.backbone.train()
            model.text_encoder.train()
            model.fusion.train()
            model.bbox_head.train()
            model.proposal_bbox_head.train()
            model.verifier.train()
            for parameter in model.parameters():
                parameter.requires_grad = True
            return

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
        model.proposal_bbox_head.train()
        model.verifier.train()

    @staticmethod
    def _epoch_metrics_template() -> dict[str, float]:
        return {
            "total": 0.0,
            "l1": 0.0,
            "giou": 0.0,
            "ver": 0.0,
        }

    def train(self) -> None:
        model = self._build_model()
        mode = self._train_mode()
        online_verifier = self._build_online_verifier() if mode == "phase1" else None
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
            False,
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
                    online_verifier=online_verifier,  # type: ignore[arg-type]
                    batch=batch,
                    scaler=scaler,
                    device=self.device,
                    use_amp=use_amp,
                    grad_accum_steps=grad_accum_steps,
                    loss_weights=loss_weights,
                    verifier_crop_size=int(self.cfg.verifier.crop_size),
                    verifier_top_k_proposals=int(self.cfg.verifier.top_k_proposals),
                    verifier_stage2_top_k=int(self.cfg.verifier.stage2_top_k),
                    use_augmented_verifier_queries=bool(self.cfg.verifier.use_augmented_queries),
                    verifier_query_selection=str(self.cfg.verifier.query_selection),
                    epoch_index=epoch,
                    total_epochs=int(self.cfg.training.epochs),
                    lambda1=float(self.cfg.lambda1),
                    lambda2=float(self.cfg.lambda2),
                    lambda3=float(self.cfg.lambda3),
                    dwbd_alpha_max=float(self.cfg.train.dwbd.alpha_max),
                    dwbd_alpha_min=float(self.cfg.train.dwbd.alpha_min),
                    dwbd_gamma=float(self.cfg.train.dwbd.gamma),
                    stal_enabled=bool(self.cfg.train.stal.enabled),
                    stal_area_threshold=float(self.cfg.train.stal.area_threshold),
                    stal_max_boost=float(self.cfg.train.stal.max_boost),
                    progloss_enabled=bool(self.cfg.train.progloss.enabled),
                    progloss_box_start=float(self.cfg.train.progloss.box_start_scale),
                    progloss_box_end=float(self.cfg.train.progloss.box_end_scale),
                    progloss_verifier_start=float(self.cfg.train.progloss.verifier_start_scale),
                    progloss_verifier_end=float(self.cfg.train.progloss.verifier_end_scale),
                    progloss_power=float(self.cfg.train.progloss.power),
                ) if mode == "phase1" else run_phase2_finetune_step(
                    model=model,
                    batch=batch,
                    scaler=scaler,
                    device=self.device,
                    use_amp=use_amp,
                    grad_accum_steps=grad_accum_steps,
                    lambda1=float(self.cfg.lambda1),
                    lambda3=float(self.cfg.lambda3),
                    vocab_size=int(self.cfg.training.vocab_size),
                    epoch_index=epoch,
                    total_epochs=int(self.cfg.training.epochs),
                    stal_enabled=bool(self.cfg.train.stal.enabled),
                    stal_area_threshold=float(self.cfg.train.stal.area_threshold),
                    stal_max_boost=float(self.cfg.train.stal.max_boost),
                    progloss_enabled=bool(self.cfg.train.progloss.enabled),
                    progloss_box_start=float(self.cfg.train.progloss.box_start_scale),
                    progloss_box_end=float(self.cfg.train.progloss.box_end_scale),
                    progloss_power=float(self.cfg.train.progloss.power),
                )

                if step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                metrics["total"] += float(losses.total.detach().cpu().item())
                metrics["l1"] += float(losses.l1.detach().cpu().item())
                metrics["giou"] += float(losses.giou.detach().cpu().item())
                metrics["ver"] += float(losses.ver.detach().cpu().item())

            denom = max(len(dataloader), 1)
            avg_loss = metrics["total"] / denom
            peak_gpu_memory = 0.0
            if self.device.type == "cuda":
                peak_gpu_memory = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

            self.logger.info(
                "Epoch %d | avg_loss=%.6f | l1=%.6f | giou=%.6f | ver=%.6f | opt_steps=%d | peak_gpu_mem_mb=%.2f | precision=%s",
                epoch + 1,
                avg_loss,
                metrics["l1"] / denom,
                metrics["giou"] / denom,
                metrics["ver"] / denom,
                global_step,
                peak_gpu_memory,
                "amp" if use_amp else "fp32",
            )

        total_time = time.time() - start_time
        self.logger.info("Training finished in %.2fs", total_time)
