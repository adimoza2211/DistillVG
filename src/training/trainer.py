from __future__ import annotations

import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.ops import box_iou

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

    @staticmethod
    def _parse_gpu_id_count(gpu_ids: str) -> int:
        ids = [value.strip() for value in gpu_ids.split(",") if value.strip()]
        return len(ids)

    def _resolve_device(self) -> torch.device:
        requested = str(self.cfg.training.device)
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested in {"auto_least_used", "auto_max_free"}:
            return self._select_max_free_cuda_device()
        return torch.device(requested)

    @staticmethod
    def _select_max_free_cuda_device() -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")

        best_index = 0
        best_free = -1
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
            if int(free_bytes) > best_free:
                best_free = int(free_bytes)
                best_index = int(index)
        return torch.device(f"cuda:{best_index}")

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
        normalized = amp_dtype.strip().lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")

    def _num_gpus_for_batch_math(self) -> int:
        available_gpu_ids = str(getattr(self.cfg.training, "available_gpu_ids", "")).strip()
        if available_gpu_ids:
            inferred = self._parse_gpu_id_count(available_gpu_ids)
            if inferred > 0:
                return inferred

        if self.device.type == "cuda":
            return max(torch.cuda.device_count(), 1)
        return 1

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

    @staticmethod
    def _load_records_from_patterns(patterns: list[str]) -> list[GroundingRecord]:
        records: list[GroundingRecord] = []
        for pattern in patterns:
            normalized = pattern.strip()
            if not normalized:
                continue
            records.extend(load_grounding_records(pattern=normalized))
        return records

    def _build_dataloader(self) -> DataLoader:
        training_cfg = self.cfg.training
        train_patterns = [
            str(getattr(self.cfg.data, "train_records_original_glob", "")).strip(),
            str(getattr(self.cfg.data, "train_records_augmented_glob", "")).strip(),
        ]
        records = self._load_records_from_patterns(train_patterns)
        if not records:
            raise RuntimeError(
                "No training grounding records were loaded from configured train patterns. "
                "Set data.train_records_original_glob / data.train_records_augmented_glob to valid paths."
            )

        max_records = int(getattr(training_cfg, "max_records", 0))
        max_records = int(getattr(training_cfg, "max_train_records", max_records))
        if max_records > 0:
            records = records[:max_records]

        self.logger.info("Loaded %d training grounding records from patterns: %s", len(records), train_patterns)

        dataset = AugmentedGroundingDataset(
            records=records,
            image_root=str(self.cfg.verifier.image_root),
            resize_long_edge=int(training_cfg.resize_long_edge),
            padded_square_size=int(training_cfg.padded_square_size),
            max_query_length=int(training_cfg.max_query_length),
            text_vocab_size=int(training_cfg.text_vocab_size),
            source_image_roots={
                str(key): str(value)
                for key, value in getattr(self.cfg.verifier, "source_image_roots", {}).items()
            },
            source_image_styles={
                str(key): str(value)
                for key, value in getattr(self.cfg.verifier, "source_image_styles", {}).items()
            },
        )
        sample_weights = self._compute_sampling_weights(records)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=int(training_cfg.batch_size_per_gpu),
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.persistent_workers),
            collate_fn=collate_grounding_batch,
            drop_last=False,
        )

    def _build_validation_dataloader(self) -> DataLoader:
        val_pattern = str(getattr(self.cfg.data, "val_records_glob", "")).strip()
        if not val_pattern:
            raise RuntimeError("data.val_records_glob must be set for per-epoch validation.")

        records = load_grounding_records(pattern=val_pattern)
        if not records:
            raise RuntimeError(f"No validation grounding records matched pattern: {val_pattern}")

        self.logger.info("Loaded %d validation grounding records from pattern: %s", len(records), val_pattern)

        dataset = AugmentedGroundingDataset(
            records=records,
            image_root=str(self.cfg.verifier.image_root),
            resize_long_edge=int(self.cfg.training.resize_long_edge),
            padded_square_size=int(self.cfg.training.padded_square_size),
            max_query_length=int(self.cfg.training.max_query_length),
            text_vocab_size=int(self.cfg.training.text_vocab_size),
            source_image_roots={
                str(key): str(value)
                for key, value in getattr(self.cfg.verifier, "source_image_roots", {}).items()
            },
            source_image_styles={
                str(key): str(value)
                for key, value in getattr(self.cfg.verifier, "source_image_styles", {}).items()
            },
        )

        return DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size_per_gpu),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.persistent_workers),
            collate_fn=collate_grounding_batch,
            drop_last=False,
        )

    @staticmethod
    def _checkpoint_dir(cfg: DictConfig) -> Path:
        configured = str(getattr(cfg.train, "checkpoint_dir", "checkpoints")).strip()
        return Path(configured)

    def _save_checkpoint(
        self,
        *,
        model: StudentModel,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler.StepLR | None,
        epoch_index: int,
        global_step: int,
        best_val_acc: float,
        latest_path: Path,
        best_path: Path,
        is_best: bool,
    ) -> None:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": int(epoch_index),
            "global_step": int(global_step),
            "best_val_acc": float(best_val_acc),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(state, latest_path)
        if is_best:
            torch.save(state, best_path)

    def _try_resume(
        self,
        *,
        model: StudentModel,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler.StepLR | None,
    ) -> tuple[int, int, float]:
        resume_enabled = bool(getattr(self.cfg.train, "resume", False))
        if not resume_enabled:
            return 0, 0, float("-inf")

        checkpoint_path = str(getattr(self.cfg.train, "resume_checkpoint_path", "")).strip()
        if not checkpoint_path:
            raise RuntimeError("train.resume=true but train.resume_checkpoint_path is empty.")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_acc = float(checkpoint.get("best_val_acc", float("-inf")))
        self.logger.info(
            "Resumed training from checkpoint %s | start_epoch=%d | global_step=%d | best_val_acc=%.6f",
            checkpoint_path,
            start_epoch,
            global_step,
            best_val_acc,
        )
        return start_epoch, global_step, best_val_acc

    def _run_validation_epoch(self, model: StudentModel, dataloader: DataLoader, epoch_index: int) -> tuple[float, float, dict[str, float]]:
        model.eval()
        total = 0
        acc_hits = 0
        iou_sum = 0.0
        source_hits: Counter[str] = Counter()
        source_total: Counter[str] = Counter()

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.device, non_blocking=True)
                token_ids = batch["token_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                target_box = batch["target_box"].to(self.device, non_blocking=True)
                sources = list(batch.get("sources", []))

                outputs = model(images=images, token_ids=token_ids, attention_mask=attention_mask)
                pred_box = outputs["bbox"].detach()

                iou_matrix = box_iou(pred_box, target_box)
                ious = iou_matrix.diagonal()
                hits = (ious >= 0.5)

                batch_size = int(ious.shape[0])
                total += batch_size
                acc_hits += int(hits.sum().item())
                iou_sum += float(ious.sum().item())

                for idx in range(batch_size):
                    source_name = sources[idx] if idx < len(sources) else "unknown"
                    source_total[source_name] += 1
                    if bool(hits[idx].item()):
                        source_hits[source_name] += 1

        overall_acc = (float(acc_hits) / float(total)) if total > 0 else 0.0
        mean_iou = (float(iou_sum) / float(total)) if total > 0 else 0.0
        per_source_acc = {
            source: (float(source_hits[source]) / float(count)) if count > 0 else 0.0
            for source, count in source_total.items()
        }

        self.logger.info(
            "Validation Epoch %d | acc@0.5=%.6f | mean_iou=%.6f | samples=%d",
            epoch_index + 1,
            overall_acc,
            mean_iou,
            total,
        )
        if per_source_acc:
            self.logger.info("Validation per-source acc@0.5 | %s", per_source_acc)

        return overall_acc, mean_iou, per_source_acc

    def _build_model(self) -> StudentModel:
        training_cfg = self.cfg.training
        model = StudentModel(
            hidden_dim=int(training_cfg.hidden_dim),
            fusion_layers=int(self.cfg.model.fusion_layers),
            attention_heads=int(self.cfg.model.attention_heads),
            vocab_size=int(training_cfg.text_vocab_size),
            roi_tokens=int(self.cfg.model.roi_tokens),
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
        freeze_text_epochs = int(getattr(self.cfg.train, "freeze_text_encoder_epochs", 5))
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

        if epoch_index < freeze_text_epochs:
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
        val_dataloader = self._build_validation_dataloader()
        optimizer = self._build_optimizer(model)

        use_amp = bool(self.cfg.train.amp) and self.device.type == "cuda"
        amp_dtype = self._resolve_amp_dtype(str(getattr(self.cfg.train, "amp_dtype", "bf16")))
        grad_accum_steps = int(self.cfg.train.grad_accum_steps)
        grad_clip_norm = float(self.cfg.train.grad_clip_norm)
        log_interval = int(getattr(self.cfg.train, "log_interval", 50))
        use_grad_scaler = use_amp and amp_dtype == torch.float16
        scaler = torch.amp.GradScaler(device=self.device.type, enabled=use_grad_scaler)
        loss_weights = self._loss_weights()
        batches_per_epoch = int(getattr(self.cfg.training, "batches_per_epoch", 0))

        lr_decay_epoch = int(getattr(self.cfg.train, "lr_decay_epoch", 0))
        lr_decay_gamma = float(getattr(self.cfg.train, "lr_decay_gamma", 0.1))
        scheduler: torch.optim.lr_scheduler.StepLR | None = None
        if lr_decay_epoch > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay_gamma)

        num_gpus = self._num_gpus_for_batch_math()
        effective_batch_size = int(self.cfg.training.batch_size_per_gpu) * grad_accum_steps * num_gpus
        self.logger.info(
            "Starting train | experiment=%s | device=%s | amp=%s | amp_dtype=%s | per_gpu_batch=%d | grad_accum=%d | num_gpus=%d | effective_batch=%d | synthetic_scaffold=%s",
            self.cfg.experiment,
            self.device,
            use_amp,
            amp_dtype,
            int(self.cfg.training.batch_size_per_gpu),
            grad_accum_steps,
            num_gpus,
            effective_batch_size,
            False,
        )

        model.train()
        start_time = time.time()
        checkpoint_dir = self._checkpoint_dir(self.cfg)
        latest_checkpoint_path = checkpoint_dir / "latest_pretrain.pt"
        best_checkpoint_path = checkpoint_dir / "best_pretrain.pt"

        start_epoch, global_step, best_val_acc = self._try_resume(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
        )

        for epoch in range(start_epoch, int(self.cfg.training.epochs)):
            self._apply_epoch_trainability(model, epoch)
            optimizer.zero_grad(set_to_none=True)
            metrics = self._epoch_metrics_template()
            processed_steps = 0
            iter_time_total = 0.0
            data_time_total = 0.0
            epoch_total_steps = len(dataloader) if batches_per_epoch <= 0 else min(len(dataloader), batches_per_epoch)
            end = time.time()

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            for step, batch in enumerate(dataloader, start=1):
                if batches_per_epoch > 0 and step > batches_per_epoch:
                    break
                processed_steps += 1
                data_time = time.time() - end
                iter_start = time.time()

                losses = run_distillation_step(
                    model=model,
                    online_verifier=online_verifier,  # type: ignore[arg-type]
                    batch=batch,
                    scaler=scaler,
                    device=self.device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
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
                    amp_dtype=amp_dtype,
                    grad_accum_steps=grad_accum_steps,
                    lambda1=float(self.cfg.lambda1),
                    lambda3=float(self.cfg.lambda3),
                    vocab_size=int(self.cfg.training.text_vocab_size),
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

                iter_time = time.time() - iter_start
                iter_time_total += iter_time
                data_time_total += data_time

                should_log_progress = (step % max(log_interval, 1) == 0) or (step == epoch_total_steps)
                if should_log_progress:
                    avg_iter_time = iter_time_total / max(processed_steps, 1)
                    avg_data_time = data_time_total / max(processed_steps, 1)
                    remaining_steps = max(epoch_total_steps - step, 0)
                    eta = str(timedelta(seconds=int(avg_iter_time * remaining_steps)))
                    current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
                    lr_value = max(current_lrs) if current_lrs else 0.0
                    max_mem = 0.0
                    if self.device.type == "cuda":
                        max_mem = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

                    self.logger.info(
                        "Epoch: [%d]  [%d/%d]  eta: %s  lr: %.6f  loss: %.4f (%.4f)  l1: %.4f (%.4f)  giou: %.4f (%.4f)  ver: %.4f (%.4f)  time: %.4f  data: %.4f  max mem: %.0f",
                        epoch + 1,
                        step,
                        epoch_total_steps,
                        eta,
                        lr_value,
                        float(losses.total.detach().cpu().item()),
                        metrics["total"] / max(processed_steps, 1),
                        float(losses.l1.detach().cpu().item()),
                        metrics["l1"] / max(processed_steps, 1),
                        float(losses.giou.detach().cpu().item()),
                        metrics["giou"] / max(processed_steps, 1),
                        float(losses.ver.detach().cpu().item()),
                        metrics["ver"] / max(processed_steps, 1),
                        avg_iter_time,
                        avg_data_time,
                        max_mem,
                    )

                end = time.time()

            if processed_steps > 0 and processed_steps % grad_accum_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            denom = max(processed_steps, 1)
            avg_loss = metrics["total"] / denom
            peak_gpu_memory = 0.0
            if self.device.type == "cuda":
                peak_gpu_memory = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

            if scheduler is not None:
                scheduler.step()

            current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            max_lr = max(current_lrs) if current_lrs else 0.0

            self.logger.info(
                "Epoch %d | avg_loss=%.6f | l1=%.6f | giou=%.6f | ver=%.6f | opt_steps=%d | lr=%.6e | peak_gpu_mem_mb=%.2f | precision=%s",
                epoch + 1,
                avg_loss,
                metrics["l1"] / denom,
                metrics["giou"] / denom,
                metrics["ver"] / denom,
                global_step,
                max_lr,
                peak_gpu_memory,
                "amp" if use_amp else "fp32",
            )

            val_acc, val_mean_iou, _ = self._run_validation_epoch(model, val_dataloader, epoch)
            self._apply_epoch_trainability(model, epoch)

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch_index=epoch,
                global_step=global_step,
                best_val_acc=best_val_acc,
                latest_path=latest_checkpoint_path,
                best_path=best_checkpoint_path,
                is_best=is_best,
            )
            self.logger.info(
                "Checkpointed epoch %d | latest=%s | best=%s | val_acc=%.6f | val_mean_iou=%.6f | best_val_acc=%.6f",
                epoch + 1,
                latest_checkpoint_path,
                best_checkpoint_path,
                val_acc,
                val_mean_iou,
                best_val_acc,
            )

        total_time = time.time() - start_time
        self.logger.info("Training finished in %.2fs", total_time)
