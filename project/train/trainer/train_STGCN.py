#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/train/trainer/train_STGCN.py
Project: /workspace/code/project/train/trainer
Created Date: Thursday April 16th 2026
Author: Kaixu Chen
-----
Comment:

ST-GCN Multi-Task Classification Trainer for gymnastics action analysis.
Trains four classification heads: twist, posture, relax, total.

Have a good code time :)
-----
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from project.train.models.st_gcn import STGCN, build_stgcn_from_hparams
from project.train.utils.helper import save_helper

logger = logging.getLogger(__name__)


class STGCNTrainer(LightningModule):
    """Multi-task skeleton-based action classifier using ST-GCN."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(hparams.loss, "lr", 1e-4))
        self.weight_decay = float(getattr(hparams.loss, "weight_decay", 1e-4))

        # Build ST-GCN model
        self.model: STGCN = build_stgcn_from_hparams(hparams)

        # Task names
        self.tasks = ["twist", "posture", "relax", "total"]
        self.num_classes = int(getattr(hparams.model, "model_class_num", 3))

        # Metrics for each task
        self.metrics: nn.ModuleDict = nn.ModuleDict()
        for task in self.tasks:
            self.metrics[task] = nn.ModuleDict(
                {
                    "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
                    "precision": MulticlassPrecision(num_classes=self.num_classes),
                    "recall": MulticlassRecall(num_classes=self.num_classes),
                    "f1": MulticlassF1Score(num_classes=self.num_classes),
                }
            )

        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir: Path = (
            Path(str(getattr(hparams, "log_path", "./logs"))) / "test_analysis"
        )
        self.test_pred_by_task: Dict[str, List[torch.Tensor]] = {
            task: [] for task in self.tasks
        }
        self.test_label_by_task: Dict[str, List[torch.Tensor]] = {
            task: [] for task in self.tasks
        }

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass returns dict of logits for each task."""
        return self.model(*args, **kwargs)

    def _extract_logits_and_labels(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        # B, T, J, C
        kpt_tensor = batch["fused_kpt3d"]
        logits = self.model(kpt_tensor)

        labels = batch.get("labels", {})
        batch_size = logits["twist"].shape[0]

        if isinstance(labels, dict):
            label_dict: Dict[str, Any] = {
                "twist": labels.get(
                    "twist", torch.full((batch_size,), -1, dtype=torch.long)
                ),
                "posture": labels.get(
                    "posture", torch.full((batch_size,), -1, dtype=torch.long)
                ),
                "relax": labels.get(
                    "relax", torch.full((batch_size,), -1, dtype=torch.long)
                ),
                "total": labels.get(
                    "total", torch.full((batch_size,), -1, dtype=torch.long)
                ),
            }
        else:
            label_dict = {task: labels for task in self.tasks}

        normalized_labels: Dict[str, torch.Tensor] = {}
        for task in self.tasks:
            value = label_dict[task]
            if isinstance(value, torch.Tensor):
                normalized_labels[task] = value.to(self.device).long()
            else:
                normalized_labels[task] = torch.as_tensor(
                    value, device=self.device, dtype=torch.long
                )

        return logits, normalized_labels, batch_size

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """Shared step for train/val/test."""
        logits, label_dict, batch_size = self._extract_logits_and_labels(batch)
        return self._compute_and_log_loss(logits, label_dict, batch_size, stage)

    def _compute_and_log_loss(
        self,
        logits: Dict[str, torch.Tensor],
        label_dict: Dict[str, torch.Tensor],
        batch_size: int,
        stage: str,
    ) -> torch.Tensor:
        """Compute multi-task loss and log metrics for a stage."""

        total_loss = 0.0
        valid_tasks = 0

        for task in self.tasks:
            if task not in logits:
                logger.warning("Model output missing task '%s'", task)
                continue

            task_label = label_dict[task]
            mask = task_label >= 0
            if mask.sum() == 0:
                logger.debug("No valid labels for task %s in this batch", task)
                continue

            task_logits = logits[task][mask]
            task_label_valid = task_label[mask]

            task_loss = F.cross_entropy(task_logits, task_label_valid)
            total_loss = total_loss + task_loss
            valid_tasks += 1

            self.log(
                f"{stage}/loss_{task}",
                task_loss.item(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

            task_preds = torch.softmax(task_logits, dim=1)
            task_metric_group = cast(nn.ModuleDict, self.metrics[task])
            metric_acc = task_metric_group["accuracy"].to(task_logits.device)
            acc = metric_acc(task_preds, task_label_valid)
            self.log(
                f"{stage}/acc_{task}",
                acc,
                on_step=False,
                on_epoch=True,
                batch_size=task_logits.shape[0],
            )

        if valid_tasks == 0:
            logger.warning("No valid tasks in %s batch", stage)
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            total_loss = total_loss / valid_tasks
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.tensor(
                    total_loss, device=self.device, dtype=torch.float32
                )

        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return total_loss

    def _build_test_pack(
        self,
        logits: Dict[str, torch.Tensor],
        label_dict: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build one test batch payload for reproducibility."""
        pack: Dict[str, Any] = {"tasks": {}}
        for task in self.tasks:
            if task not in logits:
                continue

            task_logits = logits[task].detach()
            task_labels = label_dict[task].detach()
            task_preds = torch.argmax(task_logits, dim=1)
            valid_mask = task_labels >= 0

            pack["tasks"][task] = {
                "logits": task_logits[valid_mask].cpu(),
                "preds": task_preds[valid_mask].cpu(),
                "labels": task_labels[valid_mask].cpu(),
            }

        if "person_id" in batch:
            pack["person_id"] = batch["person_id"]
        if "turn_id" in batch:
            pack["turn_id"] = batch["turn_id"]

        return pack

    def on_test_start(self) -> None:
        self.test_outputs = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)
        self.test_pred_by_task = {task: [] for task in self.tasks}
        self.test_label_by_task = {task: [] for task in self.tasks}

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        logits, label_dict, batch_size = self._extract_logits_and_labels(batch)
        loss = self._compute_and_log_loss(logits, label_dict, batch_size, stage="test")

        # Collect per-task predictions/labels for save_helper at epoch end.
        for task in self.tasks:
            if task not in logits:
                continue
            task_logits = logits[task]
            task_labels = label_dict[task]
            valid_mask = task_labels >= 0
            if valid_mask.sum() == 0:
                continue
            task_preds_softmax = (
                torch.softmax(task_logits[valid_mask], dim=1).detach().cpu()
            )
            task_labels_valid = task_labels[valid_mask].detach().cpu()
            self.test_pred_by_task[task].append(task_preds_softmax)
            self.test_label_by_task[task].append(task_labels_valid)

        self.test_outputs.append(self._build_test_pack(logits, label_dict, batch))
        return loss

    def on_test_epoch_end(self) -> None:
        if not self.test_outputs:
            logger.warning("No test outputs to save")
            return

        fold_name = (
            Path(self.logger.root_dir).name
            if self.logger is not None and hasattr(self.logger, "root_dir")
            else "fold"
        )
        for task in self.tasks:
            preds_list = self.test_pred_by_task[task]
            labels_list = self.test_label_by_task[task]
            if not preds_list or not labels_list:
                continue
            task_save_root = self.test_save_dir / task
            save_helper(
                all_pred=preds_list,
                all_label=labels_list,
                fold=fold_name,
                save_path=str(task_save_root),
                num_class=self.num_classes,
            )

        logger.info("Saved test predictions to %s", result_file)
        logger.info("Saved test metrics to %s", metrics_file)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
