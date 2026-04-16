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
Last Modified: Thursday April 16th 2026 11:20:00 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from project.train.models.st_gcn import STGCN, build_stgcn_from_hparams

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
        self.metrics = {}
        for task in self.tasks:
            self.metrics[task] = {
                "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
                "precision": MulticlassPrecision(num_classes=self.num_classes),
                "recall": MulticlassRecall(num_classes=self.num_classes),
                "f1": MulticlassF1Score(num_classes=self.num_classes),
            }

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass returns dict of logits for each task."""
        return self.model(*args, **kwargs)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """Shared step for train/val/test."""
        # Extract fused 3D keypoints (N, T, V, C) or (N, C, T, V)
        if "kpt3d_sam" not in batch:
            raise KeyError("batch must contain 'kpt3d_sam' key")

        kpt_dict = batch["kpt3d_sam"]
        if isinstance(kpt_dict, dict):
            # Dict input must provide fused keypoints.
            kpt_tensor = kpt_dict.get("fused")
            if kpt_tensor is None:
                raise KeyError("kpt3d_sam dict must contain 'fused'")
        else:
            kpt_tensor = kpt_dict

        logits = self.model(x=kpt_tensor.float())

        # Extract labels
        labels = batch.get("labels", {})
        batch_size = logits["twist"].shape[0]
        
        if isinstance(labels, dict):
            label_dict = {
                "twist": labels.get("twist", torch.full((batch_size,), -1, dtype=torch.long)),
                "posture": labels.get("posture", torch.full((batch_size,), -1, dtype=torch.long)),
                "relax": labels.get("relax", torch.full((batch_size,), -1, dtype=torch.long)),
                "total": labels.get("total", torch.full((batch_size,), -1, dtype=torch.long)),
            }
        else:
            # Fallback if labels is not dict
            label_dict = {task: labels for task in self.tasks}

        # Move labels to device
        for task in self.tasks:
            label_dict[task] = label_dict[task].to(self.device)

        # Compute loss
        total_loss = 0.0
        valid_tasks = 0
        
        for task in self.tasks:
            if task not in logits:
                logger.warning("Model output missing task '%s'", task)
                continue

            task_label = label_dict[task]
            # Filter out invalid labels (-1)
            mask = task_label >= 0
            if mask.sum() == 0:
                logger.debug("No valid labels for task %s in this batch", task)
                continue

            task_logits = logits[task][mask]
            task_label_valid = task_label[mask]

            task_loss = F.cross_entropy(task_logits, task_label_valid)
            total_loss = total_loss + task_loss
            valid_tasks += 1

            # Log per-task loss
            self.log(
                f"{stage}/loss_{task}",
                task_loss.item(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

            # Log per-task accuracy
            task_preds = torch.softmax(task_logits, dim=1)
            acc = self.metrics[task]["accuracy"](task_preds, task_label_valid)
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
            total_loss = total_loss / valid_tasks  # Average over tasks
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.tensor(total_loss, device=self.device, dtype=torch.float32)

        # Log loss
        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return total_loss

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, stage="train")
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, stage="val")
        return loss

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, stage="test")
        return loss

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
