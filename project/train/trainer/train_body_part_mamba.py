#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/trainer/train_body_part_mamba.py
Project: /workspace/skeleton/project/trainer
Created Date: Friday June 7th 2024
Author: Kaixu Chen
-----
Comment:
This file implements the training process for two stream method.
In this two streams are trained separately and then the results of two streams are fused to get the final result.
Here, saving the results and calculating the metrics are done in separate functions.

Have a good code time :)
-----
Last Modified: Friday June 7th 2024 7:50:12 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, Dict, cast
import logging


from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)

from models.skeleton_mamba import build_skeleton_mamba_model

logger = logging.getLogger(__name__)


class BodyPartMambaClassificationModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        # Support both full config and model-only config.
        data_cfg = getattr(hparams, "data", hparams)
        loss_cfg = getattr(hparams, "loss", None)

        self.lr = getattr(loss_cfg, "lr", 0.001)
        self.weight_decay = getattr(loss_cfg, "weight_decay", 0.01)
        self.label_smoothing = max(
            0.0, min(1.0, float(getattr(loss_cfg, "label_smoothing", 0.0)))
        )

        model_cfg = getattr(hparams, "model", hparams)
        num_class = getattr(model_cfg, "model_class_num", 3)
        num_total_class = getattr(model_cfg, "model_total_class_num", 5)

        self.num_joints = int(getattr(data_cfg, "num_joints", 70))
        self.root_idx = int(getattr(data_cfg, "root_idx", 0))
        scale_joints = getattr(data_cfg, "scale_joints", None)
        if isinstance(scale_joints, (list, tuple)) and len(scale_joints) == 2:
            scale_joints = (int(scale_joints[0]), int(scale_joints[1]))
        else:
            scale_joints = None

        self.model = build_skeleton_mamba_model(
            num_joints=self.num_joints,
            num_classes=num_class,
            total_class_num=num_total_class,
            d_model=int(getattr(data_cfg, "d_model", 256)),
            d_state=int(getattr(data_cfg, "d_state", 16)),
            d_conv=int(getattr(data_cfg, "d_conv", 4)),
            expand=int(getattr(data_cfg, "expand", 2)),
            root_idx=self.root_idx,
            scale_joints=scale_joints,
            dropout=float(getattr(data_cfg, "dropout", 0.0)),
        )

        # task names
        self.tasks = hparams.model.get(
            "class_task", ["twist", "posture", "relax", "total"]
        )

        # Metrics for each task
        self.metrics: nn.ModuleDict = nn.ModuleDict()
        for task in ["twist", "posture", "relax"]:
            self.metrics[task] = nn.ModuleDict(
                {
                    "accuracy": MulticlassAccuracy(num_classes=num_class),
                    "f1": MulticlassF1Score(num_classes=num_class),
                }
            )
        self.metrics["total"] = nn.ModuleDict(
            {
                "accuracy": MulticlassAccuracy(num_classes=num_total_class),
                "f1": MulticlassF1Score(num_classes=num_total_class),
            }
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
                "monitor": "val/loss",
            },
        }

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, stage="val")
        return loss

    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, stage="test")
        return loss

    @staticmethod
    def _unpack_batch(batch: Dict[str, Any]):
        if isinstance(batch, dict):
            # Compatible with existing dataloader output and legacy keys.
            kpt_3d = batch.get("fused_kpt3d", batch.get("kpt_3d"))
            labels = batch.get("labels")
        else:
            raise TypeError(
                f"Unsupported batch type: {type(batch)}. Expected dict batch from dataloader."
            )

        if kpt_3d is None:
            raise ValueError("Batch must contain fused_kpt3d or kpt_3d.")

        if not isinstance(kpt_3d, torch.Tensor):
            raise TypeError(f"Keypoints must be Tensor, got {type(kpt_3d)}")

        kpt_3d = kpt_3d.float()

        if kpt_3d.ndim != 4:
            raise ValueError(
                f"Expected kpt_3d shape (B, T, J, 3), got {tuple(kpt_3d.shape)}"
            )

        batch_size = kpt_3d.shape[0]
        label_dict: Dict[str, torch.Tensor] = {}
        for task in ["twist", "posture", "relax", "total"]:
            value = labels.get(task) if isinstance(labels, dict) else None
            if value is None:
                label_dict[task] = torch.full((batch_size,), -1, dtype=torch.long)
            elif isinstance(value, torch.Tensor):
                label_dict[task] = value.long().view(-1)
            else:
                label_dict[task] = torch.as_tensor(value, dtype=torch.long).view(-1)

        return kpt_3d, label_dict, batch_size

    @staticmethod
    def _normalize_logits(preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits: Dict[str, torch.Tensor] = {}
        key_map = {
            "twist_logits": "twist",
            "posture_logits": "posture",
            "relax_logits": "relax",
            "total_logits": "total",
        }

        for raw_key, value in preds.items():
            if not isinstance(value, torch.Tensor):
                continue
            mapped_key = key_map.get(raw_key)
            if mapped_key is not None:
                logits[mapped_key] = value

        return logits

    def _should_sync_dist(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        world_size = getattr(trainer, "world_size", 1)
        return isinstance(world_size, int) and world_size > 1

    def _compute_and_log_loss(
        self,
        stage: str,
        logits: Dict[str, torch.Tensor],
        label_dict: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        sync_dist = self._should_sync_dist()
        total_loss: torch.Tensor = torch.tensor(
            0.0, device=self.device, dtype=torch.float32
        )
        valid_tasks = 0

        for task in self.tasks:
            if task not in logits:
                continue

            task_logits = logits[task]
            if task_logits.ndim != 2 or task_logits.shape[1] <= 1:
                logger.warning(
                    "Skip task '%s': invalid logits shape %s",
                    task,
                    tuple(task_logits.shape),
                )
                continue

            task_label = label_dict[task].to(task_logits.device)
            if task_label.shape[0] != task_logits.shape[0]:
                logger.warning(
                    "Skip task '%s': label/logit batch mismatch (%s vs %s)",
                    task,
                    task_label.shape[0],
                    task_logits.shape[0],
                )
                continue

            valid_mask = (task_label >= 0) & (task_label < task_logits.shape[1])
            if valid_mask.sum() == 0:
                continue

            valid_logits = task_logits[valid_mask]
            valid_labels = task_label[valid_mask]

            task_loss: torch.Tensor = F.cross_entropy(
                valid_logits,
                valid_labels,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + task_loss
            valid_tasks += 1

            self.log(
                f"{stage}/loss_{task}",
                task_loss,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

            preds = torch.argmax(valid_logits, dim=1)
            metric_group = cast(nn.ModuleDict, self.metrics[task])
            acc = metric_group["accuracy"].to(valid_logits.device)(preds, valid_labels)
            f1 = metric_group["f1"].to(valid_logits.device)(preds, valid_labels)

            self.log(
                f"{stage}/acc_{task}",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=(stage != "train"),
                batch_size=valid_logits.shape[0],
                sync_dist=sync_dist,
            )
            self.log(
                f"{stage}/f1_{task}",
                f1,
                on_step=False,
                on_epoch=True,
                prog_bar=(stage != "train"),
                batch_size=valid_logits.shape[0],
                sync_dist=sync_dist,
            )

        stage_loss: torch.Tensor
        if valid_tasks == 0:
            stage_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            stage_loss = total_loss / valid_tasks

        self.log(
            f"{stage}/loss",
            stage_loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        return stage_loss

    def _shared_step(self, batch, stage="train"):
        kpt_3d, label_dict, batch_size = self._unpack_batch(batch)
        preds = self.model(kpt_3d, return_dict=True)
        if not isinstance(preds, dict):
            raise TypeError(
                f"Model must return dict when return_dict=True, got {type(preds)}"
            )
        logits = self._normalize_logits(preds)
        return self._compute_and_log_loss(stage, logits, label_dict, batch_size)
