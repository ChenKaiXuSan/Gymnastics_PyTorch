#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.make_model import select_model


class LateFusion3DCNNTrainer(LightningModule):
    """
    Late-fusion multi-view video classifier.

    Expected batch format:
        batch["video"]["front"] : (B, C, T, H, W)
        batch["video"]["left"]  : (B, C, T, H, W)
        batch["video"]["right"] : (B, C, T, H, W)
        batch["label"]          : (B,)
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = hparams.data.img_size
        self.lr = hparams.loss.lr
        self.num_classes = hparams.model.model_class_num

        # three independent backbones
        self.front_cnn = select_model(hparams)
        self.left_cnn = select_model(hparams)
        self.right_cnn = select_model(hparams)

        # fusion config (optional)
        self.fusion_mode = getattr(hparams.model, "fusion_mode", "logit_mean")
        # OOM guard (optional)
        self.batch_size = int(getattr(hparams.data, "batch_size", 1))
        self.video_batch_size = int(getattr(hparams.data, "video_batch_size", 8))

        # metrics
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    # ---- core ----
    def forward(self, videos: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        videos: dict with keys: front/left/right, each (B,C,T,H,W)
        returns: fused logits (B,num_classes)
        """
        front_logits = self.front_cnn(videos["front"])
        left_logits = self.left_cnn(videos["left"])
        right_logits = self.right_cnn(videos["right"])

        if self.fusion_mode == "logit_mean":
            fused_logits = (front_logits + left_logits + right_logits) / 3.0
            return fused_logits

        if self.fusion_mode == "prob_mean":
            front_p = torch.softmax(front_logits, dim=1)
            left_p = torch.softmax(left_logits, dim=1)
            right_p = torch.softmax(right_logits, dim=1)
            fused_p = (front_p + left_p + right_p) / 3.0
            # convert back to logits for CE stability
            fused_logits = torch.log(torch.clamp(fused_p, min=1e-8))
            return fused_logits

        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def _maybe_trim_batch(self, videos: Dict[str, torch.Tensor], label: torch.Tensor):
        """
        Simple OOM guard: trim batch if B is too large.
        """
        # TODO: 这里需要修改一个视频内部的多个片段的情况，目前只能按整体 batch size 来裁剪。
        bsz = label.size(0)
        if bsz <= self.video_batch_size:
            return videos, label

        idx = slice(0, self.video_batch_size)
        videos_trim = {k: v[idx].detach() for k, v in videos.items()}
        label_trim = label[idx]
        return videos_trim, label_trim

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:

        if self.batch_size == 1:
            videos = batch["video"]
            label = batch["label"].squeeze()

            # detach videos (optional; remove detach if you want grads through input preprocessing)
            videos = {k: v.detach().squeeze() for k, v in videos.items()}
        videos, label = self._maybe_trim_batch(videos, label)

        logits = self(videos)  # fused logits
        loss = F.cross_entropy(logits, label.long())

        probs = torch.softmax(logits, dim=1)

        # metrics
        acc = self._accuracy(probs, label)
        precision = self._precision(probs, label)
        recall = self._recall(probs, label)
        f1 = self._f1_score(probs, label)
        _ = self._confusion_matrix(
            probs, label
        )  # if you want to log later, store it yourself

        self.log(
            f"{stage}/loss", loss, on_step=True, on_epoch=True, batch_size=label.size(0)
        )
        self.log_dict(
            {
                f"{stage}/video_acc": acc,
                f"{stage}/video_precision": precision,
                f"{stage}/video_recall": recall,
                f"{stage}/video_f1_score": f1,
            },
            on_step=True,
            on_epoch=True,
            batch_size=label.size(0),
        )
        return loss

    # ---- lightning hooks ----
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
