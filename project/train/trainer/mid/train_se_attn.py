"""
File: train.py
Project: project
Created Date: 2023-10-19 02:29:47
Author: chenkaixu
-----
Comment:
 This file is the train/val/test process for the project.


Have a good code time!
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

from typing import Any, List, Optional, Union
import logging

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

from project.models.se_attn_res_3dcnn import SEFusionRes3DCNN

from project.utils.helper import save_helper
from project.utils.save_CAM import dump_all_feature_maps

logger = logging.getLogger(__name__)


class SEAttnTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()  # 先保存，方便 ckpt/repro

        self.img_size = hparams.data.img_size
        self.lr = getattr(hparams.loss, "lr", 1e-3)  # default lr
        self.num_classes = int(hparams.model.model_class_num)

        # define model
        self.model = SEFusionRes3DCNN(hparams)

        # metrics（torchmetrics 多数支持 logits/probs，内部会做 argmax）
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        # loss 权重（可在 YAML 覆盖）
        self.lambda_list = list(
            getattr(hparams.loss, "lambda_list", [0.25, 0.5, 0.75, 1.0])
        )

        self.save_root = getattr(hparams.train, "log_path", "./logs")


    # ------------------- training / validation -------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        logits = self.model(video, attn_map)

        probs = torch.softmax(logits, dim=1)
        loss_cls = F.cross_entropy(logits, labels)

        # logging
        self.log("train/loss", loss_cls, on_step=True, on_epoch=True, batch_size=B)
        
        self.log_dict(
            {
                "train/video_acc": self._accuracy(probs, labels),
                "train/video_precision": self._precision(probs, labels),
                "train/video_recall": self._recall(probs, labels),
                "train/video_f1_score": self._f1_score(probs, labels),
            },
            on_step=True,
            on_epoch=True,
            batch_size=B,
        )

        logger.info(
            f"train loss: {loss_cls.item():.4f} "
        )
        return loss_cls

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        logits = self.model(video, attn_map)

        probs = torch.softmax(logits, dim=1)
        loss_cls = F.cross_entropy(logits, labels)
        
        # 建议验证只 on_epoch 记录，减少噪声
        self.log("val/loss", loss_cls, on_step=False, on_epoch=True, batch_size=B)

        self.log_dict(
            {
                "val/video_acc": self._accuracy(probs, labels),
                "val/video_precision": self._precision(probs, labels),
                "val/video_recall": self._recall(probs, labels),
                "val/video_f1_score": self._f1_score(probs, labels),
            },
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )

        logger.info(
            f"val loss: {loss_cls.item():.4f} "
        )
        return {"val_loss": loss_cls}

    # ------------------- testing -------------------
    def on_test_start(self) -> None:
        
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []
        logger.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logger.info("test end")


    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        logits = self.model(video, attn_map)
        probs = torch.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=B)

        self.log_dict(
            {
                "test/video_acc": self._accuracy(probs, labels),
                "test/video_precision": self._precision(probs, labels),
                "test/video_recall": self._recall(probs, labels),
                "test/video_f1_score": self._f1_score(probs, labels),
            },
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )

        self.test_pred_list.append(probs.detach().cpu())
        self.test_label_list.append(labels.detach().cpu())

        # dump CAMs
        fold = (
            getattr(self.logger, "root_dir", "fold").split("/")[-1]
        )

        if batch_idx < 5:  # 仅保存前几个 batch，防止数据量过大
            dump_all_feature_maps(
                model=self.model,
                video=video,
                video_info=batch.get("info", None),
                attn_map=attn_map,
                save_root=f"{self.save_root}/test_all_feature_maps/{fold}/batch_{batch_idx}",
                include_types=(torch.nn.Conv3d),
                include_name_contains=("conv_c",), # 因为se 出来的是b, c, 1,1,1， 所以不保存se的中间特征
                resize_to=(256, 256),  # 指定输出大小
                resize_mode="bilinear",
            )

        return probs, logits
    
    def on_test_epoch_end(self) -> None:
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=(
                getattr(self.logger, "root_dir", "fold").split("/")[-1]
                if self.logger
                else "fold"
            ),
            save_path=self.save_root,
            num_class=self.num_classes,
        )
        logger.info("test epoch end")

    # ------------------- optimizer/scheduler -------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Lightning 里 estimated_stepping_batches 可能在早期不可用，做个稳健 fallback
        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000  # 安全兜底，后续也可换 OneCycleLR

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"},
        }
