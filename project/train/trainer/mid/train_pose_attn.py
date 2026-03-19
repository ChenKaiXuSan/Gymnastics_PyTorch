"""
File: train.py
Project: project
Created Date: 2023-10-19 02:29:47
Author: chenkaixu
-----
Comment:
 This file is the train/val/test process for the Pose Attention model.


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

from project.models.pose_fusion_res_3dcnn import PoseFusionRes3DCNN
from project.utils.helper import save_helper

from project.utils.save_CAM import dump_all_feature_maps

logger = logging.getLogger(__name__)


class PoseAttnTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()  # 先保存，方便 ckpt/repro

        self.img_size = getattr(hparams.data, "img_size", 224)

        self.lr = getattr(hparams.loss, "lr", 1e-4)
        self.num_classes = getattr(hparams.model, "model_class_num", 3)

        # * ablation: loss components
        self.lambda_list = list(
            getattr(hparams.loss, "lambda_list", [0.25, 0.5, 0.75, 1.0])
        )
        self.w_bg = float(getattr(hparams.loss, "w_bg", 0.2))
        self.w_temp = float(getattr(hparams.loss, "w_temp", 0.05))

        self.loss_selection = list(
            getattr(
                hparams.loss,
                "selection",
                ["cls", "attn_loss", "bg", "tmp"],
            )
        )

        logger.info(f"Using loss selection: {self.loss_selection}")

        # define model
        self.model = PoseFusionRes3DCNN(hparams)

        # metrics（torchmetrics 多数支持 logits/probs，内部会做 argmax）
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self.save_root = getattr(hparams.train, "log_path", "./logs")

    # ------------------- small helpers -------------------
    @staticmethod
    def _resize_3d(x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)

    @staticmethod
    def _bce_dice_loss(
        logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3, 4))
        denom = p.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + eps
        dice = 1.0 - (2.0 * inter + eps) / denom
        return bce + dice.mean()

    @staticmethod
    def _temporal_tv_l1(prob: torch.Tensor) -> torch.Tensor:
        # prob: (B,C,T,H,W) in [0,1]
        if prob.size(2) <= 1:
            return prob.new_zeros(())
        return (prob[:, :, 1:] - prob[:, :, :-1]).abs().mean()

    def _compute_attn_losses(
        self,
        logits: torch.Tensor,  # (B,C,T,H,W)
        label: torch.Tensor,  # (B,C,T,H,W) in [0,1]
        side_preds: list[torch.Tensor],  # 每层侧头 logits: (B,J,Ti,Hi,Wi)
        doctor_hm: torch.Tensor,  # (B,J,T,H,W) in [0,1]
    ) -> dict[str, torch.Tensor]:
        loss_attn_total = doctor_hm.new_zeros(())
        loss_bg_total = doctor_hm.new_zeros(())
        loss_tmp_total = doctor_hm.new_zeros(())

        if "cls" not in self.loss_selection:
            assert False, "Classification loss must be included in loss selection."
        if "attn_loss" not in self.loss_selection and len(side_preds) > 0:
            self.lambda_list = [0.0 for _ in side_preds]
        if "bg" not in self.loss_selection:
            self.w_bg = 0.0
        if "tmp" not in self.loss_selection:
            self.w_temp = 0.0

        loss_cls = F.cross_entropy(logits, label.long())

        for i, Pi in enumerate(side_preds):
            Ti, Hi, Wi = Pi.shape[2:]
            Ai = self._resize_3d(doctor_hm, (Ti, Hi, Wi))

            attn_loss = self._bce_dice_loss(Pi, Ai)

            # 背景抑制：并集 → 背景
            A_union = Ai.max(dim=1, keepdim=True).values
            A_bg = (1.0 - A_union).clamp(0, 1)
            P_max = Pi.max(dim=1, keepdim=True).values
            bg_loss = F.binary_cross_entropy_with_logits(
                P_max, torch.zeros_like(P_max), weight=A_bg
            )

            # 时间平滑（在 prob 上）
            P_sig = torch.sigmoid(Pi)
            tmp_loss = self._temporal_tv_l1(P_sig)

            lam = self.lambda_list[i]
            loss_attn_total = loss_attn_total + lam * attn_loss
            loss_bg_total = loss_bg_total + self.w_bg * bg_loss
            loss_tmp_total = loss_tmp_total + self.w_temp * tmp_loss

        return {
            "cls": loss_cls,
            "attn": loss_attn_total,
            "bg": loss_bg_total,
            "tmp": loss_tmp_total,
        }

    # ------------------- training / validation -------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        out = self.model(video, attn_map, return_aux=True)
        logits, aux = out if isinstance(out, tuple) else (out, {"side_preds": []})

        attn_loss = self._compute_attn_losses(
            logits=logits,
            label=labels,
            side_preds=aux.get("side_preds", []),
            doctor_hm=attn_map,
        )

        loss_total = (
            attn_loss["cls"] + attn_loss["attn"] + attn_loss["bg"] + attn_loss["tmp"]
        )

        # logging
        self.log("train/loss", loss_total, on_step=True, on_epoch=True, batch_size=B)
        self.log(
            "train/loss_cls",
            attn_loss["cls"],
            on_step=True,
            on_epoch=True,
            batch_size=B,
        )
        if len(aux.get("side_preds", [])) > 0:
            self.log(
                "train/loss_attn",
                attn_loss["attn"],
                on_step=True,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "train/loss_bg",
                attn_loss["bg"],
                on_step=True,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "train/loss_tmp",
                attn_loss["tmp"],
                on_step=True,
                on_epoch=True,
                batch_size=B,
            )

        probs = torch.softmax(logits, dim=1)

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
            f"train loss: {loss_total.item():.4f} "
            f"(cls {attn_loss['cls'].item():.4f} | attn {attn_loss['attn'].item():.4f} | "
            f"bg {attn_loss['bg'].item():.4f} | tmp {attn_loss['tmp'].item():.4f})"
        )
        return loss_total

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        out = self.model(video, attn_map, return_aux=True)
        logits, aux = out if isinstance(out, tuple) else (out, {"side_preds": []})

        probs = torch.softmax(logits, dim=1)

        attn_loss = self._compute_attn_losses(
            logits=logits,
            label=labels,
            side_preds=aux.get("side_preds", []),
            doctor_hm=attn_map,
        )

        loss_total = (
            attn_loss["cls"] + attn_loss["attn"] + attn_loss["bg"] + attn_loss["tmp"]
        )

        # 建议验证只 on_epoch 记录，减少噪声
        self.log("val/loss", loss_total, on_step=False, on_epoch=True, batch_size=B)
        self.log(
            "val/loss_cls", attn_loss["cls"], on_step=False, on_epoch=True, batch_size=B
        )
        if len(aux.get("side_preds", [])) > 0:
            self.log(
                "val/loss_attn",
                attn_loss["attn"],
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "val/loss_bg",
                attn_loss["bg"],
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                "val/loss_tmp",
                attn_loss["tmp"],
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )

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
            f"val loss: {loss_total.item():.4f} "
            f"(cls {attn_loss['cls'].item():.4f} | attn {attn_loss['attn'].item():.4f} | "
            f"bg {attn_loss['bg'].item():.4f} | tmp {attn_loss['tmp'].item():.4f})"
        )
        return {"val_loss": loss_total}

    # ------------------- testing -------------------
    def on_test_start(self) -> None:
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []
        logger.info("test start")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video: torch.Tensor = batch["video"]
        attn_map: torch.Tensor = batch["attn_map"]
        labels: torch.Tensor = batch["label"].long()
        B = video.size(0)

        out = self.model(video, attn_map, return_aux=False)
        logits, aux = out

        attn_loss = self._compute_attn_losses(
            logits=logits,
            label=labels,
            side_preds=aux.get("side_preds", []),
            doctor_hm=attn_map,
        )

        loss_total = (
            attn_loss["cls"] + attn_loss["attn"] + attn_loss["bg"] + attn_loss["tmp"]
        )

        self.log("test/loss", loss_total, on_step=False, on_epoch=True, batch_size=B)

        probs = torch.softmax(logits, dim=1)
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

        # save feature maps for CAM visualization
        # * only dump for first 10 batches to save space
        fold = (
            getattr(self.logger, "root_dir", "fold").split("/")[-1]
        ) if self.logger else "fold"

        if batch_idx < 10:
            dump_all_feature_maps(
                model=self.model,
                video=video,
                video_info=batch.get("info", None),
                attn_map=attn_map,
                save_root=self.save_root + f"/test_all_feature_maps/{fold}/batch_{batch_idx}",
                include_types=(torch.nn.Conv3d, torch.nn.Linear),
                include_name_contains=("conv_c", "rgb_conv", "attn_conv", "gate_conv2"),  # 只保存部分层
                exclude_name_contains=("proj", "head"),  # 排除分类 head
                resize_to=(self.img_size, self.img_size),
                resize_mode="bilinear",
            )

        return {"probs": probs, "logits": logits}

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
