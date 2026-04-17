#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from pytorch_lightning import LightningModule

from project.train.map_config import INDICES

logger = logging.getLogger(__name__)


# === 你需要实现的 SSM 分类多头版本 ===
class SSM(torch.nn.Module):
    def __init__(
        self,
        num_joints,
        d_model,
        n_layers,
        num_class=3,
        use_conf=True,
        predict_logvar=False,
    ):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(num_joints * 3 * 2, d_model),
            torch.nn.ReLU(),
            *[torch.nn.Linear(d_model, d_model) for _ in range(n_layers)],
            torch.nn.ReLU(),
        )
        self.head_twist = torch.nn.Linear(d_model, num_class)
        self.head_posture = torch.nn.Linear(d_model, num_class)
        self.head_relax = torch.nn.Linear(d_model, num_class)
        self.head_total = torch.nn.Linear(d_model, num_class)
        self.use_conf = use_conf
        self.predict_logvar = predict_logvar
        if use_conf:
            self.alpha_head = torch.nn.Linear(d_model, 1)
        if predict_logvar:
            self.logvar_head = torch.nn.Linear(d_model, num_joints * 3)

    def forward(self, p_left, p_right):
        # p_left, p_right: (B,T,J,3)
        B, T, J, C = p_left.shape
        x = torch.cat([p_left, p_right], dim=-1)  # (B,T,J,6)
        x = x.reshape(B, T, -1)  # (B,T,J*6)
        x = x.mean(dim=1)  # (B, J*6)
        feat = self.backbone(x)  # (B, d_model)
        out = {
            "twist": self.head_twist(feat),
            "posture": self.head_posture(feat),
            "relax": self.head_relax(feat),
            "total": self.head_total(feat),
        }
        if self.use_conf:
            out["alpha"] = torch.sigmoid(self.alpha_head(feat))
        if self.predict_logvar:
            out["logvar"] = self.logvar_head(feat)
        return out


class PoseLossWeights:
    def __init__(
        self, mpjpe=1.0, bone=0.2, vel=0.05, acc=0.02, agree=0.1, bone_stab=0.05
    ):
        self.mpjpe = mpjpe
        self.bone = bone
        self.vel = vel
        self.acc = acc
        self.agree = agree
        self.bone_stab = bone_stab


class PoseRefineLoss(torch.nn.Module):
    def __init__(self, bone_edges, weights: PoseLossWeights):
        super().__init__()
        self.bone_edges = bone_edges
        self.weights = weights

    def forward(
        self, p_hat, p_gt=None, logvar=None, p_left=None, p_right=None, alpha=None
    ):
        # 这里只实现mpjpe损失，其他损失项可按需补充
        loss = 0.0
        loss_dict = {}
        if p_gt is not None:
            mpjpe = torch.norm(p_hat - p_gt, dim=-1).mean()
            loss += self.weights.mpjpe * mpjpe
            loss_dict["mpjpe"] = mpjpe
        # 其他损失项（骨长、速度、加速度等）可在此补充
        loss_dict["loss"] = loss
        return loss_dict


class SSMTrainer(LightningModule):
    """Pose fusion trainer using uncertainty-aware gating + SSM refinement."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(hparams.loss, "lr", 1e-4))
        self.weight_decay = float(getattr(hparams.loss, "weight_decay", 1e-4))

        model_cfg = getattr(hparams, "model", None)
        d_model = int(getattr(model_cfg, "d_model", 256))
        n_layers = int(getattr(model_cfg, "n_layers", 4))
        use_conf = bool(getattr(model_cfg, "use_conf", True))
        predict_logvar = bool(getattr(model_cfg, "predict_logvar", False))

        self.model = SSM(
            num_joints=len(INDICES),
            d_model=d_model,
            n_layers=n_layers,
            use_conf=use_conf,
            predict_logvar=predict_logvar,
        )

        weights = PoseLossWeights(
            mpjpe=float(getattr(hparams.loss, "lambda_mpjpe", 1.0)),
            bone=float(getattr(hparams.loss, "lambda_bone", 0.2)),
            vel=float(getattr(hparams.loss, "lambda_vel", 0.05)),
            acc=float(getattr(hparams.loss, "lambda_acc", 0.02)),
            agree=float(getattr(hparams.loss, "lambda_agree", 0.1)),
            bone_stab=float(getattr(hparams.loss, "lambda_bone_stab", 0.05)),
        )
        self.loss_fn = PoseRefineLoss(
            bone_edges=None,  # 可根据需要构建骨骼边列表传入
            weights=weights,
        )
        self.save_root = str(getattr(hparams, "log_path", "./logs"))
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir: Path = Path(self.save_root) / "pose_analysis"

    @staticmethod
    def _require_pose(batch: Dict[str, Any], path: Sequence[str]) -> torch.Tensor:
        cur: Any = batch
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                raise KeyError(f"Missing batch key path: {'/'.join(path)}")
            cur = cur[key]

        if not isinstance(cur, torch.Tensor):
            raise TypeError(f"Expected tensor at {'/'.join(path)}, got {type(cur)}")
        if cur.ndim != 4 or cur.shape[-1] != 3:
            raise ValueError(
                f"Expected pose tensor shape (B,T,J,3) at {'/'.join(path)}, got {tuple(cur.shape)}"
            )
        return cur.float()

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        p_left = self._require_pose(batch, ("kpt3d_sam", "cam1"))
        p_right = self._require_pose(batch, ("kpt3d_sam", "cam2"))

        out = self.model(p_left=p_left, p_right=p_right)
        p_hat = out["p_hat"]
        alpha = out["alpha"]
        logvar = out.get("logvar", None)

        has_gt = isinstance(batch.get("kpt3d_gt", None), torch.Tensor)
        if has_gt:
            p_gt = batch["kpt3d_gt"].float()
            if p_gt.ndim != 4 or p_gt.shape[-1] != 3:
                raise ValueError(
                    f"Expected kpt3d_gt shape (B,T,J,3), got {tuple(p_gt.shape)}"
                )
            loss_dict = self.loss_fn(p_hat=p_hat, p_gt=p_gt, logvar=logvar)
            mpjpe = torch.norm(p_hat - p_gt, dim=-1).mean()
            self.log(
                f"{stage}/mpjpe",
                mpjpe,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=p_hat.shape[0],
            )
        else:
            loss_dict = self.loss_fn(
                p_hat=p_hat,
                p_left=p_left,
                p_right=p_right,
                alpha=alpha,
            )

        loss = loss_dict["loss"]
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage != "train"),
            batch_size=p_hat.shape[0],
        )

        for k, v in loss_dict.items():
            if k == "loss":
                continue
            self.log(
                f"{stage}/{k}",
                v,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )

        self.log(
            f"{stage}/alpha_mean",
            alpha.mean(),
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/alpha_std",
            alpha.std(),
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        return loss

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_test_start(self) -> None:
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("FusionSSM test start")

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        p_left = self._require_pose(batch, ("kpt3d_sam", "cam1"))
        p_right = self._require_pose(batch, ("kpt3d_sam", "cam2"))

        out = self.model(p_left=p_left, p_right=p_right)
        p_hat = out["p_hat"]
        p0 = out["p0"]
        alpha = out["alpha"]
        logvar = out.get("logvar", None)

        has_gt = isinstance(batch.get("kpt3d_gt", None), torch.Tensor)
        if has_gt:
            p_gt = batch["kpt3d_gt"].float()
            loss_dict = self.loss_fn(p_hat=p_hat, p_gt=p_gt, logvar=logvar)
            mpjpe = torch.norm(p_hat - p_gt, dim=-1).mean()
            self.log(
                "test/mpjpe",
                mpjpe,
                on_step=False,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
        else:
            p_gt = None
            loss_dict = self.loss_fn(
                p_hat=p_hat,
                p_left=p_left,
                p_right=p_right,
                alpha=alpha,
            )

        loss = loss_dict["loss"]
        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, batch_size=p_hat.shape[0]
        )
        self.log(
            "test/alpha_mean",
            alpha.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            "test/alpha_std",
            alpha.std(),
            on_step=False,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )

        pack: Dict[str, Any] = {
            "p_hat": p_hat.detach().cpu(),
            "p0": p0.detach().cpu(),
            "alpha": alpha.detach().cpu(),
            "p_left": p_left.detach().cpu(),
            "p_right": p_right.detach().cpu(),
        }
        if p_gt is not None:
            pack["label"] = p_gt.detach().cpu()
        if logvar is not None:
            pack["logvar"] = logvar.detach().cpu()
        if "meta" in batch:
            pack["meta"] = batch["meta"]
        self.test_outputs.append(pack)
        return loss

    def on_test_epoch_end(self) -> None:
        if not hasattr(self, "test_outputs") or len(self.test_outputs) == 0:
            logger.warning("No test outputs to save.")
            return

        fold = (
            getattr(self.logger, "root_dir", "fold").split("/")[-1]
            if self.logger is not None
            else "fold"
        )
        save_dir = self.test_save_dir

        payload: Dict[str, Any] = {
            "p_hat": torch.cat([x["p_hat"] for x in self.test_outputs], dim=0),
            "p0": torch.cat([x["p0"] for x in self.test_outputs], dim=0),
            "alpha": torch.cat([x["alpha"] for x in self.test_outputs], dim=0),
            "p_left": torch.cat([x["p_left"] for x in self.test_outputs], dim=0),
            "p_right": torch.cat([x["p_right"] for x in self.test_outputs], dim=0),
        }

        if all("label" in x for x in self.test_outputs):
            payload["label"] = torch.cat([x["label"] for x in self.test_outputs], dim=0)
        if all("logvar" in x for x in self.test_outputs):
            payload["logvar"] = torch.cat(
                [x["logvar"] for x in self.test_outputs], dim=0
            )
        if any("meta" in x for x in self.test_outputs):
            payload["meta"] = [x.get("meta", None) for x in self.test_outputs]

        save_file = save_dir / f"{fold}_pose_outputs.pt"
        torch.save(payload, save_file)
        logger.info("Saved pose predictions/labels to %s", save_file)

    def configure_optimizers(self):
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
