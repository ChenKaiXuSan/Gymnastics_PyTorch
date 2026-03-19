#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_velocity_confidence_proxy(pose: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build confidence proxy from per-joint temporal velocity.

    Args:
        pose: (B, T, J, 3)
    Returns:
        confidence: (B, T, J, 1), larger means more stable.
    """
    if pose.ndim != 4 or pose.shape[-1] != 3:
        raise ValueError(f"Expected pose shape (B,T,J,3), got {tuple(pose.shape)}")

    vel = torch.zeros_like(pose)
    vel[:, 1:] = pose[:, 1:] - pose[:, :-1]
    vel_mag = torch.norm(vel, dim=-1, keepdim=True)  # (B,T,J,1)

    # Robust per-sample scaling; faster motion -> lower confidence.
    scale = vel_mag.median(dim=1, keepdim=True).values.median(dim=2, keepdim=True).values
    conf = torch.exp(-vel_mag / (scale + eps))
    return torch.clamp(conf, 0.0, 1.0)


class ViewGating(nn.Module):
    """Joint-wise uncertainty-aware view gating."""

    def __init__(
        self,
        hidden_dim: int = 128,
        use_conf: bool = True,
        predict_logvar: bool = False,
    ) -> None:
        super().__init__()
        self.use_conf = use_conf
        self.predict_logvar = predict_logvar

        in_dim = 9  # pL(3), pR(3), diff(3)
        if use_conf:
            in_dim += 2  # cL(1), cR(1)

        out_dim = 2 if predict_logvar else 1  # alpha, optional logvar
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        p_left: torch.Tensor,
        p_right: torch.Tensor,
        c_left: Optional[torch.Tensor] = None,
        c_right: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            p_left, p_right: (B,T,J,3)
            c_left, c_right: (B,T,J,1)
        Returns:
            p0: (B,T,J,3)
            alpha: (B,T,J,1)
            logvar: (B,T,J,1) or None
        """
        if p_left.shape != p_right.shape:
            raise ValueError(f"Left/right pose shape mismatch: {p_left.shape} vs {p_right.shape}")

        diff = p_left - p_right
        feats = [p_left, p_right, diff]

        if self.use_conf:
            if c_left is None or c_right is None:
                raise ValueError("use_conf=True requires c_left and c_right")
            feats.extend([c_left, c_right])

        x = torch.cat(feats, dim=-1)
        out = self.mlp(x)

        if self.predict_logvar:
            alpha = torch.sigmoid(out[..., :1])
            logvar = out[..., 1:2]
        else:
            alpha = torch.sigmoid(out)
            logvar = None

        p0 = alpha * p_left + (1.0 - alpha) * p_right
        return p0, alpha, logvar


class TemporalSSMBlock(nn.Module):
    """Lightweight Mamba-style temporal mixer (conv-SSM approximation)."""

    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        inner = d_model * expansion

        self.in_proj = nn.Linear(d_model, inner)
        self.dw_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner,
        )
        self.out_proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,D)"""
        h = self.norm(x)
        h = self.in_proj(h)
        h = h.transpose(1, 2)  # (B,D,T)
        h = self.dw_conv(h)
        h = h.transpose(1, 2)  # (B,T,D)
        h = self.act(h)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h


class SSMRefiner(nn.Module):
    """Temporal pose refiner with residual correction."""

    def __init__(
        self,
        num_joints: int,
        d_model: int = 256,
        n_layers: int = 4,
        expansion: int = 2,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_proj = nn.Linear(3 * num_joints, d_model)
        self.blocks = nn.ModuleList(
            [TemporalSSMBlock(d_model, expansion=expansion, kernel_size=kernel_size) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, 3 * num_joints)

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        """p0: (B,T,J,3) -> p_hat: (B,T,J,3)"""
        bsz, t, joints, dim = p0.shape
        if joints != self.num_joints or dim != 3:
            raise ValueError(
                f"Expected (B,T,{self.num_joints},3), got {tuple(p0.shape)}"
            )

        x = p0.reshape(bsz, t, 3 * joints)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        delta = self.out_proj(x).reshape(bsz, t, joints, 3)
        return p0 + delta


class FusionSSM(nn.Module):
    """Uncertainty-aware view fusion + temporal SSM refinement."""

    def __init__(
        self,
        num_joints: int,
        d_model: int = 256,
        n_layers: int = 4,
        use_conf: bool = True,
        predict_logvar: bool = False,
    ) -> None:
        super().__init__()
        self.use_conf = use_conf
        self.gating = ViewGating(
            hidden_dim=max(64, d_model // 2),
            use_conf=use_conf,
            predict_logvar=predict_logvar,
        )
        self.refiner = SSMRefiner(num_joints=num_joints, d_model=d_model, n_layers=n_layers)

    def forward(
        self,
        p_left: torch.Tensor,
        p_right: torch.Tensor,
        c_left: Optional[torch.Tensor] = None,
        c_right: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with keys: p_hat, p0, alpha, and optional logvar."""
        if self.use_conf and (c_left is None or c_right is None):
            c_left = build_velocity_confidence_proxy(p_left)
            c_right = build_velocity_confidence_proxy(p_right)

        p0, alpha, logvar = self.gating(p_left, p_right, c_left, c_right)
        p_hat = self.refiner(p0)

        out: Dict[str, torch.Tensor] = {"p_hat": p_hat, "p0": p0, "alpha": alpha}
        if logvar is not None:
            out["logvar"] = logvar
        return out


@dataclass
class PoseLossWeights:
    mpjpe: float = 1.0
    bone: float = 0.2
    vel: float = 0.05
    acc: float = 0.02
    agree: float = 0.1
    bone_stab: float = 0.05


class PoseRefineLoss(nn.Module):
    """Loss bundle for supervised + self-supervised adaptation."""

    def __init__(
        self,
        bone_edges: Optional[Sequence[Tuple[int, int]]] = None,
        weights: Optional[PoseLossWeights] = None,
    ) -> None:
        super().__init__()
        self.bone_edges = list(bone_edges) if bone_edges is not None else []
        self.w = weights or PoseLossWeights()

    @staticmethod
    def _temporal_velocity(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]

    @staticmethod
    def _temporal_acceleration(x: torch.Tensor) -> torch.Tensor:
        return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]

    def _bone_lengths(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,J,3) -> (B,T,E)
        if not self.bone_edges:
            return x.new_zeros((x.shape[0], x.shape[1], 0))

        segs: List[torch.Tensor] = []
        for u, v in self.bone_edges:
            seg = torch.norm(x[..., u, :] - x[..., v, :], dim=-1, keepdim=True)
            segs.append(seg)
        return torch.cat(segs, dim=-1)

    def forward(
        self,
        p_hat: torch.Tensor,
        p_gt: Optional[torch.Tensor] = None,
        p_left: Optional[torch.Tensor] = None,
        p_right: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified forward for training loops.

        - If p_gt is provided: supervised loss
        - Else requires p_left/p_right/alpha: self-supervised loss
        """
        if p_gt is not None:
            return self.supervised(p_hat=p_hat, p_gt=p_gt, logvar=logvar)
        if p_left is None or p_right is None or alpha is None:
            raise ValueError("Self-supervised mode requires p_left, p_right and alpha")
        return self.self_supervised(p_hat=p_hat, p_left=p_left, p_right=p_right, alpha=alpha)

    def supervised(
        self,
        p_hat: torch.Tensor,
        p_gt: torch.Tensor,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if logvar is None:
            l_mpjpe = F.l1_loss(p_hat, p_gt)
        else:
            # heteroscedastic regression
            sq = (p_hat - p_gt).pow(2).mean(dim=-1, keepdim=True)
            l_mpjpe = torch.mean(torch.exp(-logvar) * sq + logvar)

        if self.bone_edges:
            b_hat = self._bone_lengths(p_hat)
            b_gt = self._bone_lengths(p_gt)
            l_bone = F.l1_loss(b_hat, b_gt)
        else:
            l_bone = p_hat.new_tensor(0.0)

        vel = self._temporal_velocity(p_hat)
        acc = self._temporal_acceleration(p_hat)
        l_vel = vel.abs().mean() if vel.numel() > 0 else p_hat.new_tensor(0.0)
        l_acc = acc.abs().mean() if acc.numel() > 0 else p_hat.new_tensor(0.0)

        total = (
            self.w.mpjpe * l_mpjpe
            + self.w.bone * l_bone
            + self.w.vel * l_vel
            + self.w.acc * l_acc
        )
        return {
            "loss": total,
            "loss/mpjpe": l_mpjpe,
            "loss/bone": l_bone,
            "loss/vel": l_vel,
            "loss/acc": l_acc,
        }

    def self_supervised(
        self,
        p_hat: torch.Tensor,
        p_left: torch.Tensor,
        p_right: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # agreement: alpha decides which view to trust more
        l_agree = (alpha * (p_hat - p_left).abs() + (1.0 - alpha) * (p_hat - p_right).abs()).mean()

        vel = self._temporal_velocity(p_hat)
        acc = self._temporal_acceleration(p_hat)
        l_vel = vel.abs().mean() if vel.numel() > 0 else p_hat.new_tensor(0.0)
        l_acc = acc.abs().mean() if acc.numel() > 0 else p_hat.new_tensor(0.0)

        if self.bone_edges:
            b = self._bone_lengths(p_hat)
            b_prev = b[:, :-1]
            b_next = b[:, 1:]
            l_bone_stab = (b_next - b_prev).abs().mean() if b_next.numel() > 0 else p_hat.new_tensor(0.0)
        else:
            l_bone_stab = p_hat.new_tensor(0.0)

        total = (
            self.w.agree * l_agree
            + self.w.vel * l_vel
            + self.w.acc * l_acc
            + self.w.bone_stab * l_bone_stab
        )

        return {
            "loss": total,
            "loss/agree": l_agree,
            "loss/vel": l_vel,
            "loss/acc": l_acc,
            "loss/bone_stab": l_bone_stab,
        }
