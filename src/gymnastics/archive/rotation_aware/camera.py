"""Typed optional camera conditioning for rotation-aware fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CameraConditioningConfig:
    """Fixed feature dimensions and conditioning mechanism."""

    global_channels: int
    joint_channels: int
    mode: Literal["additive", "film"] = "film"

    def __post_init__(self) -> None:
        if self.global_channels < 1 or self.joint_channels < 1:
            raise ValueError("camera feature channels must be positive")
        if self.mode not in {"additive", "film"}:
            raise ValueError("camera conditioning mode must be additive or film")


@dataclass(frozen=True)
class CameraFeatureBundle:
    """Batched global and frame-joint camera observations."""

    global_features: Tensor
    joint_features: Tensor
    valid: Tensor

    def validated(
        self,
        *,
        batch: int,
        frames: int,
        joints: int,
        global_channels: int,
        joint_channels: int,
    ) -> CameraFeatureBundle:
        if self.global_features.shape != (batch, global_channels):
            raise ValueError(
                "camera global features must have shape "
                f"[B,{global_channels}]"
            )
        if self.joint_features.shape != (
            batch,
            frames,
            joints,
            joint_channels,
        ):
            raise ValueError(
                "camera joint features must have shape "
                f"[B,T,J,{joint_channels}]"
            )
        if self.valid.shape != (batch, frames, joints):
            raise ValueError("camera feature validity must have shape [B,T,J]")
        if not torch.isfinite(self.global_features).all():
            raise ValueError("camera global features must be finite")
        valid = self.valid.bool()
        joint_finite = torch.isfinite(self.joint_features).all(dim=-1)
        if not torch.all(joint_finite | ~valid):
            raise ValueError("valid camera joint features must be finite")
        safe_joint = torch.where(
            valid[..., None],
            torch.where(
                torch.isfinite(self.joint_features),
                self.joint_features,
                torch.zeros_like(self.joint_features),
            ),
            torch.zeros_like(self.joint_features),
        )
        return CameraFeatureBundle(
            global_features=self.global_features,
            joint_features=safe_joint,
            valid=valid,
        )


class CameraConditioner(nn.Module):
    """Condition existing motion features while starting as an exact identity."""

    def __init__(
        self,
        config: CameraConditioningConfig,
        *,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.global_encoder = nn.Sequential(
            nn.Linear(config.global_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(config.joint_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        output_channels = (
            hidden_channels if config.mode == "additive" else 2 * hidden_channels
        )
        self.output = nn.Linear(hidden_channels, output_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        fused_features: Tensor,
        camera_features: CameraFeatureBundle,
        effective_mask: Tensor,
    ) -> Tensor:
        if fused_features.ndim != 4:
            raise ValueError("fused features must have shape [B,T,J,C]")
        batch, frames, joints, _ = fused_features.shape
        checked = camera_features.validated(
            batch=batch,
            frames=frames,
            joints=joints,
            global_channels=self.config.global_channels,
            joint_channels=self.config.joint_channels,
        )
        global_features = checked.global_features.to(
            device=fused_features.device, dtype=fused_features.dtype
        )
        joint_features = checked.joint_features.to(
            device=fused_features.device, dtype=fused_features.dtype
        )
        camera_valid = checked.valid.to(device=fused_features.device)
        camera_encoded = self.global_encoder(global_features)[:, None, None]
        camera_encoded = camera_encoded + self.joint_encoder(joint_features)
        usable = effective_mask.bool() & camera_valid
        camera_encoded = torch.where(
            usable[..., None], camera_encoded, torch.zeros_like(camera_encoded)
        )
        conditioned = self.output(camera_encoded)
        if self.config.mode == "additive":
            result = fused_features + conditioned
        else:
            gamma, beta = conditioned.chunk(2, dim=-1)
            result = fused_features * (1.0 + torch.tanh(gamma)) + beta
        return torch.where(
            effective_mask[..., None], result, torch.zeros_like(result)
        )
