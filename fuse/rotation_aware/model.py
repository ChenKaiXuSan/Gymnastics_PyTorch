"""Swap-invariant residual TCN for canonical multi-view keypoint fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .base_fusion import quality_weighted_fusion
from .config import SkeletonSpec
from .features import DisagreementFeatures, FeatureBundle
from .trunk import extract_trunk_features


@dataclass(frozen=True)
class FusionOutput:
    """Fused canonical keypoints and kinematics derived from those keypoints."""

    fused_kpts: Tensor
    base_kpts: Tensor
    delta_kpts: Tensor
    valid: Tensor
    fused_theta: Tensor
    fused_theta_valid: Tensor
    fused_r_pt: Tensor
    fused_r_pt_valid: Tensor


class ResidualTCNBlock(nn.Module):
    """A same-length non-causal dilated temporal residual block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        if mask.shape != (values.shape[0], 1, values.shape[2]):
            raise ValueError("TCN mask must have shape [B, 1, T]")
        mask = mask.to(dtype=values.dtype)
        values = values * mask
        residual = values
        values = self.conv1(values) * mask
        values = self.activation(values) * mask
        values = self.conv2(values) * mask
        return self.activation(values + residual) * mask


class DilatedResidualTCN(nn.Module):
    """Six non-causal residual blocks covering short and long motion context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ResidualTCNBlock(channels, dilation) for dilation in (1, 2, 4, 8, 16, 32))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            values = block(values, mask)
        return values


class SharedViewEncoder(nn.Module):
    """Encode pose and trunk observations for either input view with one weight set."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.joint = nn.Sequential(
            nn.Linear(9, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.trunk = nn.Sequential(
            nn.Linear(14, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.activation = nn.GELU()

    def forward(
        self,
        points: Tensor,
        valid: Tensor,
        effective_mask: Tensor,
        features: FeatureBundle,
        trunk_rotation: Tensor,
        trunk_angle: Tensor,
        trunk_omega: Tensor,
        trunk_alpha: Tensor,
        trunk_valid: Tensor,
    ) -> Tensor:
        pose = features.pose
        if pose.points.shape != points.shape or pose.valid.shape != valid.shape:
            raise ValueError("FeatureBundle pose features must match the supplied points and valid mask")
        if pose.velocity.shape != points.shape or pose.velocity_valid.shape != valid.shape:
            raise ValueError("FeatureBundle velocity features must match the supplied points and valid mask")
        if features.quality.score.shape != points.shape[:2]:
            raise ValueError("FeatureBundle quality score must have shape [B, T]")
        if effective_mask.shape != valid.shape:
            raise ValueError("effective_mask must have shape [B, T, J]")

        joint_features = torch.cat(
            (
                points,
                pose.velocity.to(dtype=points.dtype),
                valid.to(dtype=points.dtype).unsqueeze(-1),
                pose.velocity_valid.to(dtype=points.dtype).unsqueeze(-1),
                features.quality.score.to(dtype=points.dtype)[..., None, None].expand(-1, -1, points.shape[2], -1),
            ),
            dim=-1,
        )
        trunk_features = torch.cat(
            (
                trunk_rotation.flatten(start_dim=-2),
                torch.sin(trunk_angle).unsqueeze(-1),
                torch.cos(trunk_angle).unsqueeze(-1),
                trunk_omega.unsqueeze(-1),
                trunk_alpha.unsqueeze(-1),
                trunk_valid.to(dtype=points.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded = self.activation(self.joint(joint_features) + self.trunk(trunk_features).unsqueeze(2))
        return torch.where(effective_mask[..., None], encoded, torch.zeros_like(encoded))


class RotationAwareFusionModel(nn.Module):
    """Fuse two canonical views with symmetric features and bounded joint residuals."""

    def __init__(
        self,
        spec: SkeletonSpec,
        *,
        hidden_channels: int = 128,
        max_delta_by_joint: float | Tensor | Sequence[float] = 0.05,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.spec = spec
        self.joint_count = len(spec.joint_names)
        self.view_encoder = SharedViewEncoder(hidden_channels)
        self.cross_encoder = nn.Sequential(
            nn.Linear(13, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.fuse_projection = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.GELU(),
        )
        self.tcn = DilatedResidualTCN(hidden_channels)
        self.delta_head = nn.Linear(hidden_channels, 3)
        limits = self._residual_limits(max_delta_by_joint)
        self.register_buffer("max_delta_by_joint", limits)

    def _residual_limits(self, value: float | Tensor | Sequence[float]) -> Tensor:
        limits = torch.as_tensor(value, dtype=torch.float32)
        if limits.ndim == 0:
            limits = limits.expand(self.joint_count).clone()
        if limits.shape != (self.joint_count,) or not torch.isfinite(limits).all() or (limits < 0).any():
            raise ValueError("max_delta_by_joint must be a finite non-negative scalar or one value per SkeletonSpec joint")
        return limits

    def _safe_inputs(self, face: Tensor, side: Tensor, valid_face: Tensor, valid_side: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if face.ndim != 4 or face.shape[-1] != 3 or side.shape != face.shape:
            raise ValueError("face and side must have equal shape [B, T, J, 3]")
        if face.shape[2] != self.joint_count:
            raise ValueError("pose joint dimension must match the supplied SkeletonSpec")
        if valid_face.shape != face.shape[:-1] or valid_side.shape != face.shape[:-1]:
            raise ValueError("valid masks must have shape [B, T, J]")
        face_valid = valid_face.bool() & torch.isfinite(face).all(dim=-1)
        side_valid = valid_side.bool() & torch.isfinite(side).all(dim=-1)
        return (
            torch.where(face_valid[..., None], face, torch.zeros_like(face)),
            torch.where(side_valid[..., None], side, torch.zeros_like(side)),
            face_valid,
            side_valid,
        )

    @staticmethod
    def _temporal_valid_mask(
        temporal_valid: Tensor | None,
        valid_face: Tensor,
        valid_side: Tensor,
    ) -> Tensor:
        if temporal_valid is None:
            return (valid_face | valid_side).any(dim=-1)
        if temporal_valid.shape != valid_face.shape[:2]:
            raise ValueError("temporal_valid must have shape [B, T]")
        return temporal_valid.bool()

    @staticmethod
    def _cross_features(cross: DisagreementFeatures, shape: tuple[int, int, int], dtype: torch.dtype) -> Tensor:
        batch, frames, joints = shape
        if cross.coordinate_abs_delta.shape != (batch, frames, joints, 3):
            raise ValueError("cross coordinate_abs_delta must have shape [B, T, J, 3]")
        if cross.coordinate_valid.shape != (batch, frames, joints) or cross.validity_abs_delta.shape != (batch, frames, joints):
            raise ValueError("cross joint masks must have shape [B, T, J]")
        frame_shapes = (
            cross.angle_abs_delta,
            cross.angle_valid,
            cross.rotation_distance,
            cross.rotation_valid,
            cross.trunk_displacement_abs_delta,
            cross.trunk_displacement_valid,
        )
        if any(value.shape != (batch, frames) for value in frame_shapes[:-2]) or cross.trunk_displacement_abs_delta.shape != (batch, frames, 3) or cross.trunk_displacement_valid.shape != (batch, frames):
            raise ValueError("cross frame features must have shapes [B, T] or [B, T, 3]")

        def expand(value: Tensor) -> Tensor:
            return value.to(dtype=dtype).unsqueeze(2).expand(-1, -1, joints, -1)

        return torch.cat(
            (
                cross.coordinate_abs_delta.to(dtype=dtype),
                cross.coordinate_valid.to(dtype=dtype).unsqueeze(-1),
                expand(cross.angle_abs_delta.unsqueeze(-1)),
                expand(cross.angle_valid.unsqueeze(-1)),
                expand(cross.rotation_distance.unsqueeze(-1)),
                expand(cross.rotation_valid.unsqueeze(-1)),
                expand(cross.trunk_displacement_abs_delta),
                expand(cross.trunk_displacement_valid.unsqueeze(-1)),
                cross.validity_abs_delta.to(dtype=dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        face: Tensor,
        side: Tensor,
        face_features: FeatureBundle,
        side_features: FeatureBundle,
        cross: DisagreementFeatures,
        valid_face: Tensor | None = None,
        valid_side: Tensor | None = None,
        temporal_valid: Tensor | None = None,
    ) -> FusionOutput:
        """Return a quality-weighted base plus a masked symmetric bounded residual."""
        if (valid_face is None) != (valid_side is None):
            raise ValueError("valid_face and valid_side must be provided together")
        if valid_face is None or valid_side is None:
            valid_face, valid_side = face_features.pose.valid, side_features.pose.valid
        face, side, valid_face, valid_side = self._safe_inputs(face, side, valid_face, valid_side)
        temporal_valid = self._temporal_valid_mask(temporal_valid, valid_face, valid_side)
        valid_face = valid_face & temporal_valid[..., None]
        valid_side = valid_side & temporal_valid[..., None]
        face = torch.where(valid_face[..., None], face, torch.zeros_like(face))
        side = torch.where(valid_side[..., None], side, torch.zeros_like(side))
        base = quality_weighted_fusion(
            face,
            side,
            valid_face,
            valid_side,
            face_features.quality.score,
            side_features.quality.score,
        )
        face_trunk = extract_trunk_features(face, valid_face, self.spec, dt=1.0)
        side_trunk = extract_trunk_features(side, valid_side, self.spec, dt=1.0)
        effective_mask = valid_face & valid_side
        face_encoded = self.view_encoder(
            face,
            valid_face,
            effective_mask,
            face_features,
            face_trunk.rotation,
            face_trunk.angle,
            face_trunk.omega,
            face_trunk.alpha,
            face_trunk.rotation_valid,
        )
        side_encoded = self.view_encoder(
            side,
            valid_side,
            effective_mask,
            side_features,
            side_trunk.rotation,
            side_trunk.angle,
            side_trunk.omega,
            side_trunk.alpha,
            side_trunk.rotation_valid,
        )
        symmetric_views = torch.cat(((face_encoded + side_encoded) * 0.5, (face_encoded - side_encoded).abs()), dim=-1)
        symmetric_views = torch.where(effective_mask[..., None], symmetric_views, torch.zeros_like(symmetric_views))
        cross_encoded = self.cross_encoder(self._cross_features(cross, face.shape[:3], face.dtype))
        cross_encoded = torch.where(effective_mask[..., None], cross_encoded, torch.zeros_like(cross_encoded))
        fused_features = self.fuse_projection(torch.cat((symmetric_views, cross_encoded), dim=-1))
        fused_features = torch.where(effective_mask[..., None], fused_features, torch.zeros_like(fused_features))
        batch, frames, joints, channels = fused_features.shape
        temporal = fused_features.permute(0, 2, 3, 1).reshape(batch * joints, channels, frames)
        tcn_mask = effective_mask.permute(0, 2, 1).reshape(batch * joints, 1, frames)
        temporal = self.tcn(temporal, tcn_mask)
        raw_delta = self.delta_head(temporal.reshape(batch, joints, channels, frames).permute(0, 3, 1, 2))
        bounded_delta = torch.tanh(raw_delta) * self.max_delta_by_joint[None, None, :, None].to(dtype=face.dtype)
        delta = torch.where(effective_mask[..., None], bounded_delta, torch.zeros_like(bounded_delta))
        fused = base.points + delta
        fused_trunk = extract_trunk_features(fused, base.valid, self.spec, dt=1.0)
        return FusionOutput(
            fused_kpts=fused,
            base_kpts=base.points,
            delta_kpts=delta,
            valid=base.valid,
            fused_theta=fused_trunk.angle,
            fused_theta_valid=fused_trunk.angle_valid,
            fused_r_pt=fused_trunk.rotation,
            fused_r_pt_valid=fused_trunk.rotation_valid,
        )
