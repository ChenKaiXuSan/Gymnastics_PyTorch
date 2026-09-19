"""Plain temporal-network baseline (ablation B1) for the learned comparison.

The model shares the data pipeline, the self-supervised objectives, the
optimizer and the checkpoint-selection score of the rotation-aware model, but
none of its structural assumptions:

* no shared per-view encoder or trunk/rotation conditioning;
* no view-order symmetrization (face and side are concatenated as-is);
* B1: no bounded residual, the network regresses an unbounded correction on
  the quality-weighted base pose (only the ``minimal_residual`` objective
  limits it);
* B2: the same network with the A6 residual bound (``tanh`` times 5 cm), so
  the architecture and the bound can be separated.

They are the "would any temporal network do?" controls. The base pose is kept
as the quality-weighted mean so that ``base_kpts`` (Table 1 row A3) and the
residual objective stay identical to the rotation-aware runs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base_fusion import quality_weighted_fusion
from .camera import CameraFeatureBundle
from .config import SkeletonSpec
from .features import DisagreementFeatures, FeatureBundle
from .model import DilatedResidualTCN, FusionOutput, RotationAwareFusionModel
from .trunk import extract_trunk_features


class PlainTemporalFusionModel(nn.Module):
    """Concatenate both views per joint and run one dilated TCN over time."""

    architecture = "plain_tcn"

    def __init__(
        self,
        spec: SkeletonSpec,
        *,
        hidden_channels: int = 128,
        direct_regression: bool = False,
        max_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.joint_count = len(spec.joint_names)
        self.hidden_channels = int(hidden_channels)
        self.direct_regression = bool(direct_regression)
        self.max_delta = float(max_delta)
        if self.max_delta < 0 or (self.direct_regression and self.max_delta > 0):
            raise ValueError("max_delta must be non-negative and requires residual mode")
        # per joint: face xyz, side xyz, face valid, side valid
        self.joint_encoder = nn.Sequential(
            nn.Linear(8, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.temporal = DilatedResidualTCN(self.hidden_channels)
        self.head = nn.Linear(self.hidden_channels, 3)

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
        dt: float | Tensor = 1.0,
        camera_features: CameraFeatureBundle | None = None,
    ) -> FusionOutput:
        if camera_features is not None:
            raise ValueError("the plain temporal baseline does not accept camera features")
        if (valid_face is None) != (valid_side is None):
            raise ValueError("valid_face and valid_side must be provided together")
        if valid_face is None or valid_side is None:
            valid_face, valid_side = face_features.pose.valid, side_features.pose.valid
        face, side, valid_face, valid_side = RotationAwareFusionModel._safe_inputs(
            self, face, side, valid_face, valid_side
        )
        temporal_valid = RotationAwareFusionModel._temporal_valid_mask(
            temporal_valid, valid_face, valid_side
        )
        dt = RotationAwareFusionModel._physical_dt(dt, face, temporal_valid)
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
        batch, frames, joints, _ = face.shape
        tokens = torch.cat(
            (
                face,
                side,
                valid_face.to(face.dtype)[..., None],
                valid_side.to(face.dtype)[..., None],
            ),
            dim=-1,
        )
        encoded = self.joint_encoder(tokens)  # [B, T, J, H]
        sequence = encoded.permute(0, 2, 3, 1).reshape(batch * joints, self.hidden_channels, frames)
        mask = temporal_valid[:, None, :].expand(batch, joints, frames).reshape(
            batch * joints, 1, frames
        )
        hidden = self.temporal(sequence, mask)
        hidden = hidden.reshape(batch, joints, self.hidden_channels, frames).permute(0, 3, 1, 2)
        raw = self.head(hidden)  # [B, T, J, 3]
        if self.max_delta > 0:
            raw = torch.tanh(raw) * self.max_delta
        raw = torch.where(base.valid[..., None], raw, torch.zeros_like(raw))
        fused = raw if self.direct_regression else base.points + raw
        fused = torch.where(base.valid[..., None], fused, torch.zeros_like(fused))
        fused_trunk = extract_trunk_features(fused, base.valid, self.spec, dt=dt)
        return FusionOutput(
            fused_kpts=fused,
            base_kpts=base.points,
            delta_kpts=fused - base.points,
            valid=base.valid,
            fused_theta=fused_trunk.angle,
            fused_theta_valid=fused_trunk.angle_valid,
            fused_r_pt=fused_trunk.rotation,
            fused_r_pt_valid=fused_trunk.rotation_valid,
        )
