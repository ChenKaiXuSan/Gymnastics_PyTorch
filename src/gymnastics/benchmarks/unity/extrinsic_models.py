"""Calibrated learned fusion models for the Unity benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .schema import UnityCamera


@dataclass(frozen=True)
class CalibratedPrediction:
    """One batched calibrated prediction and model diagnostics."""

    points: Tensor
    valid: Tensor
    diagnostics: Mapping[str, Tensor]


def relative_camera_rotation(cam0: UnityCamera, cam1: UnityCamera) -> np.ndarray:
    """Return the column-vector rotation that maps cam1 vectors into cam0."""
    rotation = (
        np.asarray(cam0.world_to_camera[:3, :3], dtype=np.float64)
        @ np.asarray(cam1.camera_to_world[:3, :3], dtype=np.float64)
    )
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("relative camera geometry is not a proper rotation")
    return rotation.astype(np.float32)


def _validate_rotation(rotation: Tensor, batch: int) -> Tensor:
    if rotation.ndim == 2:
        rotation = rotation.unsqueeze(0).expand(batch, -1, -1)
    if rotation.shape != (batch, 3, 3):
        raise ValueError("rotation must have shape [3,3] or [B,3,3]")
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    orthogonal = torch.matmul(rotation.transpose(-1, -2), rotation)
    determinant = torch.linalg.det(rotation)
    if (
        not torch.isfinite(rotation).all()
        or not torch.allclose(
            orthogonal,
            identity.expand_as(orthogonal),
            atol=1e-4,
            rtol=1e-4,
        )
        or not torch.allclose(
            determinant,
            torch.ones_like(determinant),
            atol=1e-4,
            rtol=1e-4,
        )
    ):
        raise ValueError("rotation must be a finite proper rotation")
    return rotation


def _validate_pose_pair(
    face: Tensor,
    side: Tensor,
    valid_face: Tensor,
    valid_side: Tensor,
    *,
    joint_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if (
        face.ndim != 4
        or face.shape[-2:] != (joint_count, 3)
        or side.shape != face.shape
    ):
        raise ValueError("pose pair must have equal shape [B,T,J,3]")
    if valid_face.shape != face.shape[:-1] or valid_side.shape != face.shape[:-1]:
        raise ValueError("pose validity must have shape [B,T,J]")
    valid_face = valid_face.bool() & torch.isfinite(face).all(dim=-1)
    valid_side = valid_side.bool() & torch.isfinite(side).all(dim=-1)
    face = torch.where(valid_face[..., None], face, torch.zeros_like(face))
    side = torch.where(valid_side[..., None], side, torch.zeros_like(side))
    return face, side, valid_face, valid_side


def _rotate_side_to_face(
    face: Tensor,
    side: Tensor,
    valid_face: Tensor,
    valid_side: Tensor,
    rotation: Tensor,
    pelvis_indices: Sequence[int],
) -> tuple[Tensor, Tensor]:
    batch = face.shape[0]
    rotation = _validate_rotation(rotation.to(face), batch)
    indices = list(pelvis_indices)
    face_pelvis = face[:, :, indices].mean(dim=2)
    side_pelvis = side[:, :, indices].mean(dim=2)
    pelvis_valid = valid_face[:, :, indices].all(dim=2) & valid_side[
        :, :, indices
    ].all(dim=2)
    centered = side - side_pelvis[:, :, None]
    rotated = torch.einsum("btjc,bdc->btjd", centered, rotation)
    aligned = rotated + face_pelvis[:, :, None]
    aligned_valid = valid_side & pelvis_valid[:, :, None]
    aligned = torch.where(aligned_valid[..., None], aligned, torch.zeros_like(aligned))
    return aligned, aligned_valid


def _velocity_magnitude(points: Tensor, valid: Tensor) -> Tensor:
    velocity = torch.zeros_like(points)
    velocity[:, 1:] = points[:, 1:] - points[:, :-1]
    velocity_valid = torch.zeros_like(valid)
    velocity_valid[:, 1:] = valid[:, 1:] & valid[:, :-1]
    return torch.where(
        velocity_valid,
        torch.linalg.vector_norm(velocity, dim=-1),
        torch.zeros_like(velocity[..., 0]),
    )


def _pair_features(
    face: Tensor,
    aligned_side: Tensor,
    valid_face: Tensor,
    valid_side: Tensor,
    rotation: Tensor,
) -> Tensor:
    batch, frames, joints, _ = face.shape
    if rotation.ndim == 2:
        rotation = rotation.unsqueeze(0).expand(batch, -1, -1)
    rotation_feature = rotation.reshape(batch, 1, 1, 9).expand(
        -1, frames, joints, -1
    )
    return torch.cat(
        (
            face,
            aligned_side,
            (face - aligned_side).abs(),
            valid_face.to(face.dtype).unsqueeze(-1),
            valid_side.to(face.dtype).unsqueeze(-1),
            _velocity_magnitude(face, valid_face).unsqueeze(-1),
            _velocity_magnitude(aligned_side, valid_side).unsqueeze(-1),
            rotation_feature.to(face),
        ),
        dim=-1,
    )


def _validity_aware_average(
    face: Tensor,
    side: Tensor,
    valid_face: Tensor,
    valid_side: Tensor,
) -> tuple[Tensor, Tensor]:
    both = valid_face & valid_side
    face_only = valid_face & ~valid_side
    side_only = valid_side & ~valid_face
    points = torch.where(
        both[..., None],
        0.5 * (face + side),
        torch.where(
            face_only[..., None],
            face,
            torch.where(side_only[..., None], side, torch.zeros_like(face)),
        ),
    )
    return points, valid_face | valid_side


class _TemporalJointEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.GELU(),
        )
        self.temporal = nn.ModuleList(
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
            for dilation in (1, 2, 4)
        )
        self.activation = nn.GELU()

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        encoded = self.input(features)
        batch, frames, joints, channels = encoded.shape
        values = encoded.permute(0, 2, 3, 1).reshape(
            batch * joints, channels, frames
        )
        mask = (
            valid.permute(0, 2, 1)
            .reshape(batch * joints, 1, frames)
            .to(values.dtype)
        )
        values = values * mask
        for convolution in self.temporal:
            values = self.activation(convolution(values) + values) * mask
        return values.reshape(batch, joints, channels, frames).permute(0, 3, 1, 2)


class ExtrinsicGateModel(nn.Module):
    """Predict a calibrated per-frame, per-joint convex fusion gate."""

    def __init__(
        self,
        *,
        joint_count: int,
        pelvis_index: int | None = None,
        pelvis_indices: Sequence[int] | None = None,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if pelvis_indices is None:
            pelvis_indices = () if pelvis_index is None else (pelvis_index,)
        pelvis_indices = tuple(int(value) for value in pelvis_indices)
        if (
            joint_count < 1
            or not pelvis_indices
            or len(set(pelvis_indices)) != len(pelvis_indices)
            or any(not 0 <= value < joint_count for value in pelvis_indices)
        ):
            raise ValueError("joint_count and pelvis indices are inconsistent")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.joint_count = joint_count
        self.pelvis_indices = pelvis_indices
        self.encoder = _TemporalJointEncoder(22, hidden_channels)
        self.output_head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        face: Tensor,
        side: Tensor,
        valid_face: Tensor,
        valid_side: Tensor,
        rotation: Tensor,
    ) -> CalibratedPrediction:
        face, side, valid_face, valid_side = _validate_pose_pair(
            face,
            side,
            valid_face,
            valid_side,
            joint_count=self.joint_count,
        )
        checked_rotation = _validate_rotation(rotation.to(face), face.shape[0])
        aligned_side, aligned_valid = _rotate_side_to_face(
            face,
            side,
            valid_face,
            valid_side,
            checked_rotation,
            self.pelvis_indices,
        )
        both = valid_face & aligned_valid
        features = _pair_features(
            face,
            aligned_side,
            valid_face,
            aligned_valid,
            checked_rotation,
        )
        gate = torch.sigmoid(self.output_head(self.encoder(features, both)))[..., 0]
        points = torch.where(
            both[..., None],
            gate[..., None] * face + (1.0 - gate[..., None]) * aligned_side,
            torch.where(
                valid_face[..., None],
                face,
                torch.where(
                    aligned_valid[..., None],
                    aligned_side,
                    torch.zeros_like(face),
                ),
            ),
        )
        valid = valid_face | aligned_valid
        return CalibratedPrediction(
            points=points,
            valid=valid,
            diagnostics={"gate": gate, "aligned_side": aligned_side},
        )


class ExtrinsicResidualTCN(nn.Module):
    """Refine a calibrated equal-average base with a bounded 3D residual."""

    def __init__(
        self,
        *,
        joint_count: int,
        pelvis_index: int | None = None,
        pelvis_indices: Sequence[int] | None = None,
        hidden_channels: int = 32,
        max_delta_m: float = 0.05,
    ) -> None:
        super().__init__()
        if not np.isfinite(max_delta_m) or max_delta_m < 0:
            raise ValueError("max_delta_m must be finite and non-negative")
        if pelvis_indices is None:
            pelvis_indices = () if pelvis_index is None else (pelvis_index,)
        pelvis_indices = tuple(int(value) for value in pelvis_indices)
        if (
            joint_count < 1
            or not pelvis_indices
            or len(set(pelvis_indices)) != len(pelvis_indices)
            or any(not 0 <= value < joint_count for value in pelvis_indices)
        ):
            raise ValueError("joint_count and pelvis indices are inconsistent")
        self.joint_count = joint_count
        self.pelvis_indices = pelvis_indices
        self.max_delta_m = float(max_delta_m)
        self.encoder = _TemporalJointEncoder(22, hidden_channels)
        self.output_head = nn.Linear(hidden_channels, 3)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        face: Tensor,
        side: Tensor,
        valid_face: Tensor,
        valid_side: Tensor,
        rotation: Tensor,
    ) -> CalibratedPrediction:
        face, side, valid_face, valid_side = _validate_pose_pair(
            face,
            side,
            valid_face,
            valid_side,
            joint_count=self.joint_count,
        )
        checked_rotation = _validate_rotation(rotation.to(face), face.shape[0])
        aligned_side, aligned_valid = _rotate_side_to_face(
            face,
            side,
            valid_face,
            valid_side,
            checked_rotation,
            self.pelvis_indices,
        )
        base, valid = _validity_aware_average(
            face, aligned_side, valid_face, aligned_valid
        )
        features = _pair_features(
            face,
            aligned_side,
            valid_face,
            aligned_valid,
            checked_rotation,
        )
        encoded = self.encoder(features, valid)
        delta = self.max_delta_m * torch.tanh(self.output_head(encoded))
        delta = torch.where(valid[..., None], delta, torch.zeros_like(delta))
        return CalibratedPrediction(
            points=base + delta,
            valid=valid,
            diagnostics={
                "delta": delta,
                "base": base,
                "aligned_side": aligned_side,
            },
        )


class LearnableTriangulationModel(nn.Module):
    """Learn view confidence for differentiable two-view algebraic DLT."""

    def __init__(self, *, hidden_channels: int = 32) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.confidence = nn.Sequential(
            nn.Linear(6, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.zeros_(self.confidence[-1].bias)

    @staticmethod
    def _projection(
        projection: Tensor, batch: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        projection = projection.to(dtype=dtype, device=device)
        if projection.ndim == 3:
            projection = projection.unsqueeze(0).expand(batch, -1, -1, -1)
        if projection.shape != (batch, 2, 3, 4):
            raise ValueError("projection must have shape [2,3,4] or [B,2,3,4]")
        if not torch.isfinite(projection).all():
            raise ValueError("projection must be finite")
        return projection

    def forward(
        self,
        pixels: Tensor,
        valid: Tensor,
        projection: Tensor,
        *,
        image_size: Tensor,
    ) -> CalibratedPrediction:
        if pixels.ndim != 5 or pixels.shape[2] != 2 or pixels.shape[-1] != 2:
            raise ValueError("pixels must have shape [B,T,2,J,2]")
        if valid.shape != pixels.shape[:-1]:
            raise ValueError("valid must have shape [B,T,2,J]")
        batch, frames, views, joints, _ = pixels.shape
        projection = self._projection(
            projection, batch, pixels.dtype, pixels.device
        )
        image_size = image_size.to(dtype=pixels.dtype, device=pixels.device)
        if image_size.shape != (2, 2) or torch.any(image_size <= 0):
            raise ValueError("image_size must have positive shape [2,2]")
        finite = torch.isfinite(pixels).all(dim=-1)
        valid = valid.bool() & finite
        safe_pixels = torch.where(valid[..., None], pixels, torch.zeros_like(pixels))
        scale = image_size[None, None, :, None, :]
        normalized = 2.0 * safe_pixels / scale - 1.0
        velocity = torch.zeros_like(normalized)
        velocity[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        velocity_valid = torch.zeros_like(valid)
        velocity_valid[:, 1:] = valid[:, 1:] & valid[:, :-1]
        velocity = torch.where(
            velocity_valid[..., None], velocity, torch.zeros_like(velocity)
        )
        view_feature = torch.zeros(
            (1, 1, views, 1, 1),
            dtype=pixels.dtype,
            device=pixels.device,
        )
        view_feature[:, :, 1] = 1.0
        feature = torch.cat(
            (
                normalized,
                velocity,
                valid.to(pixels.dtype).unsqueeze(-1),
                view_feature.expand(batch, frames, -1, joints, -1),
            ),
            dim=-1,
        )
        confidence = 0.05 + 0.95 * torch.sigmoid(self.confidence(feature)[..., 0])
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))

        x = safe_pixels[..., 0]
        y = safe_pixels[..., 1]
        p0 = projection[:, None, :, None, 0].expand(-1, frames, -1, joints, -1)
        p1 = projection[:, None, :, None, 1].expand(-1, frames, -1, joints, -1)
        p2 = projection[:, None, :, None, 2].expand(-1, frames, -1, joints, -1)
        row_x = x[..., None] * p2 - p0
        row_y = y[..., None] * p2 - p1
        weighted = (
            torch.stack((row_x, row_y), dim=-2)
            * confidence[..., None, None]
        )
        matrix = weighted.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, frames, joints, 4, 4
        )
        _, _, vh = torch.linalg.svd(matrix)
        homogeneous = vh[..., -1, :]
        denominator = homogeneous[..., 3]
        both_valid = valid.all(dim=2)
        stable = (
            both_valid
            & torch.isfinite(homogeneous).all(dim=-1)
            & (denominator.abs() > 1e-8)
        )
        safe_denominator = torch.where(
            denominator.abs() > 1e-8,
            denominator,
            torch.ones_like(denominator),
        )
        points = homogeneous[..., :3] / safe_denominator[..., None]
        points = torch.where(stable[..., None], points, torch.zeros_like(points))
        return CalibratedPrediction(
            points=points,
            valid=stable,
            diagnostics={"confidence": confidence},
        )
