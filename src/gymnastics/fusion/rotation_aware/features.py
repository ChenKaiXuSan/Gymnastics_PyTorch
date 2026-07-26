"""Finite, deterministic pose, quality, and cross-view feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import SkeletonSpec
from .trunk import TrunkFeatures, circular_diff


@dataclass(frozen=True)
class PoseFeatures:
    points: Tensor
    valid: Tensor
    velocity: Tensor
    velocity_valid: Tensor
    bone_lengths: Tensor
    bone_valid: Tensor


@dataclass(frozen=True)
class QualityFeatures:
    score: Tensor
    loss_weight: Tensor
    shoulder_deviation: Tensor
    hip_deviation: Tensor
    torso_deviation: Tensor
    rigidity_residual: Tensor
    angular_outlier: Tensor
    frame_degenerate: Tensor
    valid_ratio: Tensor


@dataclass(frozen=True)
class QualityConfig:
    """Fixed, deterministic penalty weights for quality scoring."""

    shoulder_weight: float = 1.0
    hip_weight: float = 1.0
    torso_weight: float = 1.0
    rigidity_weight: float = 1.0
    angular_weight: float = 1.0
    frame_degenerate_weight: float = 1.0


@dataclass(frozen=True)
class DisagreementFeatures:
    coordinate_abs_delta: Tensor
    coordinate_valid: Tensor
    angle_abs_delta: Tensor
    angle_valid: Tensor
    rotation_distance: Tensor
    rotation_valid: Tensor
    trunk_displacement_abs_delta: Tensor
    trunk_displacement_valid: Tensor
    validity_abs_delta: Tensor


@dataclass(frozen=True)
class FeatureBundle:
    """Feature groups consumed by the later swap-invariant fusion model."""

    pose: PoseFeatures
    quality: QualityFeatures
    disagreement: DisagreementFeatures | None = None


def _safe_points(points: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, T, J, 3]")
    if valid.shape != points.shape[:-1]:
        raise ValueError("valid must have shape [B, T, J]")
    finite = torch.isfinite(points).all(dim=-1)
    effective_valid = valid.bool() & finite
    return torch.where(effective_valid[..., None], points, torch.zeros_like(points)), effective_valid


def _intervals(points: Tensor, dt: float | Tensor) -> Tensor:
    if isinstance(dt, Tensor):
        if dt.ndim == 0:
            return dt.to(dtype=points.dtype, device=points.device).expand(points.shape[:2])
        if dt.shape == points.shape[:2]:
            return dt.to(dtype=points.dtype, device=points.device)
        raise ValueError("dt tensor must be scalar or have shape [B, T]")
    return points.new_full(points.shape[:2], float(dt))


def extract_pose_features(points: Tensor, valid: Tensor, spec: SkeletonSpec, dt: float | Tensor) -> PoseFeatures:
    """Extract masked coordinate velocity and configured bone-length features."""
    safe_points, effective_valid = _safe_points(points, valid)
    velocity = torch.zeros_like(safe_points)
    velocity_valid = torch.zeros_like(effective_valid)
    intervals = _intervals(safe_points, dt)
    if points.shape[1] > 1:
        delta_valid = effective_valid[:, 1:] & effective_valid[:, :-1]
        usable_dt = torch.isfinite(intervals[:, 1:]) & (intervals[:, 1:] > 0)
        delta_valid = delta_valid & usable_dt[..., None]
        delta = (safe_points[:, 1:] - safe_points[:, :-1]) / intervals[:, 1:, None, None].clamp_min(1e-6)
        velocity[:, 1:] = torch.where(delta_valid[..., None], delta, torch.zeros_like(delta))
        velocity_valid[:, 1:] = delta_valid

    if not spec.bones:
        shape = points.shape[:2] + (0,)
        return PoseFeatures(safe_points, effective_valid, velocity, velocity_valid, points.new_zeros(shape), torch.zeros(shape, dtype=torch.bool, device=points.device))
    start = torch.tensor([bone[0] for bone in spec.bones], device=points.device)
    end = torch.tensor([bone[1] for bone in spec.bones], device=points.device)
    bone_valid = effective_valid[:, :, start] & effective_valid[:, :, end]
    lengths = torch.linalg.vector_norm(safe_points[:, :, end] - safe_points[:, :, start], dim=-1)
    lengths = torch.where(bone_valid, lengths, torch.zeros_like(lengths))
    return PoseFeatures(safe_points, effective_valid, velocity, velocity_valid, lengths, bone_valid)


def _role_point(points: Tensor, valid: Tensor, spec: SkeletonSpec, name: str) -> tuple[Tensor, Tensor]:
    role = spec.role(name)

    def resolve(names: tuple[str, ...]) -> tuple[Tensor, Tensor]:
        indexes = [spec.joint_index(joint) for joint in names]
        return points[:, :, indexes].mean(dim=2), valid[:, :, indexes].all(dim=-1)

    primary, primary_valid = resolve(role.joints)
    if not role.fallback:
        return primary, primary_valid
    fallback, fallback_valid = resolve(role.fallback)
    return torch.where(primary_valid[..., None], primary, fallback), primary_valid | fallback_valid


def _robust_deviation(values: Tensor, valid: Tensor) -> Tensor:
    result = torch.zeros_like(values)
    for batch_index in range(values.shape[0]):
        usable = values[batch_index][valid[batch_index]]
        if usable.numel():
            median = usable.median()
            scale = (usable - median).abs().median().clamp_min(1e-6)
            result[batch_index] = (values[batch_index] - median).abs() / scale
    return torch.where(valid, result, torch.zeros_like(result))


def _robust_temporal_bone_deviation(values: Tensor, valid: Tensor) -> Tensor:
    """Return each bone's deviation from its own robust temporal baseline."""
    result = torch.zeros_like(values)
    for batch_index in range(values.shape[0]):
        for bone_index in range(values.shape[2]):
            usable = values[batch_index, :, bone_index][valid[batch_index, :, bone_index]]
            if usable.numel():
                median = usable.median()
                scale = (usable - median).abs().median().clamp_min(1e-6)
                result[batch_index, :, bone_index] = (values[batch_index, :, bone_index] - median).abs() / scale
    return torch.where(valid, result, torch.zeros_like(result))


def compute_quality_features(
    points: Tensor,
    valid: Tensor,
    trunk: TrunkFeatures,
    spec: SkeletonSpec,
    config: QualityConfig = QualityConfig(),
) -> QualityFeatures:
    """Compute fixed robust reliability scores; all outputs are detached."""
    pose = extract_pose_features(points, valid, spec, dt=1.0)
    safe_points, effective_valid = pose.points, pose.valid
    left_shoulder, left_shoulder_valid = _role_point(safe_points, effective_valid, spec, "left_shoulder")
    right_shoulder, right_shoulder_valid = _role_point(safe_points, effective_valid, spec, "right_shoulder")
    left_hip, left_hip_valid = _role_point(safe_points, effective_valid, spec, "left_hip")
    right_hip, right_hip_valid = _role_point(safe_points, effective_valid, spec, "right_hip")
    pelvis, pelvis_valid = _role_point(safe_points, effective_valid, spec, "pelvis")
    thorax, thorax_valid = _role_point(safe_points, effective_valid, spec, "thorax")
    shoulder_valid = left_shoulder_valid & right_shoulder_valid
    hip_valid = left_hip_valid & right_hip_valid
    torso_valid = pelvis_valid & thorax_valid
    shoulder = torch.linalg.vector_norm(right_shoulder - left_shoulder, dim=-1)
    hip = torch.linalg.vector_norm(right_hip - left_hip, dim=-1)
    torso = torch.linalg.vector_norm(thorax - pelvis, dim=-1)
    shoulder_deviation = _robust_deviation(shoulder, shoulder_valid)
    hip_deviation = _robust_deviation(hip, hip_valid)
    torso_deviation = _robust_deviation(torso, torso_valid)
    if spec.bones:
        bone_residual = _robust_temporal_bone_deviation(pose.bone_lengths, pose.bone_valid)
        valid_bones = pose.bone_valid.sum(dim=-1)
        rigidity = torch.where(
            valid_bones > 0,
            bone_residual.sum(dim=-1) / valid_bones.clamp_min(1),
            torch.zeros_like(shoulder),
        )
    else:
        rigidity = torch.zeros_like(shoulder)
    angular_outlier = _robust_deviation(trunk.alpha.abs(), trunk.alpha_valid)
    frame_degenerate = (~trunk.rotation_valid).to(points.dtype)
    valid_ratio = effective_valid.to(points.dtype).mean(dim=-1)
    penalty = (
        config.shoulder_weight * shoulder_deviation
        + config.hip_weight * hip_deviation
        + config.torso_weight * torso_deviation
        + config.rigidity_weight * rigidity
        + config.angular_weight * angular_outlier
        + config.frame_degenerate_weight * frame_degenerate
    )
    score = valid_ratio / (1.0 + penalty)
    score = torch.where(torch.isfinite(score), score.clamp(0.0, 1.0), torch.zeros_like(score)).detach()
    return QualityFeatures(
        score=score,
        loss_weight=score.detach(),
        shoulder_deviation=shoulder_deviation.detach(),
        hip_deviation=hip_deviation.detach(),
        torso_deviation=torso_deviation.detach(),
        rigidity_residual=rigidity.detach(),
        angular_outlier=angular_outlier.detach(),
        frame_degenerate=frame_degenerate.detach(),
        valid_ratio=valid_ratio.detach(),
    )


def _rotation_distance(face: Tensor, side: Tensor) -> Tensor:
    relative = face.transpose(-1, -2) @ side
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def compute_disagreement_features(
    face: Tensor,
    side: Tensor,
    face_trunk: TrunkFeatures,
    side_trunk: TrunkFeatures,
    face_valid: Tensor,
    side_valid: Tensor,
) -> DisagreementFeatures:
    """Compute symmetric cross-view differences while excluding invalid joints."""
    safe_face, face_valid = _safe_points(face, face_valid)
    safe_side, side_valid = _safe_points(side, side_valid)
    coordinate_valid = face_valid & side_valid
    coordinate_delta = (safe_face - safe_side).abs()
    coordinate_delta = torch.where(coordinate_valid[..., None], coordinate_delta, torch.zeros_like(coordinate_delta))
    angle_valid = face_trunk.angle_valid & side_trunk.angle_valid
    angle_delta = circular_diff(face_trunk.angle, side_trunk.angle).abs()
    angle_delta = torch.where(angle_valid, angle_delta, torch.zeros_like(angle_delta))
    rotation_valid = face_trunk.rotation_valid & side_trunk.rotation_valid
    rotation_delta = _rotation_distance(face_trunk.rotation, side_trunk.rotation)
    rotation_delta = torch.where(rotation_valid, rotation_delta, torch.zeros_like(rotation_delta))
    common_count = coordinate_valid.sum(dim=2, keepdim=True)
    displacement_valid = common_count[..., 0] > 0
    face_center = (safe_face * coordinate_valid[..., None]).sum(dim=2) / common_count.clamp_min(1)
    side_center = (safe_side * coordinate_valid[..., None]).sum(dim=2) / common_count.clamp_min(1)
    displacement = torch.where(displacement_valid[..., None], (face_center - side_center).abs(), torch.zeros_like(face_center))
    return DisagreementFeatures(
        coordinate_delta,
        coordinate_valid,
        angle_delta,
        angle_valid,
        rotation_delta,
        rotation_valid,
        displacement,
        displacement_valid,
        (face_valid.to(face.dtype) - side_valid.to(face.dtype)).abs(),
    )
