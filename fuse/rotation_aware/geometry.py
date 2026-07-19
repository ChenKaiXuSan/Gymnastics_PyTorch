"""Differentiable canonical coordinate systems for MHR70 pose sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import SkeletonSpec


@dataclass(frozen=True)
class Frame:
    """A local-to-world frame and the frames where it was observed directly."""

    rotation: Tensor
    origin: Tensor
    valid: Tensor


@dataclass(frozen=True)
class CanonicalTransform:
    """Per-frame pose frame plus the robust scale shared by a trial."""

    rotation: Tensor
    origin: Tensor
    scale: Tensor
    valid: Tensor


@dataclass(frozen=True)
class CanonicalizedPose:
    points: Tensor
    valid: Tensor
    transform: CanonicalTransform


def _finite_points(points: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, T, J, 3]")
    if valid.shape != points.shape[:-1]:
        raise ValueError("valid must have shape [B, T, J]")
    finite = torch.isfinite(points).all(dim=-1)
    return torch.where(finite[..., None], points, torch.zeros_like(points)), valid.bool() & finite


def _role_point(points: Tensor, valid: Tensor, spec: SkeletonSpec, name: str) -> tuple[Tensor, Tensor]:
    role = spec.role(name)

    def resolve(joints: tuple[str, ...]) -> tuple[Tensor, Tensor]:
        indexes = [spec.joint_index(joint) for joint in joints]
        role_points = points[:, :, indexes]
        role_valid = valid[:, :, indexes].all(dim=-1)
        return role_points.mean(dim=2), role_valid

    primary, primary_valid = resolve(role.joints)
    if not role.fallback:
        return primary, primary_valid
    fallback, fallback_valid = resolve(role.fallback)
    return torch.where(primary_valid[..., None], primary, fallback), primary_valid | fallback_valid


def _safe_normalize(vector: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    valid = length[..., 0] > eps
    return vector / length.clamp_min(eps), valid


def _continuous_frame(rotation: Tensor, origin: Tensor, valid: Tensor) -> Frame:
    """Use the most recent observable frame only to keep transforms continuous."""
    batch, frames = valid.shape
    previous_rotation = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(batch, 3, 3)
    previous_origin = torch.zeros(batch, 3, dtype=origin.dtype, device=origin.device)
    rotations, origins = [], []
    for index in range(frames):
        observed = valid[:, index]
        previous_rotation = torch.where(observed[:, None, None], rotation[:, index], previous_rotation)
        previous_origin = torch.where(observed[:, None], origin[:, index], previous_origin)
        rotations.append(previous_rotation)
        origins.append(previous_origin)
    return Frame(rotation=torch.stack(rotations, dim=1), origin=torch.stack(origins, dim=1), valid=valid)


def _build_frame(origin: Tensor, left: Tensor, right: Tensor, vertical: Tensor, role_valid: Tensor) -> Frame:
    x_axis, x_valid = _safe_normalize(right - left)
    vertical = vertical - (vertical * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis, y_valid = _safe_normalize(vertical)
    z_axis, z_valid = _safe_normalize(torch.cross(x_axis, y_axis, dim=-1))
    # Reconstruct x from y and z so the right-handed frame is orthonormal.
    x_axis, _ = _safe_normalize(torch.cross(y_axis, z_axis, dim=-1))
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    determinant = torch.linalg.det(rotation)
    z_axis = torch.where((determinant < 0)[..., None], -z_axis, z_axis)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    observed = role_valid & x_valid & y_valid & z_valid & (determinant.abs() > 1e-6) & (torch.linalg.det(rotation) > 0)
    return _continuous_frame(rotation, origin, observed)


def build_pelvis_frame(points: Tensor, valid: Tensor, spec: SkeletonSpec) -> Frame:
    """Build a pelvis frame from hips and the pelvis-to-thorax direction."""
    points, valid = _finite_points(points, valid)
    left, left_valid = _role_point(points, valid, spec, "left_hip")
    right, right_valid = _role_point(points, valid, spec, "right_hip")
    thorax, thorax_valid = _role_point(points, valid, spec, "thorax")
    pelvis, pelvis_valid = _role_point(points, valid, spec, "pelvis")
    return _build_frame(pelvis, left, right, thorax - pelvis, left_valid & right_valid & thorax_valid & pelvis_valid)


def build_thorax_frame(points: Tensor, valid: Tensor, spec: SkeletonSpec) -> Frame:
    """Build a thorax frame with its long axis anchored to the pelvis."""
    points, valid = _finite_points(points, valid)
    shoulder_left, shoulder_left_valid = _role_point(points, valid, spec, "left_shoulder")
    shoulder_right, shoulder_right_valid = _role_point(points, valid, spec, "right_shoulder")
    shoulder_valid = shoulder_left_valid & shoulder_right_valid
    left, right, lateral_valid = shoulder_left, shoulder_right, shoulder_valid
    if "left_acromion" in spec.roles and "right_acromion" in spec.roles:
        acromion_left, acromion_left_valid = _role_point(points, valid, spec, "left_acromion")
        acromion_right, acromion_right_valid = _role_point(points, valid, spec, "right_acromion")
        acromion_valid = acromion_left_valid & acromion_right_valid
        left = torch.where(acromion_valid[..., None], acromion_left, shoulder_left)
        right = torch.where(acromion_valid[..., None], acromion_right, shoulder_right)
        lateral_valid = acromion_valid | shoulder_valid
    thorax, thorax_valid = _role_point(points, valid, spec, "thorax")
    neck, neck_valid = _role_point(points, valid, spec, "neck")
    return _build_frame(thorax, left, right, neck - thorax, lateral_valid & thorax_valid & neck_valid)


def _trial_scale(points: Tensor, valid: Tensor, spec: SkeletonSpec) -> Tensor:
    pelvis, pelvis_valid = _role_point(points, valid, spec, "pelvis")
    thorax, thorax_valid = _role_point(points, valid, spec, "thorax")
    lengths = torch.linalg.vector_norm(thorax - pelvis, dim=-1)
    usable = pelvis_valid & thorax_valid & torch.isfinite(lengths) & (lengths > 1e-6)
    scales = []
    for batch_index in range(points.shape[0]):
        samples = lengths[batch_index][usable[batch_index]]
        scales.append(samples.median() if samples.numel() else lengths.new_tensor(1.0))
    return torch.stack(scales).clamp_min(1e-6)


def canonicalize_pose(points: Tensor, valid: Tensor, spec: SkeletonSpec) -> CanonicalizedPose:
    """Center, rotate, and scale poses using an invertible trial-level transform."""
    safe_points, safe_valid = _finite_points(points, valid)
    pelvis = build_pelvis_frame(safe_points, safe_valid, spec)
    scale = _trial_scale(safe_points, safe_valid, spec)
    canonical_points = (safe_points - pelvis.origin[:, :, None, :]) @ pelvis.rotation
    canonical_points = canonical_points / scale[:, None, None, None]
    canonical_points = torch.where(safe_valid[..., None], canonical_points, torch.zeros_like(canonical_points))
    transform = CanonicalTransform(
        rotation=pelvis.rotation,
        origin=pelvis.origin,
        scale=scale,
        valid=pelvis.valid,
    )
    return CanonicalizedPose(points=canonical_points, valid=safe_valid, transform=transform)


def restore_pose(points: Tensor, transform: CanonicalTransform) -> Tensor:
    """Exactly invert :func:`canonicalize_pose` for valid canonical points."""
    if points.shape[:-2] != transform.rotation.shape[:-2] or points.shape[-1] != 3:
        raise ValueError("points and transform must have compatible [B, T, ...] shapes")
    return (
        points * transform.scale[:, None, None, None]
    ) @ transform.rotation.transpose(-1, -2) + transform.origin[:, :, None, :]
