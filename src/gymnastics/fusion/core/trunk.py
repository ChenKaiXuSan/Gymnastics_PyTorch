"""Differentiable pelvis/thorax relative-rotation kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import SkeletonSpec
from .geometry import build_pelvis_frame, build_thorax_frame


@dataclass(frozen=True)
class TrunkFeatures:
    rotation: Tensor
    rotation_valid: Tensor
    angle: Tensor
    angle_valid: Tensor
    omega: Tensor
    omega_valid: Tensor
    alpha: Tensor
    alpha_valid: Tensor


def circular_diff(a: Tensor, b: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def relative_rotation(pelvis_rotation: Tensor, thorax_rotation: Tensor) -> Tensor:
    return pelvis_rotation.transpose(-1, -2) @ thorax_rotation


def axial_rotation_angle(rotation: Tensor) -> Tensor:
    """Return the signed rotation about the frame's local y (long) axis."""
    return torch.atan2(rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 0, 0] + rotation[..., 2, 2])


def axial_rotation_angle_from_points(points: Tensor, valid: Tensor, spec: SkeletonSpec) -> tuple[Tensor, Tensor]:
    pelvis = build_pelvis_frame(points, valid, spec)
    thorax = build_thorax_frame(points, valid, spec)
    rotation = relative_rotation(pelvis.rotation, thorax.rotation)
    return torch.where(pelvis.valid & thorax.valid, axial_rotation_angle(rotation), torch.zeros_like(rotation[..., 0, 0])), pelvis.valid & thorax.valid


def _derivative(values: Tensor, valid: Tensor, dt: Tensor, *, circular: bool) -> tuple[Tensor, Tensor]:
    result = torch.zeros_like(values)
    result_valid = torch.zeros_like(valid)
    if values.shape[1] < 2:
        return result, result_valid
    difference = circular_diff(values[:, 1:], values[:, :-1]) if circular else values[:, 1:] - values[:, :-1]
    dt_slice = dt[:, 1:]
    delta_valid = valid[:, 1:] & valid[:, :-1] & torch.isfinite(dt_slice) & (dt_slice > 0)
    # Divide by a guaranteed-positive denominator so an invalid (dt<=0 / padded)
    # frame never divides by zero. The `where` below already zeroes those frames'
    # VALUES; the safe denominator also keeps the division's BACKWARD finite
    # (grad * 1/dt would otherwise be 0 * inf = nan and poison every upstream
    # gradient the moment anyone differentiates omega/alpha, e.g. 改法3 / A9).
    safe_dt = torch.where(delta_valid, dt_slice, torch.ones_like(dt_slice))
    delta = difference / safe_dt
    result[:, 1:] = torch.where(delta_valid, delta, torch.zeros_like(delta))
    result_valid[:, 1:] = delta_valid
    return result, result_valid


def extract_trunk_features(points: Tensor, valid: Tensor, spec: SkeletonSpec, dt: float | Tensor) -> TrunkFeatures:
    """Extract relative rotation, axial angle, angular velocity, and acceleration."""
    pelvis = build_pelvis_frame(points, valid, spec)
    thorax = build_thorax_frame(points, valid, spec)
    rotation = relative_rotation(pelvis.rotation, thorax.rotation)
    rotation_valid = pelvis.valid & thorax.valid
    angle = torch.where(rotation_valid, axial_rotation_angle(rotation), torch.zeros_like(rotation[..., 0, 0]))
    if isinstance(dt, Tensor):
        if dt.ndim == 0:
            intervals = dt.expand(points.shape[0], points.shape[1])
        elif dt.shape == points.shape[:2]:
            intervals = dt
        else:
            raise ValueError("dt tensor must be scalar or have shape [B, T]")
    else:
        intervals = points.new_full(points.shape[:2], float(dt))
    omega, omega_valid = _derivative(angle, rotation_valid, intervals, circular=True)
    alpha, alpha_valid = _derivative(omega, omega_valid, intervals, circular=False)
    return TrunkFeatures(rotation, rotation_valid, angle, rotation_valid, omega, omega_valid, alpha, alpha_valid)
