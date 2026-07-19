"""Deterministic, view-swap-invariant canonical pose fusion baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FusionResult:
    points: Tensor
    valid: Tensor
    face_weight: Tensor
    side_weight: Tensor


def _safe_inputs(face: Tensor, side: Tensor, face_valid: Tensor, side_valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if face.ndim != 4 or face.shape[-1] != 3 or face.shape != side.shape:
        raise ValueError("face and side must have equal shape [B, T, J, 3]")
    if face_valid.shape != face.shape[:-1] or side_valid.shape != face.shape[:-1]:
        raise ValueError("valid masks must have shape [B, T, J]")
    face_valid = face_valid.bool() & torch.isfinite(face).all(dim=-1)
    side_valid = side_valid.bool() & torch.isfinite(side).all(dim=-1)
    return torch.where(face_valid[..., None], face, torch.zeros_like(face)), torch.where(side_valid[..., None], side, torch.zeros_like(side)), face_valid, side_valid


def _fuse(face: Tensor, side: Tensor, face_weight: Tensor, side_weight: Tensor) -> FusionResult:
    total = face_weight + side_weight
    valid = total > 0
    denominator = torch.where(valid, total, torch.ones_like(total))
    points = (face * face_weight[..., None] + side * side_weight[..., None]) / denominator[..., None]
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    return FusionResult(points, valid, face_weight, side_weight)


def arithmetic_fusion(face: Tensor, side: Tensor, face_valid: Tensor, side_valid: Tensor) -> FusionResult:
    """Fuse valid views with equal weights and emit zero for both-invalid joints."""
    face, side, face_valid, side_valid = _safe_inputs(face, side, face_valid, side_valid)
    return _fuse(face, side, face_valid.to(face.dtype), side_valid.to(face.dtype))


def _quality(quality: Tensor, points: Tensor, name: str) -> Tensor:
    if quality.shape != points.shape[:2]:
        raise ValueError(f"{name} must have shape [B, T]")
    quality = quality.detach().to(dtype=points.dtype, device=points.device)
    return torch.where(torch.isfinite(quality), quality.clamp_min(0), torch.zeros_like(quality))


def quality_weighted_fusion(
    face: Tensor,
    side: Tensor,
    face_valid: Tensor,
    side_valid: Tensor,
    quality_face: Tensor,
    quality_side: Tensor,
) -> FusionResult:
    """Fuse with detached deterministic frame quality and valid-view fallback."""
    face, side, face_valid, side_valid = _safe_inputs(face, side, face_valid, side_valid)
    face_quality = _quality(quality_face, face, "quality_face")
    side_quality = _quality(quality_side, side, "quality_side")
    face_weight = face_valid.to(face.dtype) * face_quality[..., None]
    side_weight = side_valid.to(side.dtype) * side_quality[..., None]
    both_valid = face_valid & side_valid
    face_only = face_valid & ~side_valid
    side_only = side_valid & ~face_valid
    both_face_weight = torch.where(both_valid, face_weight, torch.zeros_like(face_weight))
    both_side_weight = torch.where(both_valid, side_weight, torch.zeros_like(side_weight))
    zero_both_weight = both_valid & ((both_face_weight + both_side_weight) == 0)
    both_face_weight = torch.where(zero_both_weight, torch.ones_like(both_face_weight), both_face_weight)
    both_side_weight = torch.where(zero_both_weight, torch.ones_like(both_side_weight), both_side_weight)
    both = _fuse(face, side, both_face_weight, both_side_weight)
    points = torch.where(face_only[..., None], face, torch.where(side_only[..., None], side, both.points))
    return FusionResult(points, face_valid | side_valid, face_weight, side_weight)
