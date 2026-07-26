"""Differentiable geometry for Unity-native 3D supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mapping import EVALUATION_JOINT_NAMES, MHR70_EVALUATION_SOURCES


def torch_map_mhr70_to_unity16(
    points: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map MHR70 tensors to the exact Unity16 semantic subset."""
    if points.ndim < 2 or points.shape[-2:] != (70, 3):
        raise ValueError("MHR70 points must end with shape [70,3]")
    if valid.shape != points.shape[:-1]:
        raise ValueError("MHR70 validity must match points without xyz")
    source_valid = (
        valid.bool()
        & torch.isfinite(points).all(dim=-1)
        & torch.any(points != 0, dim=-1)
    )
    mapped_points: list[torch.Tensor] = []
    mapped_valid: list[torch.Tensor] = []
    for name in EVALUATION_JOINT_NAMES:
        sources = MHR70_EVALUATION_SOURCES[name]
        selected = points[..., list(sources), :]
        selected_valid = source_valid[..., list(sources)]
        mapped_points.append(selected.mean(dim=-2))
        mapped_valid.append(selected_valid.all(dim=-1))
    output = torch.stack(mapped_points, dim=-2)
    output_valid = torch.stack(mapped_valid, dim=-1)
    output = torch.where(
        output_valid[..., None], output, torch.zeros_like(output)
    )
    return output, output_valid


@dataclass(frozen=True)
class DifferentiableSim3:
    """Batched row-vector similarity transforms."""

    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor


def masked_window_sim3(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> DifferentiableSim3:
    """Fit one masked Umeyama transform over all frames/joints per batch."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have shape [B,T,J,3]")
    if prediction.shape[-1] != 3 or valid.shape != prediction.shape[:-1]:
        raise ValueError("valid must match [B,T,J]")
    mask = valid.bool()
    finite = torch.isfinite(prediction).all(dim=-1) & torch.isfinite(target).all(
        dim=-1
    )
    if torch.any(mask & ~finite):
        raise ValueError("degenerate Sim3: non-finite usable points")
    batch = prediction.shape[0]
    flat_mask = mask.reshape(batch, -1)
    counts = flat_mask.sum(dim=1)
    if torch.any(counts < 3):
        raise ValueError("degenerate Sim3: fewer than three usable points")
    flat_prediction = prediction.reshape(batch, -1, 3)
    flat_target = target.reshape(batch, -1, 3)
    mask_xyz = flat_mask[..., None]
    safe_prediction = torch.where(
        mask_xyz, flat_prediction, torch.zeros_like(flat_prediction)
    )
    safe_target = torch.where(
        mask_xyz, flat_target, torch.zeros_like(flat_target)
    )
    denominator = counts.to(prediction.dtype)[:, None]
    prediction_mean = safe_prediction.sum(dim=1) / denominator
    target_mean = safe_target.sum(dim=1) / denominator
    centered_prediction = torch.where(
        mask_xyz,
        flat_prediction - prediction_mean[:, None],
        torch.zeros_like(flat_prediction),
    )
    centered_target = torch.where(
        mask_xyz,
        flat_target - target_mean[:, None],
        torch.zeros_like(flat_target),
    )
    prediction_variance = centered_prediction.square().sum(dim=(1, 2))
    target_variance = centered_target.square().sum(dim=(1, 2))
    epsilon = 1e-10
    if torch.any(prediction_variance <= epsilon) or torch.any(
        target_variance <= epsilon
    ):
        raise ValueError("degenerate Sim3: zero pose variance")
    covariance = torch.einsum(
        "bnc,bnd->bcd", centered_prediction, centered_target
    )
    u, singular_values, vh = torch.linalg.svd(covariance)
    determinant = torch.linalg.det(torch.matmul(u, vh))
    correction = torch.ones(
        (batch, 3), dtype=prediction.dtype, device=prediction.device
    )
    correction[:, -1] = torch.where(
        determinant < 0,
        -torch.ones_like(determinant),
        torch.ones_like(determinant),
    )
    rotation = torch.matmul(
        u * correction[:, None, :],
        vh,
    )
    scale = (singular_values * correction).sum(dim=-1) / prediction_variance
    if torch.any(scale <= epsilon):
        raise ValueError("degenerate Sim3: non-positive scale")
    rotated_mean = torch.einsum(
        "bc,bcd->bd", prediction_mean, rotation
    )
    translation = target_mean - scale[:, None] * rotated_mean
    if not (
        torch.isfinite(scale).all()
        and torch.isfinite(rotation).all()
        and torch.isfinite(translation).all()
    ):
        raise ValueError("degenerate Sim3: non-finite transform")
    return DifferentiableSim3(scale, rotation, translation)


def apply_torch_sim3(
    points: torch.Tensor,
    transform: DifferentiableSim3,
) -> torch.Tensor:
    """Apply a batch-level row-vector Sim3 to `[B,T,J,3]` points."""
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,T,J,3]")
    if transform.scale.shape != (points.shape[0],):
        raise ValueError("transform batch size does not match points")
    rotated = torch.einsum(
        "btjc,bcd->btjd", points, transform.rotation
    )
    return (
        transform.scale[:, None, None, None] * rotated
        + transform.translation[:, None, None, :]
    )
