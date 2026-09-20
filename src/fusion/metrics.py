"""Evaluation metrics against reference poses.

Training is label-free, but the DataModules may attach reference poses
(triangulated pseudo ground truth for the private data, markerless multi-view
references for FreeMan, native 3D for Unity) to validation and test windows.
Because the fused pose lives in the canonical body frame of View A while the
references live in their own world frames, the primary metric is the
Procrustes-aligned MPJPE, which is invariant to a per-frame similarity
transform.  Translation-only aligned MPJPE is reported as well for datasets
whose reference shares the canonical frame (e.g. the synthetic DataModule).

Shapes:
    prediction  [B, T, J, 3]
    reference   [B, T, J, 3]
    valid       [B, T, J]  joints where both prediction and reference exist
"""

from __future__ import annotations

import torch
from torch import Tensor


def procrustes_align(prediction: Tensor, reference: Tensor, valid: Tensor) -> Tensor:
    """Align each frame of ``prediction`` to ``reference`` by a similarity transform.

    Args:
        prediction: ``[N, J, 3]`` frames to align.
        reference: ``[N, J, 3]`` target frames.
        valid: ``[N, J]`` bool joints used for the alignment.

    Returns:
        ``[N, J, 3]`` aligned prediction (frames with fewer than three valid
        joints are returned centred but otherwise unchanged).
    """
    weight = valid.to(prediction.dtype)[..., None]
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_p = (prediction * weight).sum(dim=1, keepdim=True) / count
    mean_r = (reference * weight).sum(dim=1, keepdim=True) / count
    centred_p = (prediction - mean_p) * weight
    centred_r = (reference - mean_r) * weight
    covariance = centred_p.transpose(1, 2) @ centred_r  # [N, 3, 3]
    u, singular, vt = torch.linalg.svd(covariance)
    # Reflection guard: force a proper rotation.
    sign = torch.sign(torch.linalg.det(u @ vt))
    correction = torch.ones_like(singular)
    correction[:, -1] = sign
    rotation = (u * correction[:, None, :]) @ vt
    variance_p = centred_p.square().sum(dim=(1, 2))
    scale = (singular * correction).sum(dim=1) / variance_p.clamp_min(1e-8)
    enough = (valid.sum(dim=1) >= 3)
    scale = torch.where(enough, scale, torch.ones_like(scale))
    rotation = torch.where(enough[:, None, None], rotation, torch.eye(3, dtype=rotation.dtype, device=rotation.device)[None])
    aligned = scale[:, None, None] * ((prediction - mean_p) @ rotation) + mean_r
    return aligned


def per_joint_error(prediction: Tensor, reference: Tensor, valid: Tensor, *, align: str = "procrustes") -> tuple[Tensor, Tensor]:
    """Per-joint Euclidean error after the requested alignment.

    Args:
        prediction: ``[B, T, J, 3]``.
        reference: ``[B, T, J, 3]``.
        valid: ``[B, T, J]`` joints that exist in both.
        align: ``"procrustes"`` (similarity), ``"translation"`` (centroid) or
            ``"none"``.

    Returns:
        Tuple ``(error, valid)`` with error ``[B, T, J]`` (zero where invalid).
    """
    batch, frames, joints, _ = prediction.shape
    flat_p = prediction.reshape(batch * frames, joints, 3)
    flat_r = reference.reshape(batch * frames, joints, 3)
    flat_v = valid.reshape(batch * frames, joints).bool()
    if align == "procrustes":
        flat_p = procrustes_align(flat_p, flat_r, flat_v)
    elif align == "translation":
        weight = flat_v.to(flat_p.dtype)[..., None]
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        flat_p = flat_p - (flat_p * weight).sum(dim=1, keepdim=True) / count + (flat_r * weight).sum(dim=1, keepdim=True) / count
    elif align != "none":
        raise ValueError("align must be procrustes, translation or none")
    error = torch.linalg.vector_norm(flat_p - flat_r, dim=-1)
    error = torch.where(flat_v, error, torch.zeros_like(error))
    return error.reshape(batch, frames, joints), flat_v.reshape(batch, frames, joints)


def mean_per_joint_position_error(prediction: Tensor, reference: Tensor, valid: Tensor, *, align: str = "procrustes") -> Tensor:
    """Scalar MPJPE over valid joints (see :func:`per_joint_error`)."""
    error, mask = per_joint_error(prediction, reference, valid, align=align)
    return error.sum() / mask.sum().clamp_min(1).to(error.dtype)
