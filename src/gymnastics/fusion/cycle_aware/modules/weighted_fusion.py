"""Weighted Pose Fusion of the original view predictions.

Research Motivation:
    The fusion network is designed to be *conservative*: whatever the neural
    branches compute, the base estimate must stay a convex combination of the
    two measured poses.  Anchoring the fusion to the ORIGINAL input
    predictions ``P_A`` and ``P_B`` (rather than to separate per-view 3D pose
    heads regressed from features) guarantees that

    * a joint that both views measured identically is returned unchanged,
    * the fused joint always lies on the segment between the two
      measurements, so the reliability weights are directly interpretable as
      "how much of each camera" was used,
    * no capacity is spent re-learning to regress 3D poses, which the
      monocular estimator already does far better than a small fusion
      network could from scratch.

    Any systematic correction beyond that convex combination is delegated to
    the explicitly bounded residual refinement stage.

Method:
    P_base = w_A * P_A + w_B * P_B          with w_A + w_B = 1

    Joints valid in only one view take that view's position (its weight is
    one after masking); joints valid in neither view are zero and flagged
    invalid.

Shapes:
    P_A, P_B        [B, T, J, 3]
    w_A, w_B        [B, T, J, 1]
    valid_A/valid_B [B, T, J]
    P_base          [B, T, J, 3]
    valid           [B, T, J]
"""

from __future__ import annotations

import torch


def weighted_pose_fusion(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Form the reliability-weighted base pose from the original inputs.

    Args:
        pose_a: ``P_A`` with shape ``[B, T, J, 3]``.
        pose_b: ``P_B`` with shape ``[B, T, J, 3]``.
        weight_a: ``w_A`` with shape ``[B, T, J, 1]``.
        weight_b: ``w_B`` with shape ``[B, T, J, 1]``.
        valid_a: ``[B, T, J]`` bool validity of View A.
        valid_b: ``[B, T, J]`` bool validity of View B.

    Returns:
        Tuple ``(P_base, valid)``: the fused pose ``[B, T, J, 3]`` and the
        bool mask ``[B, T, J]`` of joints valid in at least one view.

    Raises:
        ValueError: If the shapes are inconsistent.
    """
    if pose_a.shape != pose_b.shape or pose_a.ndim != 4 or pose_a.shape[-1] != 3:
        raise ValueError("pose_a and pose_b must both have shape [B, T, J, 3]")
    if weight_a.shape != pose_a.shape[:-1] + (1,) or weight_b.shape != weight_a.shape:
        raise ValueError("weights must have shape [B, T, J, 1]")
    if valid_a.shape != pose_a.shape[:-1] or valid_b.shape != pose_a.shape[:-1]:
        raise ValueError("validity masks must have shape [B, T, J]")
    valid_a, valid_b = valid_a.bool(), valid_b.bool()
    safe_a = torch.where(valid_a[..., None], pose_a, torch.zeros_like(pose_a))
    safe_b = torch.where(valid_b[..., None], pose_b, torch.zeros_like(pose_b))
    fused = weight_a * safe_a + weight_b * safe_b
    valid = valid_a | valid_b
    return torch.where(valid[..., None], fused, torch.zeros_like(fused)), valid
