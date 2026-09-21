"""Weighted Pose Fusion of the original view predictions.

Research Motivation:
    The fusion network is designed to be *conservative*: whatever the neural
    branches compute, the base estimate must stay a (matrix-)weighted average
    of the two measured poses.  Anchoring the fusion to the ORIGINAL input
    predictions ``P_A`` and ``P_B`` (rather than to separate per-view 3D pose
    heads regressed from features) guarantees that

    * a joint that both views measured identically is returned unchanged,
    * the fused coordinate lies between the two measurements along every
      axis, so the reliability weights are directly interpretable as
      "how much of each camera" was used,
    * no capacity is spent re-learning to regress 3D poses, which the
      monocular estimator already does far better than a small fusion
      network could from scratch.

    Any systematic correction beyond that combination is delegated to the
    explicitly bounded residual refinement stage.

Method (architecture v1.0, :func:`weighted_pose_fusion`):
    P_base = w_A * P_A + w_B * P_B          with w_A + w_B = 1

    Joints valid in only one view take that view's position (its weight is
    one after masking); joints valid in neither view are zero and flagged
    invalid.

Method (architecture v1.1, :func:`depth_aware_pose_fusion`):
    A monocular 3D estimate is reliable in its image plane and unreliable
    along its own optical axis (metric depth).  Each view's optical axis
    expressed in the body frame, ``d_v`` (row 2 of that view's canonicalisation
    rotation, no camera calibration involved), is therefore known a priori,
    and the scalar reliability is turned into a per-view precision matrix

        Lambda_v = w_v * (I - alpha * d_v d_v^T)                     [B, T, J, 3, 3]

    (precision ``w_v`` in the image plane, ``w_v (1 - alpha)`` along depth) and
    the base pose is the precision-weighted least-squares combination

        P_base = (Lambda_A + Lambda_B)^-1 (Lambda_A P_A + Lambda_B P_B)

    With ``alpha = 0`` this is exactly the v1.0 rule; with ``alpha = 1`` and
    orthogonal cameras every coordinate comes from the view that measures it
    in the image plane.  ``alpha`` is a fixed prior (0.8, selected on the
    FreeMan reference): the label-free objectives are built from view
    consensus, so a learned ``alpha`` would collapse to the plain average.
    The learned scalar ``w_v`` keeps its role of trusting one view more when
    a joint is damaged in the other.  Because ``w_A + w_B = 1`` and
    ``alpha < 1`` the system is always invertible (eigenvalues >= 1 - alpha);
    for near-parallel optical axes ``alpha`` is additionally capped per frame
    so the summed precision keeps eigenvalues >= ``min_precision`` and the
    unobservable direction degrades to an average instead of amplifying the
    small misalignment between the two body frames.

Shapes:
    P_A, P_B        [B, T, J, 3]
    w_A, w_B        [B, T, J, 1]
    valid_A/valid_B [B, T, J]
    d_A, d_B        [B, T, 3]      unit vectors, zero where unknown (-> isotropic)
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


def depth_aware_pose_fusion(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    depth_a: torch.Tensor | None,
    depth_b: torch.Tensor | None,
    *,
    alpha: float,
    min_precision: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precision-form base pose with each view's depth axis discounted (v1.1).

    Args:
        pose_a: ``P_A`` with shape ``[B, T, J, 3]``.
        pose_b: ``P_B`` with shape ``[B, T, J, 3]``.
        weight_a: ``w_A`` with shape ``[B, T, J, 1]`` (``w_A + w_B = 1``).
        weight_b: ``w_B`` with shape ``[B, T, J, 1]``.
        valid_a: ``[B, T, J]`` bool validity of View A.
        valid_b: ``[B, T, J]`` bool validity of View B.
        depth_a: ``[B, T, 3]`` unit optical axis of View A's camera in the
            body frame; zero rows mean "unknown" (isotropic weighting).
            ``None`` disables the depth prior for both views.
        depth_b: ``[B, T, 3]`` likewise for View B.
        alpha: Fraction of a view's precision removed along its own depth
            axis, in ``[0, 1)``; ``0`` reproduces :func:`weighted_pose_fusion`
            exactly.
        min_precision: Lower bound kept on the eigenvalues of the unit-weight
            precision sum ``W_A + W_B`` (per frame, by capping ``alpha`` when
            the two optical axes are nearly parallel; never active for
            ``alpha = 0.8`` unless the axes are within ~29 degrees).

    Returns:
        Tuple ``(P_base, valid)`` exactly as :func:`weighted_pose_fusion`.

    Raises:
        ValueError: If the shapes are inconsistent or ``alpha`` is not in ``[0, 1)``.
    """
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if float(alpha) == 0.0 or depth_a is None or depth_b is None:
        return weighted_pose_fusion(pose_a, pose_b, weight_a, weight_b, valid_a, valid_b)
    if pose_a.shape != pose_b.shape or pose_a.ndim != 4 or pose_a.shape[-1] != 3:
        raise ValueError("pose_a and pose_b must both have shape [B, T, J, 3]")
    if weight_a.shape != pose_a.shape[:-1] + (1,) or weight_b.shape != weight_a.shape:
        raise ValueError("weights must have shape [B, T, J, 1]")
    if valid_a.shape != pose_a.shape[:-1] or valid_b.shape != pose_a.shape[:-1]:
        raise ValueError("validity masks must have shape [B, T, J]")
    if depth_a.shape != pose_a.shape[:2] + (3,) or depth_b.shape != depth_a.shape:
        raise ValueError("depth axes must have shape [B, T, 3]")
    valid_a, valid_b = valid_a.bool(), valid_b.bool()
    dtype = pose_a.dtype
    # The 3x3 solves are not autocast-safe (the matmuls would produce bfloat16
    # numerators against an fp32 system): run the fusion in fp32 and return
    # in the callers' dtype.
    with torch.autocast(device_type=pose_a.device.type, enabled=False):
        pose_a, pose_b = pose_a.float(), pose_b.float()
        weight_a, weight_b = weight_a.float(), weight_b.float()
        depth_a = depth_a.float()
        depth_b = depth_b.float()
        # Per-frame cap (same rule as the deterministic baseline): the eigenvalues
        # of d_A d_A^T + d_B d_B^T are 1 +- |cos| and 0, so the unit-weight sum
        # W_A + W_B keeps eigenvalues >= min_precision iff
        # alpha_t <= (2 - min_precision) / (1 + |cos|); the reliability-weighted
        # sum (w_A + w_B = 1) is then >= min_precision / 2 > 0 as well.
        cosine = (depth_a * depth_b).sum(dim=-1).abs()
        alpha_t = torch.clamp((2.0 - float(min_precision)) / (1.0 + cosine), max=float(alpha))  # [B, T]
        eye = torch.eye(3, dtype=torch.float32, device=pose_a.device)
        projector_a = eye - alpha_t[..., None, None] * depth_a[..., :, None] * depth_a[..., None, :]  # [B, T, 3, 3]
        projector_b = eye - alpha_t[..., None, None] * depth_b[..., :, None] * depth_b[..., None, :]
        # Invalid views contribute nothing (the reliability head already masks them;
        # this makes the rule exact for any weights).
        weight_a = torch.where(valid_a[..., None], weight_a, torch.zeros_like(weight_a))
        weight_b = torch.where(valid_b[..., None], weight_b, torch.zeros_like(weight_b))
        precision_a = weight_a[..., None] * projector_a[:, :, None]  # [B, T, J, 3, 3]
        precision_b = weight_b[..., None] * projector_b[:, :, None]
        safe_a = torch.where(valid_a[..., None], pose_a, torch.zeros_like(pose_a))
        safe_b = torch.where(valid_b[..., None], pose_b, torch.zeros_like(pose_b))
        numerator = (precision_a @ safe_a[..., None]) + (precision_b @ safe_b[..., None])  # [B, T, J, 3, 1]
        system = precision_a + precision_b
        valid = valid_a | valid_b
        # Joints valid in neither view have a zero system: substitute the identity
        # so the batched solve stays well posed (their output is zeroed below).
        system = torch.where(valid[..., None, None], system, eye.expand_as(system))
        fused = torch.linalg.solve(system, numerator)[..., 0]
        fused = torch.where(valid[..., None], fused, torch.zeros_like(fused))
    return fused.to(dtype), valid
