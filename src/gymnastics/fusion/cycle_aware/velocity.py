"""Physical joint velocity from 3D pose sequences.

This module computes the velocity features that feed the short-term and
long-term motion transformers of the cycle-aware fusion model.

Research Motivation:
    The motion branches are meant to encode *how fast* and *in which
    direction* each joint moves.  After cycle phase normalisation the samples
    of one cycle are equally spaced in *phase*, not in *time*: a slow cycle
    and a fast cycle produce the same number of samples.  A naive finite
    difference ``P[t] - P[t-1]`` would therefore measure displacement per
    phase step and lose the physical motion speed.  Dividing by the physical
    interval ``delta_t[t]`` between consecutive samples restores velocity in
    units per second and keeps speed information available to the network.

Method:
    v[t] = (P[t] - P[t-1]) / delta_t[t]        for t >= 1

    The first sample has no predecessor.  Two boundary strategies exist:

    * ``"replicate"`` copies ``v[1]`` into ``v[0]`` (zero-acceleration start);
    * ``"zero"`` sets ``v[0] = 0``.

    A velocity is valid only when both frames of the difference are valid;
    invalid velocities are set to zero and reported through the mask.

Shapes:
    pose      [B, T, J, 3]
    delta_t   [B, T]   seconds between sample t-1 and sample t (index 0 unused)
    valid     [B, T, J]  optional joint validity
    velocity  [B, T, J, 3]
    velocity_valid [B, T, J]
"""

from __future__ import annotations

import torch

BOUNDARY_STRATEGIES = ("replicate", "zero")


def compute_velocity(
    pose: torch.Tensor,
    delta_t: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    boundary: str = "replicate",
    min_delta_t: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes physical joint velocity from a 3D pose sequence.

    Velocity is computed using the actual temporal interval rather than the
    phase-normalized sampling interval.  This preserves real motion-speed
    information after cycle phase normalization.

    Args:
        pose: 3D joint coordinates with shape ``[B, T, J, 3]``.
        delta_t: Time interval between consecutive samples with shape
            ``[B, T]``; ``delta_t[:, t]`` is the interval from sample ``t-1``
            to sample ``t``.  A scalar tensor or a ``[B, T - 1]`` tensor is
            also accepted and broadcast/aligned accordingly.
        valid: Optional joint validity ``[B, T, J]``.  Velocities that involve
            an invalid frame are zeroed and marked invalid.
        boundary: ``"replicate"`` or ``"zero"`` handling of the first sample.
        min_delta_t: Lower clamp of the interval to avoid division by zero.

    Returns:
        Tuple ``(velocity, velocity_valid)`` with shapes ``[B, T, J, 3]`` and
        ``[B, T, J]``.

    Raises:
        ValueError: If the pose tensor does not contain an XYZ coordinate
            dimension of size 3.
        ValueError: If ``delta_t`` is incompatible with the temporal dimension.
        ValueError: If ``boundary`` is not a supported strategy.

    Notes:
        Phase normalization changes the number and spacing of samples within a
        motion cycle.  Therefore, velocity should not be naively interpreted as
        ``P[t] - P[t-1]`` after resampling when physical motion speed matters.
    """
    if pose.ndim != 4 or pose.shape[-1] != 3:
        raise ValueError("pose must have shape [B, T, J, 3]")
    if boundary not in BOUNDARY_STRATEGIES:
        raise ValueError(f"boundary must be one of {BOUNDARY_STRATEGIES}")
    batch, frames, joints, _ = pose.shape
    delta = torch.as_tensor(delta_t, dtype=pose.dtype, device=pose.device)
    if delta.ndim == 0:
        delta = delta.expand(batch, frames)
    elif delta.shape == (batch, frames - 1):
        # Align a [B, T-1] interval list with the "interval to the previous sample"
        # convention by prepending a copy of the first interval.
        delta = torch.cat((delta[:, :1], delta), dim=1)
    elif delta.shape != (batch, frames):
        raise ValueError("delta_t must be scalar, [B, T] or [B, T - 1]")
    if valid is None:
        valid = torch.ones(batch, frames, joints, dtype=torch.bool, device=pose.device)
    elif valid.shape != (batch, frames, joints):
        raise ValueError("valid must have shape [B, T, J]")
    valid = valid.bool() & torch.isfinite(pose).all(dim=-1)
    safe_pose = torch.where(valid[..., None], pose, torch.zeros_like(pose))

    velocity = torch.zeros_like(safe_pose)
    velocity_valid = torch.zeros_like(valid)
    if frames > 1:
        interval = delta[:, 1:].clamp_min(min_delta_t)[:, :, None, None]
        finite_dt = torch.isfinite(delta[:, 1:]) & (delta[:, 1:] > 0)
        pair_valid = valid[:, 1:] & valid[:, :-1] & finite_dt[..., None]
        difference = (safe_pose[:, 1:] - safe_pose[:, :-1]) / interval
        velocity[:, 1:] = torch.where(pair_valid[..., None], difference, torch.zeros_like(difference))
        velocity_valid[:, 1:] = pair_valid
        if boundary == "replicate":
            velocity[:, 0] = velocity[:, 1]
            velocity_valid[:, 0] = velocity_valid[:, 1] & valid[:, 0]
    return velocity, velocity_valid
