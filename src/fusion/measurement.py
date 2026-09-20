"""Measurement-quality metrics: does fusion preserve the motion it measures?

Averaging two noisy views lowers positional error but also shrinks motion
extremes (range of motion, peak angular velocity), which is what the
downstream gymnastics analysis actually measures.  The earlier
rotation-aware study found that learning bought measurement preservation
(ROM retention 1.000 versus 0.917 for plain averaging) rather than accuracy.
This module computes those quantities for the cycle-aware model so that
symmetry / periodicity objectives can be judged on what they are meant to
improve.

Trunk twist:
    The signed angle between the shoulder line ``s`` and the hip line ``h``
    around the torso axis ``u`` (pelvis -> shoulder centre):

        h_p = h - (h.u) u,   s_p = s - (s.u) u
        theta(t) = atan2( (h_p x s_p) . u,  h_p . s_p )

    unwrapped over time.  The definition is frame independent, so it applies
    unchanged to the canonical views, the fused pose and a world-frame
    reference (it needs both hips and both shoulders, which the COCO17
    reference of FreeMan provides; the Unity reference lacks separate hips).

Per-cycle quantities (cycles with valid ``cycle_index``):
    ROM       = max theta - min theta over the cycle
    peak_omega = max |d theta / dt|              (physical rad/s, uses delta_t)

Retention ratios (per window, then averaged):
    rom_retention        = ROM(fused) / mean(ROM(view A), ROM(view B))
    peak_omega_retention = the same for peak angular velocity

    A ratio of 1 means the fused motion keeps the extremes of the inputs; the
    plain average typically gives < 1.  Against a reference (when attached)
    the analogous ``*_vs_reference`` ratios use ROM(reference).

Shapes: poses ``[B, T, J, 3]``, masks ``[B, T, J]`` / ``[B, T]``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .skeleton import CommonSkeleton


def trunk_twist(pose: Tensor, valid: Tensor, skeleton: CommonSkeleton) -> tuple[Tensor, Tensor]:
    """Unwrapped shoulder-versus-hip twist angle per frame (frame independent).

    Args:
        pose: ``[B, T, J, 3]`` keypoints in any frame.
        valid: ``[B, T, J]`` validity.
        skeleton: Common skeleton (provides hip and shoulder indices).

    Returns:
        Tuple ``(theta, theta_valid)`` with shapes ``[B, T]``; ``theta`` is
        unwrapped along time and zero where invalid.
    """
    ls, rs = skeleton.index("left-shoulder"), skeleton.index("right-shoulder")
    lh, rh = skeleton.left_hip_index, skeleton.right_hip_index
    hip = pose[:, :, rh] - pose[:, :, lh]
    shoulder = pose[:, :, rs] - pose[:, :, ls]
    axis = 0.5 * (pose[:, :, rs] + pose[:, :, ls]) - 0.5 * (pose[:, :, rh] + pose[:, :, lh])
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)
    hip_p = hip - (hip * axis).sum(-1, keepdim=True) * axis
    shoulder_p = shoulder - (shoulder * axis).sum(-1, keepdim=True) * axis
    theta = torch.atan2((torch.cross(hip_p, shoulder_p, dim=-1) * axis).sum(-1), (hip_p * shoulder_p).sum(-1))
    theta_valid = valid[:, :, ls].bool() & valid[:, :, rs].bool() & valid[:, :, lh].bool() & valid[:, :, rh].bool()
    theta_valid = theta_valid & (torch.linalg.vector_norm(hip_p, dim=-1) > 1e-4) & (torch.linalg.vector_norm(shoulder_p, dim=-1) > 1e-4)
    # Unwrap: accumulate wrapped differences so consecutive turns stay continuous.
    diff = theta[:, 1:] - theta[:, :-1]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    unwrapped = torch.cat((theta[:, :1], theta[:, :1] + torch.cumsum(diff, dim=1)), dim=1)
    return torch.where(theta_valid, unwrapped, torch.zeros_like(unwrapped)), theta_valid


def per_cycle_extremes(theta: Tensor, theta_valid: Tensor, cycle_index: Tensor, delta_t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ROM and peak angular velocity per cycle present in each window.

    Args:
        theta: ``[B, T]`` unwrapped twist.
        theta_valid: ``[B, T]`` bool.
        cycle_index: ``[B, T]`` cycle index (``-1`` outside cycles).
        delta_t: ``[B, T]`` physical sample intervals (seconds).

    Returns:
        Tuple ``(rom, peak_omega, cycle_valid)`` with shape ``[B, C]`` where
        ``C`` is the largest number of cycles in the window; ``cycle_valid``
        marks cycles with at least four valid samples.
    """
    batch, frames = theta.shape
    n_cycles = int(cycle_index.max().item()) + 1 if cycle_index.numel() and cycle_index.max() >= 0 else 0
    rom = theta.new_zeros((batch, max(n_cycles, 1)))
    peak = theta.new_zeros((batch, max(n_cycles, 1)))
    cycle_valid = torch.zeros((batch, max(n_cycles, 1)), dtype=torch.bool, device=theta.device)
    if n_cycles == 0:
        return rom, peak, cycle_valid
    omega = torch.zeros_like(theta)
    omega[:, 1:] = (theta[:, 1:] - theta[:, :-1]) / delta_t[:, 1:].clamp_min(1e-4)
    omega_valid = torch.zeros_like(theta_valid)
    omega_valid[:, 1:] = theta_valid[:, 1:] & theta_valid[:, :-1]
    for c in range(n_cycles):
        member = (cycle_index == c) & theta_valid
        count = member.sum(dim=1)
        hi = torch.where(member, theta, torch.full_like(theta, -1e9)).max(dim=1).values
        lo = torch.where(member, theta, torch.full_like(theta, 1e9)).min(dim=1).values
        rom[:, c] = torch.where(count >= 4, hi - lo, torch.zeros_like(hi))
        peak[:, c] = torch.where(member & omega_valid, omega.abs(), torch.zeros_like(omega)).max(dim=1).values
        cycle_valid[:, c] = count >= 4
    return rom, peak, cycle_valid


def retention_ratios(
    fused: Tensor,
    fused_valid: Tensor,
    view_a: Tensor,
    valid_a: Tensor,
    view_b: Tensor,
    valid_b: Tensor,
    cycle_index: Tensor,
    delta_t: Tensor,
    skeleton: CommonSkeleton,
    *,
    reference: Tensor | None = None,
    reference_valid: Tensor | None = None,
) -> dict[str, Tensor]:
    """ROM and peak-velocity retention of ``fused`` relative to the inputs (and reference).

    Returns:
        Dictionary of scalar tensors: ``rom_retention``, ``peak_omega_retention``
        and, when a reference is given, ``rom_retention_vs_reference`` and
        ``peak_omega_retention_vs_reference``; NaN when no cycle is available.
    """
    def extremes(pose: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        theta, theta_valid = trunk_twist(pose, valid, skeleton)
        return per_cycle_extremes(theta, theta_valid, cycle_index, delta_t)

    rom_f, peak_f, ok_f = extremes(fused, fused_valid)
    rom_a, peak_a, ok_a = extremes(view_a, valid_a)
    rom_b, peak_b, ok_b = extremes(view_b, valid_b)
    ok = ok_f & ok_a & ok_b
    out: dict[str, Tensor] = {}
    denominator_rom = 0.5 * (rom_a + rom_b)
    denominator_peak = 0.5 * (peak_a + peak_b)
    usable = ok & (denominator_rom > 1e-3) & (denominator_peak > 1e-3)
    out["rom_retention"] = (rom_f / denominator_rom.clamp_min(1e-6))[usable].mean() if usable.any() else fused.new_tensor(float("nan"))
    out["peak_omega_retention"] = (peak_f / denominator_peak.clamp_min(1e-6))[usable].mean() if usable.any() else fused.new_tensor(float("nan"))
    if reference is not None and reference_valid is not None and bool(reference_valid.any()):
        rom_r, peak_r, ok_r = extremes(reference, reference_valid)
        usable_r = ok_f & ok_r & (rom_r > 1e-3) & (peak_r > 1e-3)
        out["rom_retention_vs_reference"] = (rom_f / rom_r.clamp_min(1e-6))[usable_r].mean() if usable_r.any() else fused.new_tensor(float("nan"))
        out["peak_omega_retention_vs_reference"] = (peak_f / peak_r.clamp_min(1e-6))[usable_r].mean() if usable_r.any() else fused.new_tensor(float("nan"))
    return out
