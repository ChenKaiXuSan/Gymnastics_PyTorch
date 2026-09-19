"""Self-supervised objectives of the cycle-aware fusion model.

No 3D ground truth is used anywhere in training.  Four objectives shape the
fused pose ``P_hat`` and the residual ``Delta_P`` returned by the model:

    L = w_rec * L_recovery + w_per * L_periodicity + w_sym * L_symmetry + w_res * L_residual

Recovery / reconstruction (``L_recovery``):
    The model receives *corrupted* views (see :mod:`.corruptions`) and must
    reproduce a pseudo-target ``P*`` built from the *clean* views:

        P*(t, j) = 0.5 (P_A + P_B)   if both clean views are valid and
                                     |P_A - P_B| <= consensus_distance
                 = P_A or P_B        if exactly one clean view is valid
                 = undefined         otherwise (excluded from the loss)

        L_recovery = mean_{valid} rho(P_hat - P*)

    with ``rho`` a smooth-L1 (Huber), L1 or L2 penalty.  The consensus
    threshold prevents the average of two strongly disagreeing views, which is
    usually wrong for both, from becoming a target.  Frames the corruption did
    not touch contribute an identity term that keeps clean inputs unchanged.

Periodicity (``L_periodicity``):
    In phase-normalised windows, sample ``t`` and sample ``t + S`` (``S =
    samples_per_cycle``) share the same phase in consecutive cycles:

        L_periodicity = mean_{t : cycle(t+S) = cycle(t) + 1} |P_hat(t) - P_hat(t + S)|^2

    It regularises the output towards cycle-consistent motion and is only
    active where phase is valid, so unannotated data are unaffected.

Bilateral symmetry (``L_symmetry``):
    Mirrored bones of a human body have equal length; a single-view
    reconstruction error usually violates this.  For every ``(left bone,
    right bone)`` pair of the skeleton

        L_symmetry = mean | ||b_left|| - ||b_right|| |

    computed on ``P_hat`` where both bones are valid.  This is a weak, static
    prior (it does not assume left/right *motion* symmetry, which gymnastics
    trunk rotations do not have).

Residual regularisation (``L_residual``):
    ``L_residual = mean_{valid} ||Delta_P||^2`` keeps the learned correction
    small so the output stays anchored to the measurements; the fused pose is
    then explainable as "weighted measurement plus a small correction".

Shapes:
    All inputs follow ``[B, T, J, *]``; masks are ``[B, T, J]`` or ``[B, T]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .outputs import PoseFusionOutput
from .skeleton import CommonSkeleton

RECOVERY_KINDS = ("smooth_l1", "l1", "l2")


@dataclass(frozen=True)
class LossConfig:
    """Loss weights and thresholds (see ``configs/cycle_aware/loss``).

    Attributes:
        recovery_weight: Weight of the recovery term.
        recovery_kind: ``"smooth_l1"``, ``"l1"`` or ``"l2"``.
        recovery_beta: Transition point of the smooth-L1 penalty (canonical units).
        consensus_distance: Maximum clean-view disagreement for an averaged target.
        periodicity_weight: Weight of the periodicity term.
        symmetry_weight: Weight of the bilateral bone-length term.
        residual_weight: Weight of the residual regulariser.
    """

    recovery_weight: float = 1.0
    recovery_kind: str = "smooth_l1"
    recovery_beta: float = 0.05
    consensus_distance: float = 0.15
    periodicity_weight: float = 0.1
    symmetry_weight: float = 0.1
    residual_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.recovery_kind not in RECOVERY_KINDS:
            raise ValueError(f"recovery_kind must be one of {RECOVERY_KINDS}")
        if min(self.recovery_weight, self.periodicity_weight, self.symmetry_weight, self.residual_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.recovery_beta <= 0 or self.consensus_distance < 0:
            raise ValueError("recovery_beta must be positive and consensus_distance non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "LossConfig" | None) -> "LossConfig":
        """Build from a mapping (e.g. an OmegaConf node); ``None`` gives defaults."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)  # type: ignore[assignment]
        except ImportError:  # pragma: no cover
            pass
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class LossBreakdown:
    """The individual objectives and their weighted sum.

    Attributes:
        recovery: Unweighted recovery term.
        periodicity: Unweighted periodicity term.
        symmetry: Unweighted bilateral symmetry term.
        residual: Unweighted residual regulariser.
        total: Weighted sum used for optimisation.
    """

    recovery: Tensor
    periodicity: Tensor
    symmetry: Tensor
    residual: Tensor
    total: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Dictionary view, convenient for logging."""
        return {
            "recovery": self.recovery,
            "periodicity": self.periodicity,
            "symmetry": self.symmetry,
            "residual": self.residual,
            "total": self.total,
        }


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values`` over true entries of ``mask`` (zero when empty)."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have equal shapes")
    usable = mask.bool() & torch.isfinite(values)
    safe = torch.where(usable, values, torch.zeros_like(values))
    return safe.sum() / usable.sum().clamp_min(1).to(dtype=values.dtype)


def pseudo_target(
    clean_a: Tensor,
    clean_b: Tensor,
    valid_a: Tensor,
    valid_b: Tensor,
    *,
    consensus_distance: float,
) -> tuple[Tensor, Tensor]:
    """Build the label-free recovery target from the clean views.

    Args:
        clean_a: ``[B, T, J, 3]`` clean View A.
        clean_b: ``[B, T, J, 3]`` clean View B.
        valid_a: ``[B, T, J]`` validity of View A.
        valid_b: ``[B, T, J]`` validity of View B.
        consensus_distance: Maximum disagreement for averaging.

    Returns:
        Tuple ``(target, target_valid)`` with shapes ``[B, T, J, 3]`` and
        ``[B, T, J]``.
    """
    valid_a, valid_b = valid_a.bool(), valid_b.bool()
    both = valid_a & valid_b
    distance = torch.linalg.vector_norm(clean_a - clean_b, dim=-1)
    consensus = both & (distance <= consensus_distance)
    only_a = valid_a & ~valid_b
    only_b = valid_b & ~valid_a
    target = torch.where(consensus[..., None], 0.5 * (clean_a + clean_b), torch.zeros_like(clean_a))
    target = torch.where(only_a[..., None], clean_a, target)
    target = torch.where(only_b[..., None], clean_b, target)
    return target, consensus | only_a | only_b


def _penalty(difference: Tensor, kind: str, beta: float) -> Tensor:
    """Per-joint scalar penalty of a ``[..., 3]`` difference."""
    if kind == "l2":
        return difference.square().sum(dim=-1)
    if kind == "l1":
        return difference.abs().sum(dim=-1)
    absolute = difference.abs()
    huber = torch.where(absolute < beta, 0.5 * absolute.square() / beta, absolute - 0.5 * beta)
    return huber.sum(dim=-1)


def recovery_loss(prediction: Tensor, prediction_valid: Tensor, target: Tensor, target_valid: Tensor, *, kind: str, beta: float) -> Tensor:
    """Recovery term ``mean rho(P_hat - P*)`` over jointly valid joints."""
    mask = prediction_valid.bool() & target_valid.bool()
    return masked_mean(_penalty(prediction - target, kind, beta), mask)


def periodicity_loss(prediction: Tensor, valid: Tensor, cycle_index: Tensor, samples_per_cycle: int) -> Tensor:
    """Squared distance between samples one cycle apart in consecutive cycles.

    Args:
        prediction: ``[B, T, J, 3]`` fused pose.
        valid: ``[B, T, J]`` validity of the fused pose.
        cycle_index: ``[B, T]`` cycle index per sample (``-1`` outside cycles).
        samples_per_cycle: Number of samples per normalised cycle ``S``.

    Returns:
        Scalar loss (zero when no consecutive-cycle pairs exist).
    """
    frames = prediction.shape[1]
    shift = int(samples_per_cycle)
    if shift <= 0 or shift >= frames:
        return prediction.new_zeros(())
    current, following = cycle_index[:, :-shift], cycle_index[:, shift:]
    pair_valid = (current >= 0) & (following == current + 1)
    mask = pair_valid[..., None] & valid[:, :-shift].bool() & valid[:, shift:].bool()
    difference = (prediction[:, :-shift] - prediction[:, shift:]).square().sum(dim=-1)
    return masked_mean(difference, mask)


def symmetry_loss(prediction: Tensor, valid: Tensor, skeleton: CommonSkeleton) -> Tensor:
    """Absolute difference of mirrored bone lengths on the fused pose."""
    pairs = skeleton.bilateral_bone_pairs()
    if not pairs:
        return prediction.new_zeros(())
    starts = torch.tensor([b[0] for b in skeleton.bones], device=prediction.device)
    ends = torch.tensor([b[1] for b in skeleton.bones], device=prediction.device)
    lengths = torch.linalg.vector_norm(prediction[:, :, ends] - prediction[:, :, starts], dim=-1)
    bone_valid = valid.bool()[:, :, starts] & valid.bool()[:, :, ends]
    left = torch.tensor([p[0] for p in pairs], device=prediction.device)
    right = torch.tensor([p[1] for p in pairs], device=prediction.device)
    difference = (lengths[:, :, left] - lengths[:, :, right]).abs()
    return masked_mean(difference, bone_valid[:, :, left] & bone_valid[:, :, right])


def residual_loss(delta: Tensor, valid: Tensor) -> Tensor:
    """Mean squared norm of ``Delta_P`` over valid joints."""
    return masked_mean(delta.square().sum(dim=-1), valid.bool())


def compute_losses(
    output: PoseFusionOutput,
    batch: Mapping[str, Any],
    *,
    skeleton: CommonSkeleton,
    config: LossConfig,
    samples_per_cycle: int,
) -> LossBreakdown:
    """Evaluate every objective on one batch.

    Args:
        output: Model output computed from ``batch["pose_a"]`` and
            ``batch["pose_b"]`` (possibly corrupted).
        batch: :class:`~gymnastics.fusion.cycle_aware.sample.FusionBatch`.
            When ``clean_a``/``clean_b`` are present they define the recovery
            target; otherwise the inputs themselves do (consistency mode).
        skeleton: Common skeleton for the symmetry term.
        config: Weights and thresholds.
        samples_per_cycle: ``S`` for the periodicity shift.

    Returns:
        :class:`LossBreakdown` with finite scalar tensors.
    """
    clean_a = batch.get("clean_a", batch["pose_a"])
    clean_b = batch.get("clean_b", batch["pose_b"])
    clean_valid_a = batch.get("clean_valid_a", batch["valid_a"])
    clean_valid_b = batch.get("clean_valid_b", batch["valid_b"])
    frame_mask = batch.get("frame_mask")
    if frame_mask is None:
        frame_mask = torch.ones(output.pose.shape[:2], dtype=torch.bool, device=output.pose.device)
    target, target_valid = pseudo_target(clean_a, clean_b, clean_valid_a, clean_valid_b, consensus_distance=config.consensus_distance)
    target_valid = target_valid & frame_mask[..., None]
    fused_valid = output.valid & frame_mask[..., None]
    recovery = recovery_loss(output.pose, fused_valid, target, target_valid, kind=config.recovery_kind, beta=config.recovery_beta)
    cycle_index = batch.get("cycle_index")
    if cycle_index is None:
        periodicity = output.pose.new_zeros(())
    else:
        periodicity = periodicity_loss(output.pose, fused_valid, cycle_index.to(output.pose.device), samples_per_cycle)
    symmetry = symmetry_loss(output.pose, fused_valid, skeleton)
    residual = residual_loss(output.delta_pose, fused_valid)
    total = (
        config.recovery_weight * recovery
        + config.periodicity_weight * periodicity
        + config.symmetry_weight * symmetry
        + config.residual_weight * residual
    )
    return LossBreakdown(recovery=recovery, periodicity=periodicity, symmetry=symmetry, residual=residual, total=total)
