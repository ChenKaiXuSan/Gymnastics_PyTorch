"""Self-supervised objectives of the cycle-aware fusion model.

Two generations of objectives coexist; ``configs/cycle_aware/loss`` selects
the weights.

Version 2 (default, ``loss/v2.yaml``) removes every target that is a
function of the current window's own inputs:

    L = w_c * L_cycle + w_r * L_rel + w_p * L_period + w_s * L_sym + w_d * L_res

    L_cycle   confidence-weighted Huber distance between P_hat and the
              leave-one-cycle-out cross-cycle target (cycle_target.py):
              the same phase in the person's OTHER cycles, both views,
              robustly aggregated.  Main pose-level supervision.
    L_rel     cross-entropy on the reliability logits where synthetic
              corruption damaged exactly one view (label = the other view);
              trains w_A / w_B without any absolute 3D target.
    L_period  1 - cos between the motion feature F_M at phase phi of cycle i
              and at the same phase of cycle i + 1 (feature level).
    L_sym     1 - cos between F_M(phi, j) and F_M(phi + 0.5, mirror(j)) in the
              same cycle, mirror = left/right joint swap; the two halves of a
              trunk-rotation cycle are mirror states (checked on the private
              data: twist +0.36 / -0.05 / -0.38 / +0.10 rad at phases
              0 / 0.25 / 0.5 / 0.75).  Feature level, small weight.
    L_res     L1 norm of Delta_P; with L_recovery gone this is what keeps the
              refinement anchored to the measurements.

Version 1 (``loss/v1_recovery.yaml``) keeps the original recovery objective
below; its terms are still available (weights) so v1 runs stay reproducible.

    L = w_rec * L_recovery + w_per * L_periodicity + w_sym * L_symmetry
        + w_half * L_half_symmetry + w_res * L_residual

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

    On clean inputs this target *is* the two-view average, so a model trained
    with it alone has no reason to deviate from the average when both views
    look plausible (observed on the private data: learned == arithmetic).
    ``recovery_target = "reference"`` replaces the target by an external
    reference pose wherever one is attached to the training window (FreeMan
    multi-view references, Unity ground truth; never the triangulated
    pseudo-reference of the private data, which is derived from the same
    views).  The reference lives in its own world frame, so it is first
    brought into the frame of the (detached) weighted base pose by a
    per-frame similarity (Procrustes) transform; joints without a reference
    fall back to the pseudo-target.  This gives the reliability head and the
    residual a real signal about which view is closer to the truth.

Half-cycle (turn-around) symmetry (``L_half_symmetry``):
    Cycles with a recorded middle are resampled so that the middle sits at
    phase 0.5; the return half is then approximately the time reversal of the
    outward half, i.e. sample ``k`` of a cycle mirrors sample ``S - k``:

        L_half_symmetry = mean_k |P_hat(start + k) - P_hat(start + S - k)|^2,  0 < k < S/2

    It is a measurement-quality prior (cycle shape), off by default; real
    motion has some hysteresis, so its weight must stay small.

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
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .outputs import PoseFusionOutput
from .skeleton import CommonSkeleton

RECOVERY_KINDS = ("smooth_l1", "l1", "l2")
RECOVERY_TARGETS = ("pseudo", "reference")
RESIDUAL_NORMS = ("l1", "l2")


@dataclass(frozen=True)
class LossConfig:
    """Loss weights and thresholds (see ``configs/cycle_aware/loss``).

    Attributes:
        recovery_weight: Weight of the recovery term.
        recovery_kind: ``"smooth_l1"``, ``"l1"`` or ``"l2"``.
        recovery_target: ``"pseudo"`` (clean two-view target, label-free) or
            ``"reference"`` (attached reference pose where available).
        recovery_beta: Transition point of the smooth-L1 penalty (canonical units).
        consensus_distance: Maximum clean-view disagreement for an averaged target.
        periodicity_weight: Weight of the periodicity term.
        symmetry_weight: Weight of the bilateral bone-length term.
        half_symmetry_weight: Weight of the half-cycle (turn-around) symmetry term (positions, v1).
        residual_weight: Weight of the residual regulariser.
        residual_norm: ``"l1"`` (v2) or ``"l2"`` (v1) residual penalty.
        cycle_weight: Weight of the cross-cycle pose term (v2 main supervision).
        cycle_kind: Penalty of the cross-cycle term (``smooth_l1`` / ``l1`` / ``l2``).
        cycle_beta: Transition point of its smooth-L1 penalty (canonical units).
        reliability_weight: Weight of the corruption-labelled reliability term.
        feature_periodicity_weight: Weight of the feature-level periodicity term.
        feature_symmetry_weight: Weight of the feature-level half-cycle mirror term.
    """

    recovery_weight: float = 0.0
    recovery_kind: str = "smooth_l1"
    recovery_target: str = "pseudo"
    recovery_beta: float = 0.05
    consensus_distance: float = 0.15
    periodicity_weight: float = 0.0
    symmetry_weight: float = 0.0
    half_symmetry_weight: float = 0.0
    residual_weight: float = 0.01
    residual_norm: str = "l1"
    cycle_weight: float = 1.0
    cycle_kind: str = "smooth_l1"
    cycle_beta: float = 0.05
    reliability_weight: float = 0.05
    feature_periodicity_weight: float = 0.1
    feature_symmetry_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.recovery_kind not in RECOVERY_KINDS or self.cycle_kind not in RECOVERY_KINDS:
            raise ValueError(f"recovery_kind and cycle_kind must be one of {RECOVERY_KINDS}")
        if self.recovery_target not in RECOVERY_TARGETS:
            raise ValueError(f"recovery_target must be one of {RECOVERY_TARGETS}")
        if self.residual_norm not in RESIDUAL_NORMS:
            raise ValueError(f"residual_norm must be one of {RESIDUAL_NORMS}")
        if min(self.weights.values()) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.recovery_beta <= 0 or self.cycle_beta <= 0 or self.consensus_distance < 0:
            raise ValueError("betas must be positive and consensus_distance non-negative")

    @property
    def weights(self) -> dict[str, float]:
        """Weight of every term, keyed like :meth:`LossBreakdown.as_dict`."""
        return {
            "recovery": self.recovery_weight,
            "periodicity": self.periodicity_weight,
            "symmetry": self.symmetry_weight,
            "half_symmetry": self.half_symmetry_weight,
            "residual": self.residual_weight,
            "cycle": self.cycle_weight,
            "reliability": self.reliability_weight,
            "feature_periodicity": self.feature_periodicity_weight,
            "feature_symmetry": self.feature_symmetry_weight,
        }

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
        recovery: Unweighted recovery term (v1).
        periodicity: Unweighted position periodicity term (v1).
        symmetry: Unweighted bilateral bone-length term (v1).
        half_symmetry: Unweighted position half-cycle term (v1).
        residual: Unweighted residual regulariser.
        cycle: Unweighted cross-cycle pose term (v2).
        reliability: Unweighted corruption-labelled reliability term (v2).
        feature_periodicity: Unweighted feature periodicity term (v2).
        feature_symmetry: Unweighted feature half-cycle mirror term (v2).
        total: Weighted sum used for optimisation.
    """

    recovery: Tensor
    periodicity: Tensor
    symmetry: Tensor
    half_symmetry: Tensor
    residual: Tensor
    cycle: Tensor
    reliability: Tensor
    feature_periodicity: Tensor
    feature_symmetry: Tensor
    total: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Dictionary view, convenient for logging."""
        return {
            "recovery": self.recovery,
            "periodicity": self.periodicity,
            "symmetry": self.symmetry,
            "half_symmetry": self.half_symmetry,
            "residual": self.residual,
            "cycle": self.cycle,
            "reliability": self.reliability,
            "feature_periodicity": self.feature_periodicity,
            "feature_symmetry": self.feature_symmetry,
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


def reference_target(
    reference: Tensor,
    reference_valid: Tensor,
    anchor: Tensor,
    anchor_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Bring a world-frame reference into the frame of ``anchor`` per frame.

    A similarity (Procrustes) transform is fitted per frame on the joints
    valid in both, from the reference onto the detached anchor, so the target
    keeps the reference's internal configuration but the anchor's rotation,
    translation and scale.  Frames with fewer than three usable joints get no
    target.

    Args:
        reference: ``[B, T, J, 3]`` reference pose (world frame, any units).
        reference_valid: ``[B, T, J]`` validity of the reference.
        anchor: ``[B, T, J, 3]`` pose defining the target frame (detached).
        anchor_valid: ``[B, T, J]`` validity of the anchor.

    Returns:
        Tuple ``(target, target_valid)`` with shapes ``[B, T, J, 3]`` and
        ``[B, T, J]``.
    """
    from .metrics import procrustes_align

    batch, frames, joints, _ = reference.shape
    usable = reference_valid.bool() & anchor_valid.bool()
    with torch.no_grad():
        aligned = procrustes_align(
            reference.reshape(batch * frames, joints, 3),
            anchor.detach().reshape(batch * frames, joints, 3),
            usable.reshape(batch * frames, joints),
        ).reshape(batch, frames, joints, 3)
        enough = usable.sum(dim=-1, keepdim=True) >= 3
    target_valid = usable & enough
    return torch.where(target_valid[..., None], aligned, torch.zeros_like(aligned)), target_valid


def half_symmetry_loss(prediction: Tensor, valid: Tensor, phase: Tensor, phase_valid: Tensor, cycle_index: Tensor, half_index: Tensor, samples_per_cycle: int) -> Tensor:
    """Time-reversal symmetry about the cycle middle.

    Sample ``k`` of a cycle (``0 < k < S/2``) is compared with sample
    ``S - k`` of the same cycle.  Only cycles resampled with a known middle
    (``half_index >= 0``) participate.

    Args:
        prediction: ``[B, T, J, 3]`` fused pose.
        valid: ``[B, T, J]`` validity of the fused pose.
        phase: ``[B, T]`` phase in ``[0, 1)`` (multiples of ``1/S``).
        phase_valid: ``[B, T]`` bool.
        cycle_index: ``[B, T]`` cycle index (``-1`` outside cycles).
        half_index: ``[B, T]`` half index (``-1`` when no middle is known).
        samples_per_cycle: ``S``.

    Returns:
        Scalar loss (zero when no mirror pairs exist).
    """
    batch, frames = phase.shape
    S = int(samples_per_cycle)
    k = torch.round(phase * S).long()
    mirror = torch.arange(frames, device=phase.device)[None, :] - 2 * k + S  # t' = t - k + (S - k)
    first_half = phase_valid & (half_index == 0) & (k > 0) & (mirror >= 0) & (mirror < frames)
    mirror_safe = mirror.clamp(0, frames - 1)
    same_cycle = torch.gather(cycle_index, 1, mirror_safe) == cycle_index
    mirrored_valid = torch.gather(valid.bool(), 1, mirror_safe[..., None].expand(-1, -1, valid.shape[-1]))
    mirrored = torch.gather(prediction, 1, mirror_safe[..., None, None].expand(-1, -1, *prediction.shape[2:]))
    mask = (first_half & same_cycle)[..., None] & valid.bool() & mirrored_valid
    return masked_mean((prediction - mirrored).square().sum(dim=-1), mask)


def cycle_loss(prediction: Tensor, valid: Tensor, target: Tensor, confidence: Tensor, *, kind: str, beta: float) -> Tensor:
    """Confidence-weighted penalty between ``P_hat`` and the cross-cycle target.

        L_cycle = sum_{t,j} C(t,j) rho(P_hat(t,j) - P_ref(t,j)) / sum_{t,j} C(t,j)

    Args:
        prediction: ``[B, T, J, 3]`` fused pose.
        valid: ``[B, T, J]`` validity of the fused pose.
        target: ``[B, T, J, 3]`` leave-one-cycle-out target.
        confidence: ``[B, T, J]`` confidence in ``[0, 1]`` (0 = no target).
        kind: Penalty kind.
        beta: Smooth-L1 transition point.

    Returns:
        Scalar loss (zero when no confident target exists).
    """
    weight = torch.where(valid.bool(), confidence, torch.zeros_like(confidence))
    penalty = _penalty(prediction - target, kind, beta)
    usable = (weight > 0) & torch.isfinite(penalty)
    weight = torch.where(usable, weight, torch.zeros_like(weight))
    return (weight * torch.where(usable, penalty, torch.zeros_like(penalty))).sum() / weight.sum().clamp_min(1e-6)


def reliability_loss(
    logits: Tensor,
    corruption_mask_a: Tensor,
    corruption_mask_b: Tensor,
    valid_a: Tensor,
    valid_b: Tensor,
) -> Tensor:
    """Cross-entropy on the reliability logits where exactly one view was corrupted.

    The label is the undamaged view (0 = A, 1 = B).  Joints where both or
    neither view was corrupted, and joints that the corruption removed
    entirely (invalid in a view, where the softmax is masked anyway), carry
    no label.

    Args:
        logits: ``[B, T, J, 2]`` raw reliability logits.
        corruption_mask_a: ``[B, T, J]`` joints altered in View A.
        corruption_mask_b: ``[B, T, J]`` joints altered in View B.
        valid_a: ``[B, T, J]`` validity of View A after corruption.
        valid_b: ``[B, T, J]`` validity of View B after corruption.

    Returns:
        Scalar loss (zero when no labelled joint exists).
    """
    a, b = corruption_mask_a.bool(), corruption_mask_b.bool()
    labelled = (a ^ b) & valid_a.bool() & valid_b.bool()
    if not bool(labelled.any()):
        return logits.new_zeros(())
    label = b.long()  # B corrupted -> trust A (0); A corrupted -> trust B (1)
    label = torch.where(a, torch.ones_like(label), torch.zeros_like(label))
    loss = torch.nn.functional.cross_entropy(logits[labelled], label[labelled], reduction="mean")
    return loss


def _cosine_distance(left: Tensor, right: Tensor) -> Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(left, right, dim=-1, eps=1e-6)


def feature_periodicity_loss(feature: Tensor, valid: Tensor, cycle_index: Tensor, samples_per_cycle: int) -> Tensor:
    """``1 - cos(F(t), F(t + S))`` for samples one cycle apart in consecutive cycles.

    Args:
        feature: ``[B, T, J, D]`` motion feature.
        valid: ``[B, T, J]`` validity of the feature.
        cycle_index: ``[B, T]`` cycle index (``-1`` outside cycles).
        samples_per_cycle: ``S``.

    Returns:
        Scalar loss (zero when no consecutive-cycle pairs exist).
    """
    frames = feature.shape[1]
    shift = int(samples_per_cycle)
    if shift <= 0 or shift >= frames:
        return feature.new_zeros(())
    current, following = cycle_index[:, :-shift], cycle_index[:, shift:]
    pair_valid = (current >= 0) & (following == current + 1)
    mask = pair_valid[..., None] & valid[:, :-shift].bool() & valid[:, shift:].bool()
    return masked_mean(_cosine_distance(feature[:, :-shift], feature[:, shift:]), mask)


def feature_symmetry_loss(
    feature: Tensor,
    valid: Tensor,
    cycle_index: Tensor,
    half_index: Tensor,
    samples_per_cycle: int,
    left_right_pairs: Sequence[tuple[int, int]],
) -> Tensor:
    """``1 - cos(F(t, j), F(t + S/2, mirror(j)))`` within one cycle.

    Only the outward half (``half_index == 0``) is used as the query so every
    pair is counted once; ``mirror`` swaps left/right joints and leaves
    mid-line joints in place.  Requires cycles resampled with a known middle
    (mid at ``S/2``).

    Args:
        feature: ``[B, T, J, D]`` motion feature.
        valid: ``[B, T, J]`` validity of the feature.
        cycle_index: ``[B, T]`` cycle index.
        half_index: ``[B, T]`` half index (``-1`` when no middle is known).
        samples_per_cycle: ``S`` (even).
        left_right_pairs: ``(left, right)`` joint index pairs.

    Returns:
        Scalar loss (zero when no mirror pairs exist).
    """
    frames, joints = feature.shape[1], feature.shape[2]
    half = int(samples_per_cycle) // 2
    if half <= 0 or half >= frames:
        return feature.new_zeros(())
    mirror = torch.arange(joints, device=feature.device)
    for left, right in left_right_pairs:
        mirror[left], mirror[right] = right, left
    query = (half_index[:, :-half] == 0) & (cycle_index[:, :-half] == cycle_index[:, half:]) & (cycle_index[:, :-half] >= 0)
    partner = feature[:, half:][:, :, mirror]
    partner_valid = valid[:, half:][:, :, mirror].bool()
    mask = query[..., None] & valid[:, :-half].bool() & partner_valid
    return masked_mean(_cosine_distance(feature[:, :-half], partner), mask)


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
        batch: :class:`~gymnastics.fusion.sample.FusionBatch`.
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
    if config.recovery_target == "reference":
        reference_valid = batch.get("reference_valid")
        if reference_valid is not None and bool(reference_valid.any()):
            ref_target, ref_valid = reference_target(batch["reference"], reference_valid & frame_mask[..., None], output.base_pose, fused_valid)
            # Reference where available, pseudo-target elsewhere.
            target = torch.where(ref_valid[..., None], ref_target, target)
            target_valid = target_valid | ref_valid
    recovery = recovery_loss(output.pose, fused_valid, target, target_valid, kind=config.recovery_kind, beta=config.recovery_beta)
    cycle_index = batch.get("cycle_index")
    if cycle_index is None:
        periodicity = output.pose.new_zeros(())
        half_symmetry = output.pose.new_zeros(())
    else:
        cycle_index = cycle_index.to(output.pose.device)
        periodicity = periodicity_loss(output.pose, fused_valid, cycle_index, samples_per_cycle)
        half_index = batch.get("half_index")
        if config.half_symmetry_weight > 0 and half_index is not None:
            half_symmetry = half_symmetry_loss(output.pose, fused_valid, batch["phase"].to(output.pose.device), batch["phase_valid"].to(output.pose.device), cycle_index, half_index.to(output.pose.device), samples_per_cycle)
        else:
            half_symmetry = output.pose.new_zeros(())
    symmetry = symmetry_loss(output.pose, fused_valid, skeleton)
    if config.residual_norm == "l1":
        residual = masked_mean(output.delta_pose.abs().sum(dim=-1), fused_valid)
    else:
        residual = residual_loss(output.delta_pose, fused_valid)
    # ---- version 2 terms
    zero = output.pose.new_zeros(())
    cycle_target = batch.get("cycle_target")
    if config.cycle_weight > 0 and cycle_target is not None:
        cycle = cycle_loss(output.pose, fused_valid, cycle_target.to(output.pose.device), batch["cycle_confidence"].to(output.pose.device), kind=config.cycle_kind, beta=config.cycle_beta)
    else:
        cycle = zero
    if config.reliability_weight > 0 and "corruption_mask_a" in batch:
        reliability = reliability_loss(output.reliability_logits, batch["corruption_mask_a"], batch["corruption_mask_b"], batch["valid_a"] & frame_mask[..., None], batch["valid_b"] & frame_mask[..., None])
    else:
        reliability = zero
    feature_periodicity = zero
    feature_symmetry = zero
    if cycle_index is not None:
        views = ((output.motion_feature_a, batch["valid_a"] & frame_mask[..., None]), (output.motion_feature_b, batch["valid_b"] & frame_mask[..., None]))
        if config.feature_periodicity_weight > 0:
            feature_periodicity = sum(feature_periodicity_loss(f, v, cycle_index, samples_per_cycle) for f, v in views) / len(views)
        half_index = batch.get("half_index")
        if config.feature_symmetry_weight > 0 and half_index is not None:
            feature_symmetry = sum(feature_symmetry_loss(f, v, cycle_index, half_index.to(output.pose.device), samples_per_cycle, skeleton.left_right_pairs) for f, v in views) / len(views)
    total = (
        config.recovery_weight * recovery
        + config.periodicity_weight * periodicity
        + config.symmetry_weight * symmetry
        + config.half_symmetry_weight * half_symmetry
        + config.residual_weight * residual
        + config.cycle_weight * cycle
        + config.reliability_weight * reliability
        + config.feature_periodicity_weight * feature_periodicity
        + config.feature_symmetry_weight * feature_symmetry
    )
    return LossBreakdown(
        recovery=recovery, periodicity=periodicity, symmetry=symmetry, half_symmetry=half_symmetry, residual=residual,
        cycle=cycle, reliability=reliability, feature_periodicity=feature_periodicity, feature_symmetry=feature_symmetry, total=total,
    )
