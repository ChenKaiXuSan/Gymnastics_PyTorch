"""Objectives of the cycle-aware fusion model.

Two generations of objectives coexist; ``src/configs/fusion/loss`` selects
them.  Every weight and threshold is a Hydra value (``LossConfig``); nothing
is hard-coded here.

Version 2 (default, ``loss/v2.yaml``) uses no target that is a function of
the current window's own two views:

    L = w_c * L_cycle + w_r * L_rel + w_p * L_period + w_s * L_sym + w_d * L_res

    L_cycle   (main pose supervision) confidence-gated, leave-one-cycle-out,
              hierarchical view -> cycle consensus target
              (:mod:`fusion.cycle_target`) with a natural-variation dead zone
              and a robust Huber penalty:

                  d(k, j)     = |P_hat(k, j) - P_ref(k, j)|_2
                  delta(k, j) = max(minimum, scale * MAD(k, j))       (dead zone)
                  d_eff       = max(0, d - delta)
                  L_cycle     = sum C(k, j) Huber(d_eff) / sum C(k, j)

              Only deviations beyond the natural cycle-to-cycle variation of
              the other cycles are penalised, so the model is not forced to
              collapse every cycle onto the median cycle.  Where an external
              reference is attached (FreeMan / Unity training with
              ``data.train_with_reference``) it takes priority over the
              cross-cycle target: confidence 1, no dead zone (supervision
              priority: external reference > reliable cross-cycle reference >
              no pose-level supervision).
    L_rel     cross-entropy on the reliability logits where synthetic
              corruption damaged exactly one view (label = the other view).
    L_period  motion-feature periodicity between cycle i and i + 1 at equal
              phase; ``type = cosine`` (``1 - cos``) or ``type = contrastive``
              (InfoNCE: positive = same phase of the next cycle, negatives =
              phases of the next cycle farther than ``negative_phase_margin``,
              temperature ``tau``), see :func:`contrastive_periodicity_loss`.
    L_sym     ``1 - cos(F_M(phi, j), F_M(phi + 0.5, mirror(j)))`` in the same
              cycle, mirror = left/right joint swap.
    L_res     L1 norm of Delta_P; the anchor to the measurements.

Version 1 (``loss/v1_recovery.yaml``) keeps the original recovery objective
(clean two-view target under corruption) plus position-level periodicity /
bone-length symmetry / half-cycle symmetry; its terms remain available so v1
runs stay reproducible, and ``recovery.target = reference`` gives the
reference-supervised variant.

:class:`LossBreakdown` reports every term raw and weighted (``as_dict`` keys
``<term>_raw`` / ``<term>_weighted``) plus ``total`` so the coefficient and
the actual contribution can be compared in the logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .modules.weighted_fusion import depth_aware_pose_fusion
from .outputs import PoseFusionOutput
from .skeleton import CommonSkeleton

PENALTY_KINDS = ("smooth_l1", "l1", "l2")
RECOVERY_TARGETS = ("pseudo", "reference")
RESIDUAL_NORMS = ("l1", "l2")
PERIODICITY_TYPES = ("cosine", "contrastive")
DEAD_ZONE_METHODS = ("mad",)


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DeadZoneConfig:
    """Natural-variation dead zone of ``L_cycle`` (``loss.cycle.dead_zone``).

    Attributes:
        enabled: Apply the dead zone.
        method: Dispersion statistic; ``"mad"`` uses the MAD of the other
            cycles shipped as ``cycle_dispersion``.
        scale: ``delta = scale * MAD``.
        minimum: Lower bound of ``delta`` (canonical units).
        maximum: Optional upper bound of ``delta`` (``None`` = none).
    """

    enabled: bool = True
    method: str = "mad"
    scale: float = 1.0
    minimum: float = 0.0
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.method not in DEAD_ZONE_METHODS:
            raise ValueError(f"dead-zone method must be one of {DEAD_ZONE_METHODS}")
        if self.scale < 0 or self.minimum < 0 or (self.maximum is not None and self.maximum < self.minimum):
            raise ValueError("dead-zone scale/minimum must be non-negative and maximum >= minimum")


@dataclass(frozen=True)
class CycleLossConfig:
    """``L_cycle`` (``loss.cycle``).

    Attributes:
        weight: Coefficient.
        kind: Penalty (``smooth_l1`` / ``l1`` / ``l2``) applied to ``d_eff``.
        beta: Smooth-L1 transition point (canonical units).
        external_reference: Let an attached reference override the
            cross-cycle target (supervision priority).
        dead_zone: :class:`DeadZoneConfig`.
    """

    weight: float = 1.0
    kind: str = "smooth_l1"
    beta: float = 0.05
    external_reference: bool = True
    dead_zone: DeadZoneConfig = field(default_factory=DeadZoneConfig)

    def __post_init__(self) -> None:
        if self.kind not in PENALTY_KINDS or self.beta <= 0 or self.weight < 0:
            raise ValueError("cycle loss needs a valid kind, positive beta and non-negative weight")


@dataclass(frozen=True)
class ReliabilityLossConfig:
    """``L_rel`` (``loss.reliability``)."""

    weight: float = 0.02

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("reliability weight must be non-negative")


@dataclass(frozen=True)
class PeriodicityLossConfig:
    """``L_period`` (``loss.periodicity``).

    Attributes:
        weight: Coefficient.
        type: ``"cosine"`` or ``"contrastive"``.
        temperature: InfoNCE temperature (contrastive).
        negative_phase_margin: Minimum circular phase distance (in cycles,
            ``0..0.5``) for a sample of the next cycle to count as a negative.
    """

    weight: float = 0.1
    type: str = "cosine"
    temperature: float = 0.1
    negative_phase_margin: float = 0.25

    def __post_init__(self) -> None:
        if self.type not in PERIODICITY_TYPES:
            raise ValueError(f"periodicity type must be one of {PERIODICITY_TYPES}")
        if self.weight < 0 or self.temperature <= 0 or not 0.0 < self.negative_phase_margin <= 0.5:
            raise ValueError("periodicity needs non-negative weight, positive temperature and margin in (0, 0.5]")


@dataclass(frozen=True)
class SymmetryLossConfig:
    """``L_sym`` (``loss.symmetry``), feature-level half-cycle mirror term."""

    weight: float = 0.1

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("symmetry weight must be non-negative")


@dataclass(frozen=True)
class ResidualLossConfig:
    """``L_res`` (``loss.residual``)."""

    weight: float = 0.01
    norm: str = "l1"

    def __post_init__(self) -> None:
        if self.norm not in RESIDUAL_NORMS or self.weight < 0:
            raise ValueError(f"residual norm must be one of {RESIDUAL_NORMS} and weight non-negative")


@dataclass(frozen=True)
class RecoveryLossConfig:
    """Version-1 recovery objective (``loss.recovery``), weight 0 by default.

    Attributes:
        weight: Coefficient.
        kind: Penalty kind.
        target: ``"pseudo"`` (clean two-view target) or ``"reference"``.
        beta: Smooth-L1 transition point.
        consensus_distance: Maximum clean-view disagreement for an averaged target.
    """

    weight: float = 0.0
    kind: str = "smooth_l1"
    target: str = "pseudo"
    beta: float = 0.05
    consensus_distance: float = 0.15

    def __post_init__(self) -> None:
        if self.kind not in PENALTY_KINDS or self.target not in RECOVERY_TARGETS:
            raise ValueError("invalid recovery kind or target")
        if self.weight < 0 or self.beta <= 0 or self.consensus_distance < 0:
            raise ValueError("recovery weight/consensus_distance must be non-negative and beta positive")


@dataclass(frozen=True)
class PositionPriorConfig:
    """Version-1 position-level priors (``loss.position_priors``), weights 0 by default.

    Attributes:
        periodicity_weight: Squared distance between samples one cycle apart.
        bone_symmetry_weight: Mirrored bone-length difference.
        half_symmetry_weight: Time-reversal symmetry about the cycle middle.
    """

    periodicity_weight: float = 0.0
    bone_symmetry_weight: float = 0.0
    half_symmetry_weight: float = 0.0

    def __post_init__(self) -> None:
        if min(self.periodicity_weight, self.bone_symmetry_weight, self.half_symmetry_weight) < 0:
            raise ValueError("position prior weights must be non-negative")


_LEGACY_FLAT_KEYS = {
    "recovery_weight": ("recovery", "weight"),
    "recovery_kind": ("recovery", "kind"),
    "recovery_target": ("recovery", "target"),
    "recovery_beta": ("recovery", "beta"),
    "consensus_distance": ("recovery", "consensus_distance"),
    "periodicity_weight": ("position_priors", "periodicity_weight"),
    "symmetry_weight": ("position_priors", "bone_symmetry_weight"),
    "half_symmetry_weight": ("position_priors", "half_symmetry_weight"),
    "residual_weight": ("residual", "weight"),
    "residual_norm": ("residual", "norm"),
    "cycle_weight": ("cycle", "weight"),
    "cycle_kind": ("cycle", "kind"),
    "cycle_beta": ("cycle", "beta"),
    "reliability_weight": ("reliability", "weight"),
    "feature_periodicity_weight": ("periodicity", "weight"),
    "feature_symmetry_weight": ("symmetry", "weight"),
}


@dataclass(frozen=True)
class LossConfig:
    """All objective settings (``src/configs/fusion/loss/*.yaml``).

    Attributes:
        cycle: :class:`CycleLossConfig`.
        reliability: :class:`ReliabilityLossConfig`.
        periodicity: :class:`PeriodicityLossConfig`.
        symmetry: :class:`SymmetryLossConfig`.
        residual: :class:`ResidualLossConfig`.
        recovery: :class:`RecoveryLossConfig` (v1, weight 0 by default).
        position_priors: :class:`PositionPriorConfig` (v1, weights 0 by default).
    """

    cycle: CycleLossConfig = field(default_factory=CycleLossConfig)
    reliability: ReliabilityLossConfig = field(default_factory=ReliabilityLossConfig)
    periodicity: PeriodicityLossConfig = field(default_factory=PeriodicityLossConfig)
    symmetry: SymmetryLossConfig = field(default_factory=SymmetryLossConfig)
    residual: ResidualLossConfig = field(default_factory=ResidualLossConfig)
    recovery: RecoveryLossConfig = field(default_factory=RecoveryLossConfig)
    position_priors: PositionPriorConfig = field(default_factory=PositionPriorConfig)

    @property
    def weights(self) -> dict[str, float]:
        """Coefficient of every term, keyed by the term names of :class:`LossBreakdown`."""
        return {
            "cycle": self.cycle.weight,
            "reliability": self.reliability.weight,
            "periodicity": self.periodicity.weight,
            "symmetry": self.symmetry.weight,
            "residual": self.residual.weight,
            "recovery": self.recovery.weight,
            "position_periodicity": self.position_priors.periodicity_weight,
            "bone_symmetry": self.position_priors.bone_symmetry_weight,
            "half_symmetry": self.position_priors.half_symmetry_weight,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "LossConfig" | None) -> "LossConfig":
        """Build from a nested mapping (Hydra) or the legacy flat key layout; ``None`` = defaults."""
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
        payload: dict[str, Any] = {}
        for key, item in dict(value).items():
            if key in _LEGACY_FLAT_KEYS:
                section, name = _LEGACY_FLAT_KEYS[key]
                payload.setdefault(section, {})[name] = item
            else:
                payload[key] = item
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(payload) - set(known))
        if unknown:
            raise ValueError(f"unknown loss config fields: {unknown}")
        kwargs: dict[str, Any] = {}
        for name, item in payload.items():
            section_type = known[name].default_factory  # type: ignore[union-attr]
            if isinstance(item, Mapping):
                kwargs[name] = _section(section_type, item)  # type: ignore[arg-type]
            else:
                kwargs[name] = item
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dictionary."""
        return asdict(self)


def _section(section_type: Any, item: Mapping[str, Any]) -> Any:
    """Instantiate a (possibly nested) config section from a mapping."""
    instance_defaults = section_type()
    kwargs: dict[str, Any] = {}
    for key, val in item.items():
        current = getattr(instance_defaults, key, None)
        if is_dataclass(current) and isinstance(val, Mapping):
            kwargs[key] = _section(type(current), val)
        else:
            kwargs[key] = val
    return section_type(**kwargs)


# --------------------------------------------------------------------------- breakdown
@dataclass(frozen=True)
class LossBreakdown:
    """Raw value of every term, the weights, and the weighted total.

    Attributes:
        raw: Unweighted value per term (``cycle``, ``reliability``,
            ``periodicity``, ``symmetry``, ``residual``, ``recovery``,
            ``position_periodicity``, ``bone_symmetry``, ``half_symmetry``).
        weights: Coefficient per term.
        total: ``sum_k weights[k] * raw[k]``.
    """

    raw: Mapping[str, Tensor]
    weights: Mapping[str, float]
    total: Tensor

    def weighted(self, name: str) -> Tensor:
        """Weighted contribution of one term."""
        return self.raw[name] * self.weights[name]

    # Convenience accessors used throughout the code base.
    @property
    def cycle(self) -> Tensor:
        return self.raw["cycle"]

    @property
    def reliability(self) -> Tensor:
        return self.raw["reliability"]

    @property
    def periodicity(self) -> Tensor:
        return self.raw["periodicity"]

    @property
    def symmetry(self) -> Tensor:
        return self.raw["symmetry"]

    @property
    def residual(self) -> Tensor:
        return self.raw["residual"]

    @property
    def recovery(self) -> Tensor:
        return self.raw["recovery"]

    def as_dict(self) -> dict[str, Tensor]:
        """``<term>_raw`` / ``<term>_weighted`` for every term plus ``total``."""
        out: dict[str, Tensor] = {}
        for name, value in self.raw.items():
            out[f"{name}_raw"] = value
            out[f"{name}_weighted"] = value * self.weights[name]
        out["total"] = self.total
        return out


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
    depth_a: Tensor | None = None,
    depth_b: Tensor | None = None,
    depth_alpha: float = 0.0,
    min_precision: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Build the label-free recovery target from the clean views.

    The target is the model's own closed-form base rule applied to the clean
    views with equal reliability: the plain average for architecture v1.0
    (``depth_alpha = 0``) and the depth-aware precision fusion for v1.1.
    Using the same rule as ``P_base`` matters: a target built with the plain
    average would carry the depth bias that the v1.1 base removes, and the
    residual would learn to put it back.

    Args:
        clean_a: ``[B, T, J, 3]`` clean View A.
        clean_b: ``[B, T, J, 3]`` clean View B.
        valid_a: ``[B, T, J]`` validity of View A.
        valid_b: ``[B, T, J]`` validity of View B.
        consensus_distance: Maximum disagreement for fusing both views.
        depth_a: Optional ``[B, T, 3]`` camera-depth axis of View A.
        depth_b: Optional ``[B, T, 3]`` camera-depth axis of View B.
        depth_alpha: ``fusion.depth_alpha`` of the model (0 = average).
        min_precision: ``fusion.min_precision`` of the model.

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
    half = torch.full_like(clean_a[..., :1], 0.5)
    fused, _ = depth_aware_pose_fusion(clean_a, clean_b, half, half, consensus, consensus, depth_a, depth_b, alpha=float(depth_alpha), min_precision=float(min_precision))
    target = torch.where(consensus[..., None], fused, torch.zeros_like(clean_a))
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


def natural_variation_dead_zone(dispersion: Tensor, config: DeadZoneConfig) -> Tensor:
    """Per-(sample, joint) dead zone ``delta = clip(scale * MAD, minimum, maximum)``.

    Where the dispersion is undefined (``inf``, e.g. padding or too few
    candidates) the dead zone is ``minimum`` (the confidence is zero there
    anyway).

    Args:
        dispersion: ``[B, T, J]`` MAD of the other cycles (canonical units).
        config: :class:`DeadZoneConfig`.

    Returns:
        ``[B, T, J]`` dead-zone radius; zeros when disabled.
    """
    if not config.enabled:
        return torch.zeros_like(dispersion)
    finite = torch.isfinite(dispersion)
    delta = torch.where(finite, config.scale * dispersion, torch.zeros_like(dispersion))
    delta = delta.clamp_min(config.minimum)
    if config.maximum is not None:
        delta = delta.clamp_max(config.maximum)
    return delta


def cycle_loss(
    prediction: Tensor,
    valid: Tensor,
    target: Tensor,
    confidence: Tensor,
    dead_zone: Tensor,
    *,
    kind: str,
    beta: float,
) -> Tensor:
    """Confidence-weighted robust penalty on the deviation beyond the dead zone.

        d       = |P_hat - P_ref|_2
        d_eff   = max(0, d - delta)
        L_cycle = sum C * rho(d_eff) / sum C

    Args:
        prediction: ``[B, T, J, 3]`` fused pose.
        valid: ``[B, T, J]`` validity of the fused pose.
        target: ``[B, T, J, 3]`` target pose.
        confidence: ``[B, T, J]`` confidence in ``[0, 1]`` (0 = no target).
        dead_zone: ``[B, T, J]`` dead-zone radius ``delta`` (0 = none).
        kind: Penalty kind applied to ``d_eff`` (scalar distance).
        beta: Smooth-L1 transition point.

    Returns:
        Scalar loss (zero when no confident target exists).
    """
    weight = torch.where(valid.bool(), confidence, torch.zeros_like(confidence))
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    effective = (distance - dead_zone).clamp_min(0.0)
    penalty = _penalty(effective[..., None], kind, beta)
    usable = (weight > 0) & torch.isfinite(penalty)
    weight = torch.where(usable, weight, torch.zeros_like(weight))
    return (weight * torch.where(usable, penalty, torch.zeros_like(penalty))).sum() / weight.sum().clamp_min(1e-6)


def phase_pair_masks(
    phase: Tensor,
    phase_valid: Tensor,
    cycle_index: Tensor,
    negative_phase_margin: float,
) -> tuple[Tensor, Tensor]:
    """Positive / negative pair masks between samples of consecutive cycles.

    For anchor ``t`` and candidate ``t'`` in the *next* cycle:

        positive: same phase (circular distance < 1 / (2 S) is not needed —
                  phases are multiples of 1/S, so equality up to 1e-4)
        negative: circular phase distance > ``negative_phase_margin``

    Args:
        phase: ``[B, T]`` phase in ``[0, 1)``.
        phase_valid: ``[B, T]`` bool.
        cycle_index: ``[B, T]`` cycle index (``-1`` outside cycles).
        negative_phase_margin: Margin in cycles (``0..0.5``).

    Returns:
        Tuple ``(positive, negative)`` of ``[B, T, T]`` bool masks indexed
        ``[b, anchor, candidate]``.
    """
    next_cycle = (cycle_index[:, None, :] == cycle_index[:, :, None] + 1) & (cycle_index[:, :, None] >= 0)
    both_valid = phase_valid[:, :, None] & phase_valid[:, None, :]
    difference = (phase[:, None, :] - phase[:, :, None]).abs()
    circular = torch.minimum(difference, 1.0 - difference)
    positive = next_cycle & both_valid & (circular < 1e-4)
    negative = next_cycle & both_valid & (circular > negative_phase_margin)
    return positive, negative


def contrastive_periodicity_loss(
    feature: Tensor,
    valid: Tensor,
    phase: Tensor,
    phase_valid: Tensor,
    cycle_index: Tensor,
    *,
    temperature: float,
    negative_phase_margin: float,
) -> Tensor:
    """InfoNCE periodicity: same phase of the next cycle against far phases.

        s(t, t')        = cos(F(t, j), F(t', j))
        L_period(t, j)  = -log  exp(s(t, t+) / tau)
                                / [ exp(s(t, t+) / tau) + sum_{t- in N(t)} exp(s(t, t-) / tau) ]

    with ``t+`` the sample of the next cycle at the same phase and ``N(t)`` the
    samples of the next cycle whose circular phase distance exceeds the
    margin.  A constant feature makes every similarity equal and the loss
    ``log(1 + |N|)``, so the trivial solution of the cosine version is no
    longer optimal.  Anchors without a positive or without negatives are
    skipped.

    Args:
        feature: ``[B, T, J, D]`` motion feature.
        valid: ``[B, T, J]`` validity of the feature.
        phase: ``[B, T]`` phase in ``[0, 1)``.
        phase_valid: ``[B, T]`` bool.
        cycle_index: ``[B, T]`` cycle index.
        temperature: InfoNCE temperature ``tau``.
        negative_phase_margin: Margin in cycles.

    Returns:
        Scalar loss (zero when no anchor has a positive and a negative).
    """
    batch, frames, joints, _ = feature.shape
    unit = torch.nn.functional.normalize(feature, dim=-1)
    # Cosine similarity between every pair of samples, per joint: [B, J, T, T].
    similarity = torch.einsum("btjd,bsjd->bjts", unit, unit) / temperature
    positive, negative = phase_pair_masks(phase, phase_valid, cycle_index, negative_phase_margin)
    joint_valid = valid.bool().permute(0, 2, 1)  # [B, J, T]
    pair_valid = joint_valid[:, :, :, None] & joint_valid[:, :, None, :]  # [B, J, T, T]
    positive = positive[:, None] & pair_valid
    negative = negative[:, None] & pair_valid
    has_positive = positive.any(dim=-1)
    has_negative = negative.any(dim=-1)
    anchors = has_positive & has_negative
    if not bool(anchors.any()):
        return feature.new_zeros(())
    candidates = positive | negative
    logits = torch.where(candidates, similarity, torch.full_like(similarity, -1e4))
    log_denominator = torch.logsumexp(logits, dim=-1)
    # Exactly one positive per anchor (same phase in the next cycle).
    positive_logit = torch.where(positive, similarity, torch.zeros_like(similarity)).sum(dim=-1)
    loss = log_denominator - positive_logit
    return masked_mean(loss, anchors)



def _external_reference_override(
    batch: Mapping[str, Any],
    output: PoseFusionOutput,
    fused_valid: Tensor,
    target: Tensor,
    confidence: Tensor,
    dead_zone: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Give an attached external reference priority over the cross-cycle target."""
    reference_valid = batch.get("reference_valid")
    if reference_valid is None or not bool(reference_valid.any()):
        return target, confidence, dead_zone
    frame_mask = batch["frame_mask"][..., None]
    ref_target, ref_valid = reference_target(batch["reference"], reference_valid & frame_mask, output.base_pose, fused_valid)
    target = torch.where(ref_valid[..., None], ref_target, target)
    confidence = torch.where(ref_valid, torch.ones_like(confidence), confidence)
    dead_zone = torch.where(ref_valid, torch.zeros_like(dead_zone), dead_zone)
    return target, confidence, dead_zone


def compute_losses(
    output: PoseFusionOutput,
    batch: Mapping[str, Any],
    *,
    skeleton: CommonSkeleton,
    config: LossConfig,
    samples_per_cycle: int,
    depth_alpha: float = 0.0,
    min_precision: float = 0.5,
) -> LossBreakdown:
    """Evaluate every objective on one batch.

    Args:
        output: Model output computed from ``batch["pose_a"]`` and
            ``batch["pose_b"]`` (possibly corrupted).
        batch: :class:`~fusion.sample.FusionBatch`.  ``clean_*`` fields define
            the v1 recovery target when present; ``cycle_target`` /
            ``cycle_confidence`` / ``cycle_dispersion`` define ``L_cycle``;
            ``corruption_mask_*`` define ``L_rel``; ``reference`` overrides
            the cross-cycle target where valid.
        skeleton: Common skeleton (bones, left/right pairs).
        config: :class:`LossConfig`.
        samples_per_cycle: ``S``.
        depth_alpha: The model's ``fusion.depth_alpha`` so the recovery
            target uses the same base rule as ``P_base`` (0 = average).
        min_precision: The model's ``fusion.min_precision``.

    Returns:
        :class:`LossBreakdown` with finite scalar tensors.
    """
    device = output.pose.device
    zero = output.pose.new_zeros(())
    frame_mask = batch.get("frame_mask")
    if frame_mask is None:
        frame_mask = torch.ones(output.pose.shape[:2], dtype=torch.bool, device=device)
    fused_valid = output.valid & frame_mask[..., None]
    cycle_index = batch.get("cycle_index")
    if cycle_index is not None:
        cycle_index = cycle_index.to(device)
    raw: dict[str, Tensor] = {}

    # ---- L_cycle (v2 main supervision)
    cycle_target = batch.get("cycle_target")
    if config.cycle.weight > 0 and cycle_target is not None:
        confidence = batch["cycle_confidence"].to(device)
        dispersion = batch.get("cycle_dispersion")
        dispersion = dispersion.to(device) if dispersion is not None else torch.full_like(confidence, float("inf"))
        dead_zone = natural_variation_dead_zone(dispersion, config.cycle.dead_zone)
        target = cycle_target.to(device)
        if config.cycle.external_reference:
            target, confidence, dead_zone = _external_reference_override(batch, output, fused_valid, target, confidence, dead_zone)
        raw["cycle"] = cycle_loss(output.pose, fused_valid, target, confidence, dead_zone, kind=config.cycle.kind, beta=config.cycle.beta)
    else:
        raw["cycle"] = zero

    # ---- L_rel
    if config.reliability.weight > 0 and "corruption_mask_a" in batch:
        raw["reliability"] = reliability_loss(output.reliability_logits, batch["corruption_mask_a"], batch["corruption_mask_b"], batch["valid_a"] & frame_mask[..., None], batch["valid_b"] & frame_mask[..., None])
    else:
        raw["reliability"] = zero

    # ---- L_period / L_sym on the motion features of both views
    raw["periodicity"] = zero
    raw["symmetry"] = zero
    if cycle_index is not None:
        views = ((output.motion_feature_a, batch["valid_a"] & frame_mask[..., None]), (output.motion_feature_b, batch["valid_b"] & frame_mask[..., None]))
        if config.periodicity.weight > 0:
            if config.periodicity.type == "cosine":
                raw["periodicity"] = sum(feature_periodicity_loss(f, v, cycle_index, samples_per_cycle) for f, v in views) / len(views)
            else:
                phase, phase_valid = batch["phase"].to(device), batch["phase_valid"].to(device)
                raw["periodicity"] = sum(contrastive_periodicity_loss(f, v, phase, phase_valid, cycle_index, temperature=config.periodicity.temperature, negative_phase_margin=config.periodicity.negative_phase_margin) for f, v in views) / len(views)
        half_index = batch.get("half_index")
        if config.symmetry.weight > 0 and half_index is not None:
            raw["symmetry"] = sum(feature_symmetry_loss(f, v, cycle_index, half_index.to(device), samples_per_cycle, skeleton.left_right_pairs) for f, v in views) / len(views)

    # ---- L_res
    if config.residual.norm == "l1":
        raw["residual"] = masked_mean(output.delta_pose.abs().sum(dim=-1), fused_valid)
    else:
        raw["residual"] = residual_loss(output.delta_pose, fused_valid)

    # ---- version-1 terms (weights 0 in v2, still computed for the logs when enabled)
    if config.recovery.weight > 0:
        clean_a = batch.get("clean_a", batch["pose_a"])
        clean_b = batch.get("clean_b", batch["pose_b"])
        clean_valid_a = batch.get("clean_valid_a", batch["valid_a"])
        clean_valid_b = batch.get("clean_valid_b", batch["valid_b"])
        depth_a, depth_b = batch.get("depth_a"), batch.get("depth_b")
        target, target_valid = pseudo_target(
            clean_a,
            clean_b,
            clean_valid_a,
            clean_valid_b,
            consensus_distance=config.recovery.consensus_distance,
            depth_a=None if depth_a is None else depth_a.to(device),
            depth_b=None if depth_b is None else depth_b.to(device),
            depth_alpha=depth_alpha,
            min_precision=min_precision,
        )
        target_valid = target_valid & frame_mask[..., None]
        if config.recovery.target == "reference":
            reference_valid = batch.get("reference_valid")
            if reference_valid is not None and bool(reference_valid.any()):
                ref_target, ref_valid = reference_target(batch["reference"], reference_valid & frame_mask[..., None], output.base_pose, fused_valid)
                target = torch.where(ref_valid[..., None], ref_target, target)
                target_valid = target_valid | ref_valid
        raw["recovery"] = recovery_loss(output.pose, fused_valid, target, target_valid, kind=config.recovery.kind, beta=config.recovery.beta)
    else:
        raw["recovery"] = zero
    priors = config.position_priors
    raw["position_periodicity"] = periodicity_loss(output.pose, fused_valid, cycle_index, samples_per_cycle) if (priors.periodicity_weight > 0 and cycle_index is not None) else zero
    raw["bone_symmetry"] = symmetry_loss(output.pose, fused_valid, skeleton) if priors.bone_symmetry_weight > 0 else zero
    half_index = batch.get("half_index")
    if priors.half_symmetry_weight > 0 and cycle_index is not None and half_index is not None:
        raw["half_symmetry"] = half_symmetry_loss(output.pose, fused_valid, batch["phase"].to(device), batch["phase_valid"].to(device), cycle_index, half_index.to(device), samples_per_cycle)
    else:
        raw["half_symmetry"] = zero

    weights = config.weights
    total = sum(weights[name] * value for name, value in raw.items())
    return LossBreakdown(raw=raw, weights=weights, total=total)
