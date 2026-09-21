"""Leave-one-cycle-out cross-cycle pose target (``L_cycle`` supervision).

Research Motivation:
    A target built from the current window's own two views (their average)
    is a function of the inputs, so on clean inputs the model has nothing to
    learn beyond reproducing that function (observed: learned == arithmetic
    average).  Repeated-cycle motion offers a target that is *independent of
    the current cycle*: the same movement, at the same normalised phase, in
    the person's other cycles.  On the private data the pose at the same
    phase of the neighbouring cycle is about three times closer than the
    other view of the same frame (0.048 vs 0.144 torso units), so it is both
    an external and a tight target.

Two-level aggregation (per sample, after phase normalisation to ``S``
samples per cycle):

    Stage 1 - within-cycle view consensus (:func:`view_consensus`)
        For every cycle ``c`` the two views are fused into one observation
        per (phase, joint):

            both views valid and |P_A - P_B| <= disagreement_threshold
                -> consensus: the plain mean (``method: mean``, architecture
                   v1.0) or the model's depth-aware base rule with equal
                   weights (``method: depth_aware``, v1.1).  The consensus
                   must use the same rule as ``P_base``: a mean target would
                   carry the depth bias the v1.1 base removes and pull the
                   prediction back toward the average.
            exactly one view valid -> that view
            otherwise              -> no reliable consensus (invalid)

        View observations are therefore never mixed with cycle observations
        as if they were independent samples.

    Stage 2 - cross-cycle consensus (:func:`cross_cycle_target`)
        For cycle ``i`` the candidates are the stage-1 consensus poses of
        the other cycles ``c in {i-n, ..., i-1, i+1, ..., i+n}``
        (``n = neighbors``; ``None`` = every other cycle).  The current cycle
        is never a candidate (leave-one-cycle-out):

            P_ref^i(k, j) = RobustMedian_{c != i} P_consensus^c(k, j)

        together with the robust dispersion of the candidates around it,

            MAD(k, j) = median_c |P_consensus^c(k, j) - P_ref^i(k, j)|_2 ,

        which the loss uses both for the confidence and for the natural
        variation dead zone (:mod:`fusion.losses`).

Confidence (:class:`CycleConfidence`), factorised so gates can be added
without changing the loss interface:

    C = C_repeatability * C_compatibility * C_validity

    C_repeatability = 1 / (1 + (MAD / tau)^2)         other cycles agree with each other
    C_validity      = [candidate count >= min_candidates]
    C_compatibility = 1 (hook; e.g. agreement of the current window's own
                      views with the reference, not enabled in this round)

All per-(phase, joint) statistics are returned in :class:`CrossCycleTarget`
so they can be logged and inspected.

Notes:
    * Targets use the clean inputs of the other cycles; training-time
      corruption of the current window never touches them.
    * Views are canonicalised independently into their own pelvis frame, so
      the consensus poses are directly comparable with the prediction
      (which lives in the frame of view A).
    * Shipped with every window as ``cycle_target [T, J, 3]``,
      ``cycle_confidence [T, J]`` and ``cycle_dispersion [T, J]``.

Shapes: ``view_a``/``view_b`` ``[T, J, 3]``, masks ``[T, J]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

CONSENSUS_METHODS = ("mean", "depth_aware")
AGGREGATIONS = ("median", "trimmed_mean")


@dataclass(frozen=True)
class ViewConsensusConfig:
    """Stage-1 within-cycle view fusion (``data.cycle_target.consensus``).

    Attributes:
        disagreement_threshold: Maximum |P_A - P_B| (canonical units) for the
            two views to be fused; larger disagreements yield no consensus.
        method: Fusion rule for agreeing views: ``"mean"`` or
            ``"depth_aware"`` (the v1.1 base rule with equal weights, see
            :func:`fusion.modules.weighted_fusion.depth_aware_pose_fusion`).
        depth_alpha: ``fusion.depth_alpha`` of the model for ``depth_aware``.
        min_precision: ``fusion.min_precision`` of the model.
    """

    disagreement_threshold: float = 0.15
    method: str = "mean"
    depth_alpha: float = 0.8
    min_precision: float = 0.5

    def __post_init__(self) -> None:
        if self.disagreement_threshold < 0:
            raise ValueError("disagreement_threshold must be non-negative")
        if self.method not in CONSENSUS_METHODS:
            raise ValueError(f"method must be one of {CONSENSUS_METHODS}")
        if not 0.0 <= float(self.depth_alpha) < 1.0 or not 0.0 < float(self.min_precision) <= 1.0:
            raise ValueError("depth_alpha must be in [0, 1) and min_precision in (0, 1]")


@dataclass(frozen=True)
class ConfidenceConfig:
    """Factors of the cross-cycle confidence (``data.cycle_target.confidence``).

    Attributes:
        tau: Dispersion scale of the repeatability factor
            ``1 / (1 + (MAD / tau)^2)`` in canonical units.
        min_candidates: Validity factor: fewer valid candidates -> 0.
        compatibility: Enable the compatibility gate (not implemented in this
            round; must stay ``False``).
    """

    tau: float = 0.05
    min_candidates: int = 2
    compatibility: bool = False

    def __post_init__(self) -> None:
        if self.tau <= 0 or self.min_candidates < 1:
            raise ValueError("tau must be positive and min_candidates at least 1")
        if self.compatibility:
            raise ValueError("the compatibility gate is not implemented yet")


@dataclass(frozen=True)
class CycleTargetConfig:
    """Parameters of the cross-cycle target (``data.cycle_target``).

    Attributes:
        enabled: Compute the target (needed by ``loss.cycle.weight > 0``).
        neighbors: Cycles on each side used as candidates; ``None`` = all
            other cycles of the sequence.
        aggregation: Stage-2 rule: ``"median"`` or ``"trimmed_mean"`` (drops
            the farthest quarter of the candidates before averaging).
        consensus: Stage-1 settings.
        confidence: Confidence factors.
    """

    enabled: bool = True
    neighbors: int | None = 2
    aggregation: str = "median"
    consensus: ViewConsensusConfig = field(default_factory=ViewConsensusConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)

    def __post_init__(self) -> None:
        if self.neighbors is not None and self.neighbors < 1:
            raise ValueError("neighbors must be positive or None")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "CycleTargetConfig" | None) -> "CycleTargetConfig":
        """Build from a (nested) mapping; ``None`` gives defaults.

        Legacy flat keys ``tau`` / ``min_candidates`` are accepted and routed
        to the confidence section.
        """
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
        payload = dict(value)
        confidence = dict(payload.pop("confidence", {}) or {})
        for legacy in ("tau", "min_candidates"):
            if legacy in payload:
                confidence[legacy] = payload.pop(legacy)
        consensus = dict(payload.pop("consensus", {}) or {})
        return cls(consensus=ViewConsensusConfig(**consensus), confidence=ConfidenceConfig(**confidence), **payload)


@dataclass(frozen=True)
class CrossCycleTarget:
    """Target and per-(phase, joint) statistics of one sample.

    Attributes:
        target: ``[T, J, 3]`` cross-cycle consensus (zero where undefined).
        confidence: ``[T, J]`` product of the confidence factors, in ``[0, 1]``.
        dispersion: ``[T, J]`` MAD of the candidates (canonical units), ``inf``
            where fewer than two candidates exist.
        candidate_count: ``[T, J]`` number of valid stage-1 candidates.
        repeatability: ``[T, J]`` repeatability factor.
        validity: ``[T, J]`` validity factor.
        compatibility: ``[T, J]`` compatibility factor (ones in this round).
    """

    target: np.ndarray
    confidence: np.ndarray
    dispersion: np.ndarray
    candidate_count: np.ndarray
    repeatability: np.ndarray
    validity: np.ndarray
    compatibility: np.ndarray

    @classmethod
    def empty(cls, frames: int, joints: int) -> "CrossCycleTarget":
        """A target with no defined entries (single-cycle sequences, disabled)."""
        return cls(
            target=np.zeros((frames, joints, 3), dtype=np.float32),
            confidence=np.zeros((frames, joints), dtype=np.float32),
            dispersion=np.full((frames, joints), np.inf, dtype=np.float32),
            candidate_count=np.zeros((frames, joints), dtype=np.int32),
            repeatability=np.zeros((frames, joints), dtype=np.float32),
            validity=np.zeros((frames, joints), dtype=np.float32),
            compatibility=np.ones((frames, joints), dtype=np.float32),
        )


def view_consensus(
    view_a: np.ndarray,
    view_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    config: ViewConsensusConfig,
    depth_a: np.ndarray | None = None,
    depth_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 1: fuse the two views of one cycle into one observation per joint.

    Args:
        view_a: ``[T, J, 3]`` canonical View A.
        view_b: ``[T, J, 3]`` canonical View B.
        valid_a: ``[T, J]`` validity of View A.
        valid_b: ``[T, J]`` validity of View B.
        config: Disagreement threshold and fusion rule.
        depth_a: ``[T, 3]`` camera-depth axis of View A (``depth_aware`` rule).
        depth_b: ``[T, 3]`` camera-depth axis of View B.

    Returns:
        Tuple ``(pose, valid, disagreement)``: the consensus pose
        ``[T, J, 3]`` (zero where invalid), its validity ``[T, J]`` and the
        view disagreement ``|P_A - P_B|`` ``[T, J]`` (``nan`` unless both views
        are valid).
    """
    valid_a, valid_b = np.asarray(valid_a, dtype=bool), np.asarray(valid_b, dtype=bool)
    both = valid_a & valid_b
    disagreement = np.where(both, np.linalg.norm(view_a - view_b, axis=-1), np.nan)
    agree = both & (disagreement <= config.disagreement_threshold)
    only_a = valid_a & ~valid_b
    only_b = valid_b & ~valid_a
    if config.method == "depth_aware" and depth_a is not None and depth_b is not None:
        fused = _depth_aware_consensus(view_a, view_b, agree, depth_a, depth_b, config)
    else:
        fused = np.where(agree[..., None], 0.5 * (view_a + view_b), 0.0)  # "mean" rule
    fused = np.where(only_a[..., None], view_a, fused)
    fused = np.where(only_b[..., None], view_b, fused)
    valid = agree | only_a | only_b
    return fused.astype(np.float32), valid, disagreement.astype(np.float32)


def _depth_aware_consensus(view_a: np.ndarray, view_b: np.ndarray, agree: np.ndarray, depth_a: np.ndarray, depth_b: np.ndarray, config: ViewConsensusConfig) -> np.ndarray:
    """Equal-weight v1.1 base rule on agreeing joints (zero elsewhere), via the torch implementation."""
    import torch

    from .modules.weighted_fusion import depth_aware_pose_fusion

    a = torch.from_numpy(np.ascontiguousarray(view_a, dtype=np.float32))[None]
    b = torch.from_numpy(np.ascontiguousarray(view_b, dtype=np.float32))[None]
    mask = torch.from_numpy(np.ascontiguousarray(agree, dtype=bool))[None]
    half = torch.full_like(a[..., :1], 0.5)
    fused, _ = depth_aware_pose_fusion(
        a,
        b,
        half,
        half,
        mask,
        mask,
        torch.from_numpy(np.ascontiguousarray(depth_a, dtype=np.float32))[None],
        torch.from_numpy(np.ascontiguousarray(depth_b, dtype=np.float32))[None],
        alpha=float(config.depth_alpha),
        min_precision=float(config.min_precision),
    )
    return fused[0].numpy().astype(np.float32)


def _aggregate(candidates: np.ndarray, valid: np.ndarray, aggregation: str) -> tuple[np.ndarray, np.ndarray]:
    """Robust centre of ``[N, S, J, 3]`` candidates and its ``[S, J]`` MAD.

    Invalid candidates are ignored per (phase, joint).  The MAD is ``inf``
    where fewer than two candidates are valid.
    """
    masked = np.where(valid[..., None], candidates, np.nan)
    with np.errstate(all="ignore"):
        if aggregation == "median":
            centre = np.nanmedian(masked, axis=0)
        else:
            distance = np.linalg.norm(masked - np.nanmedian(masked, axis=0, keepdims=True), axis=-1)  # [N, S, J]
            keep = int(np.ceil(0.75 * candidates.shape[0]))
            order = np.argsort(np.where(np.isfinite(distance), distance, np.inf), axis=0)
            rank = np.argsort(order, axis=0)
            trimmed = np.where(((rank < keep) & valid)[..., None], masked, np.nan)
            centre = np.nanmean(trimmed, axis=0)
        deviation = np.linalg.norm(masked - centre[None], axis=-1)  # [N, S, J]
        mad = np.nanmedian(deviation, axis=0)
    count = valid.sum(axis=0)
    # np.nan_to_num would turn inf into the largest finite float; keep inf explicit.
    mad = np.where((count >= 2) & np.isfinite(mad), mad, np.inf)
    return np.nan_to_num(centre, nan=0.0).astype(np.float32), mad.astype(np.float32)


class CycleConfidence:
    """Factorised confidence ``C = C_repeatability * C_compatibility * C_validity``.

    Args:
        config: :class:`ConfidenceConfig`.
        compatibility_gate: Optional callable ``(count, mad) -> [S, J]``
            factor in ``[0, 1]``; the hook for a future compatibility gate.
    """

    def __init__(self, config: ConfidenceConfig, compatibility_gate: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> None:
        self.config = config
        self.compatibility_gate = compatibility_gate

    def repeatability(self, mad: np.ndarray) -> np.ndarray:
        """``1 / (1 + (MAD / tau)^2)``; zero where the MAD is undefined."""
        with np.errstate(all="ignore"):
            factor = 1.0 / (1.0 + (mad / self.config.tau) ** 2)
        return np.where(np.isfinite(mad), factor, 0.0).astype(np.float32)

    def validity(self, count: np.ndarray) -> np.ndarray:
        """``1`` where at least ``min_candidates`` candidates exist."""
        return (count >= self.config.min_candidates).astype(np.float32)

    def compatibility(self, count: np.ndarray, mad: np.ndarray) -> np.ndarray:
        """Compatibility factor (ones unless a gate is supplied)."""
        if self.compatibility_gate is None:
            return np.ones(mad.shape, dtype=np.float32)
        return np.clip(self.compatibility_gate(count, mad), 0.0, 1.0).astype(np.float32)

    def __call__(self, count: np.ndarray, mad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(confidence, repeatability, validity, compatibility)``."""
        rep, val, comp = self.repeatability(mad), self.validity(count), self.compatibility(count, mad)
        return (rep * val * comp).astype(np.float32), rep, val, comp


def cross_cycle_target(
    view_a: np.ndarray,
    view_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    cycle_bounds: Sequence[tuple[int, int]],
    *,
    samples_per_cycle: int,
    config: CycleTargetConfig,
    compatibility_gate: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    depth_a: np.ndarray | None = None,
    depth_b: np.ndarray | None = None,
) -> CrossCycleTarget:
    """Leave-one-cycle-out target and statistics for a phase-normalised sample.

    Args:
        view_a: ``[T, J, 3]`` canonical View A (phase normalised).
        view_b: ``[T, J, 3]`` canonical View B.
        valid_a: ``[T, J]`` validity of View A.
        valid_b: ``[T, J]`` validity of View B.
        cycle_bounds: Cycles as ``(start, end)``; every cycle must span
            exactly ``samples_per_cycle`` samples.
        samples_per_cycle: ``S``.
        config: :class:`CycleTargetConfig`.
        compatibility_gate: Optional compatibility hook (see :class:`CycleConfidence`).
        depth_a: Optional ``[T, 3]`` camera-depth axis of View A (used by the
            ``depth_aware`` consensus).
        depth_b: Optional ``[T, 3]`` camera-depth axis of View B.

    Returns:
        :class:`CrossCycleTarget` (all fields zero / ``inf`` where undefined).

    Raises:
        ValueError: If a cycle does not span ``samples_per_cycle`` samples.
    """
    frames, joints = valid_a.shape
    bounds = [(int(s), int(e)) for s, e in cycle_bounds]
    for start, end in bounds:
        if end - start != samples_per_cycle:
            raise ValueError("cross-cycle targets require phase-normalised cycles of samples_per_cycle samples")
    result = CrossCycleTarget.empty(frames, joints)
    if len(bounds) < 2 or not config.enabled:
        return result
    # Stage 1: one consensus observation per cycle.
    consensus = [
        view_consensus(
            view_a[s:e],
            view_b[s:e],
            valid_a[s:e],
            valid_b[s:e],
            config.consensus,
            None if depth_a is None else depth_a[s:e],
            None if depth_b is None else depth_b[s:e],
        )
        for s, e in bounds
    ]
    confidence = CycleConfidence(config.confidence, compatibility_gate)
    target, conf = result.target.copy(), result.confidence.copy()
    dispersion, count = result.dispersion.copy(), result.candidate_count.copy()
    rep, val, comp = result.repeatability.copy(), result.validity.copy(), result.compatibility.copy()
    # Stage 2: leave-one-cycle-out robust centre over the other cycles.
    for i, (start, end) in enumerate(bounds):
        others = [c for c in range(len(bounds)) if c != i and (config.neighbors is None or abs(c - i) <= config.neighbors)]
        if not others:
            continue
        candidates = np.stack([consensus[c][0] for c in others])  # [N, S, J, 3]
        valid = np.stack([consensus[c][1] for c in others])       # [N, S, J]
        centre, mad = _aggregate(candidates, valid, config.aggregation)
        n_valid = valid.sum(axis=0)
        c_total, c_rep, c_val, c_comp = confidence(n_valid, mad)
        target[start:end] = np.where(c_total[..., None] > 0, centre, 0.0)
        conf[start:end], dispersion[start:end], count[start:end] = c_total, mad, n_valid
        rep[start:end], val[start:end], comp[start:end] = c_rep, c_val, c_comp
    return CrossCycleTarget(target=target, confidence=conf, dispersion=dispersion, candidate_count=count, repeatability=rep, validity=val, compatibility=comp)
