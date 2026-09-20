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

Definition (per sample, after phase normalisation to ``S`` samples per cycle):
    For cycle ``i``, phase index ``k`` and joint ``j``, the candidate set is
    the canonical poses of BOTH input views at the same phase in the other
    cycles ``c in {i-n, ..., i-1, i+1, ..., i+n}`` (``n = neighbors``; ``None``
    = every other cycle).  The current cycle is never a candidate
    (leave-one-cycle-out):

        P_ref^i(k, j) = median_{c != i, view in {A, B}} P_c^view(k, j)      (per coordinate)

    Confidence from the dispersion of the candidates around that median:

        MAD(k, j) = median_c |P_c(k, j) - P_ref(k, j)|_2
        C(k, j)   = 1 / (1 + (MAD / tau)^2)                                  in (0, 1]

    ``C = 0`` where fewer than ``min_candidates`` valid candidates exist (a
    sequence with a single cycle therefore has no cross-cycle target).

Notes:
    * The target uses the clean inputs of the other cycles; training-time
      corruption of the current window never touches it.
    * Views are canonicalised independently into their own pelvis frame, so
      the candidates of both views are directly comparable with the
      prediction (which lives in the frame of view A).
    * Computed by the DataModule and shipped with every window as
      ``cycle_target [T, J, 3]`` and ``cycle_confidence [T, J]``.

Shapes: ``view_a``/``view_b`` ``[T, J, 3]``, masks ``[T, J]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CycleTargetConfig:
    """Parameters of the cross-cycle target (``data.cycle_target``).

    Attributes:
        enabled: Compute the target (needed by ``loss.cycle_weight > 0``).
        neighbors: Cycles on each side used as candidates; ``None`` = all
            other cycles of the sequence.
        tau: Dispersion scale of the confidence in canonical units.
        min_candidates: Minimum number of valid candidates for a target.
        aggregation: ``"median"`` (default) or ``"trimmed_mean"`` (drops the
            farthest quarter of the candidates before averaging).
    """

    enabled: bool = True
    neighbors: int | None = 2
    tau: float = 0.05
    min_candidates: int = 4
    aggregation: str = "median"

    def __post_init__(self) -> None:
        if self.neighbors is not None and self.neighbors < 1:
            raise ValueError("neighbors must be positive or None")
        if self.tau <= 0 or self.min_candidates < 1:
            raise ValueError("tau must be positive and min_candidates at least 1")
        if self.aggregation not in ("median", "trimmed_mean"):
            raise ValueError("aggregation must be median or trimmed_mean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "CycleTargetConfig" | None) -> "CycleTargetConfig":
        """Build from a mapping; ``None`` gives defaults."""
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


def _aggregate(candidates: np.ndarray, valid: np.ndarray, aggregation: str) -> tuple[np.ndarray, np.ndarray]:
    """Robust centre of ``[N, S, J, 3]`` candidates and its ``[S, J]`` MAD.

    Invalid candidates are ignored per (phase, joint).
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
    return np.nan_to_num(centre, nan=0.0).astype(np.float32), np.nan_to_num(mad, nan=np.inf)


def cross_cycle_target(
    view_a: np.ndarray,
    view_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    cycle_bounds: Sequence[tuple[int, int]],
    *,
    samples_per_cycle: int,
    config: CycleTargetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-cycle-out target and confidence for a phase-normalised sample.

    Args:
        view_a: ``[T, J, 3]`` canonical View A (phase normalised).
        view_b: ``[T, J, 3]`` canonical View B.
        valid_a: ``[T, J]`` validity of View A.
        valid_b: ``[T, J]`` validity of View B.
        cycle_bounds: Cycles as ``(start, end)``; every cycle must span
            exactly ``samples_per_cycle`` samples.
        samples_per_cycle: ``S``.
        config: :class:`CycleTargetConfig`.

    Returns:
        Tuple ``(target, confidence)`` with shapes ``[T, J, 3]`` (zero where
        undefined) and ``[T, J]`` (zero where undefined).

    Raises:
        ValueError: If a cycle does not span ``samples_per_cycle`` samples.
    """
    frames, joints = valid_a.shape
    target = np.zeros((frames, joints, 3), dtype=np.float32)
    confidence = np.zeros((frames, joints), dtype=np.float32)
    bounds = [(int(s), int(e)) for s, e in cycle_bounds]
    for start, end in bounds:
        if end - start != samples_per_cycle:
            raise ValueError("cross-cycle targets require phase-normalised cycles of samples_per_cycle samples")
    if len(bounds) < 2 or not config.enabled:
        return target, confidence
    poses = [np.stack((view_a[s:e], view_b[s:e])) for s, e in bounds]       # each [2, S, J, 3]
    valids = [np.stack((valid_a[s:e], valid_b[s:e])) for s, e in bounds]    # each [2, S, J]
    for i, (start, end) in enumerate(bounds):
        others = [c for c in range(len(bounds)) if c != i and (config.neighbors is None or abs(c - i) <= config.neighbors)]
        if not others:
            continue
        candidates = np.concatenate([poses[c] for c in others])   # [N, S, J, 3]
        valid = np.concatenate([valids[c] for c in others])       # [N, S, J]
        centre, mad = _aggregate(candidates, valid, config.aggregation)
        count = valid.sum(axis=0)
        conf = 1.0 / (1.0 + (mad / config.tau) ** 2)
        conf = np.where(count >= config.min_candidates, conf, 0.0).astype(np.float32)
        target[start:end] = np.where(conf[..., None] > 0, centre, 0.0)
        confidence[start:end] = conf
    return target, confidence
