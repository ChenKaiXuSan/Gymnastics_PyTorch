"""Movement-cycle detection on the right-hand angle signal.

This module is the single place where "one cycle" and "the middle of a
cycle" are defined.  It is used by ``gymnastics align`` for the private
recordings and by ``gymnastics align cycles`` for the public datasets, so
every dataset shares one definition.  Training code never calls it; it reads
the records written by :mod:`gymnastics.alignment.cycle_records`.

Signal:
    In the pelvis-centred body frame (x: left hip -> right hip, y: pelvis ->
    shoulder centre, z: forward) the right wrist describes a loop around the
    body during a trunk rotation.  Its azimuth

        theta(t) = atan2(z_wrist, x_wrist)

    is smoothed with an odd moving-average window and unwrapped so that the
    curve is continuous over consecutive turns (:func:`hand_theta_unwrapped`).

Cycle start:
    ``theta_ref`` is a reference azimuth (by default the 10th percentile of
    the unwrapped signal, i.e. close to the trough).  A cycle starts each time
    the signal crosses ``theta_ref`` (modulo 2 pi) *upwards while rotating
    counter-clockwise* (positive angular velocity).  Consecutive crossings
    must be at least ``min_period_sec`` apart.  When fewer than two
    counter-clockwise crossings exist the clockwise direction is tried.
    Cycle ``k`` is the half-open interval ``[crossing_k, crossing_{k+1})``;
    the incomplete motion before the first and after the last crossing is
    discarded.

Cycle middle (turn-around point):
    Within a cycle the wrist rotates away from the start azimuth and then
    returns.  The middle is the frame where the rotation reverses: the
    maximum of the smoothed unwrapped signal for counter-clockwise cycles and
    the minimum for clockwise cycles (``mid_rule = "theta_extremum"``).  It
    splits the cycle into an outward half ``[start, mid)`` and a return half
    ``[mid, end)`` that are approximately time-reversed mirrors of each other,
    which is what the later symmetry analysis needs.  The middle is *not*
    assumed to be at phase 0.5.

Plausibility filter (public datasets):
    Cycles longer than ``max_period_sec`` are rejected, and a sequence whose
    signal amplitude (95th minus 5th percentile) is below
    ``min_amplitude_rad`` yields no cycles at all.  Private recordings are
    known to be periodic and use no filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

Direction = Literal["ccw", "cw"]

DEFAULT_THETA_REF = -3.0 * np.pi / 4.0
DEFAULT_SMOOTH_WINDOW = 11
DEFAULT_MIN_PERIOD_SEC = 0.8
RIGHT_WRIST_INDEX = 41
MID_RULE = "theta_extremum"


@dataclass(frozen=True)
class CycleSpan:
    """One detected cycle on a frame timeline.

    Attributes:
        start: First frame of the cycle (inclusive).
        mid: Turn-around frame, ``start < mid < end``.
        end: End frame (exclusive).
    """

    start: int
    mid: int
    end: int

    def __post_init__(self) -> None:
        if not self.start < self.mid < self.end:
            raise ValueError(f"cycle requires start < mid < end, got {self}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def shifted(self, offset: int) -> "CycleSpan":
        """Return the span moved by ``offset`` frames."""
        return CycleSpan(self.start + offset, self.mid + offset, self.end + offset)


@dataclass(frozen=True)
class DetectionSettings:
    """Parameters of one detection run, stored in every record for provenance.

    Attributes:
        signal: Name of the 1-D signal (``"right_wrist_theta_body"``).
        smooth_window: Moving-average window in frames (odd).
        theta_ref: Reference azimuth in radians used for the crossings
            (``None`` when unknown, e.g. middles added to legacy records).
        theta_ref_mode: ``"auto_p10"``, ``"manual"`` or ``"legacy_align"``.
        min_period_sec: Minimum spacing of consecutive cycle starts.
        max_period_sec: Maximum cycle duration (``None`` = unlimited).
        min_amplitude_rad: Minimum signal amplitude (``None`` = unlimited).
        direction: Rotation direction of the accepted crossings.
        mid_rule: How the middle frame is chosen.
    """

    signal: str = "right_wrist_theta_body"
    smooth_window: int = DEFAULT_SMOOTH_WINDOW
    theta_ref: Optional[float] = DEFAULT_THETA_REF
    theta_ref_mode: str = "auto_p10"
    min_period_sec: float = DEFAULT_MIN_PERIOD_SEC
    max_period_sec: Optional[float] = None
    min_amplitude_rad: Optional[float] = None
    direction: Direction = "ccw"
    mid_rule: str = MID_RULE

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        if payload["theta_ref"] is not None and not np.isfinite(payload["theta_ref"]):
            payload["theta_ref"] = None
        return payload


def smooth_1d(x: np.ndarray, win: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """Odd-window moving average with edge padding (same as ``gymnastics align``)."""
    win = max(3, int(win) | 1)
    pad = win // 2
    xp = np.pad(np.asarray(x, dtype=np.float32), (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")


def hand_theta_unwrapped(
    kpts_body: np.ndarray,
    *,
    wrist_index: int = RIGHT_WRIST_INDEX,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> np.ndarray:
    """Smoothed, unwrapped azimuth of the right wrist in the body frame.

    Args:
        kpts_body: ``[T, J, 3]`` body-frame keypoints (NaN allowed; filled by
            linear interpolation before smoothing).
        wrist_index: Joint index of the right wrist (MHR70: 41).
        smooth_window: Moving-average window in frames.

    Returns:
        ``[T]`` float32 unwrapped angle in radians.
    """
    hand = np.asarray(kpts_body, dtype=np.float32)[:, wrist_index, :]
    theta = np.arctan2(hand[:, 2], hand[:, 0]).astype(np.float64)
    missing = ~np.isfinite(theta)
    if missing.all():
        return np.zeros(len(theta), dtype=np.float32)
    if missing.any():
        good = np.flatnonzero(~missing)
        theta[missing] = np.interp(np.flatnonzero(missing), good, theta[good])
    theta = np.unwrap(theta)
    return smooth_1d(theta, smooth_window).astype(np.float32)


def auto_theta_ref(theta_unwrap: np.ndarray, percentile: float = 10.0) -> float:
    """Reference azimuth near the trough: the given percentile of the signal."""
    return float(np.percentile(np.asarray(theta_unwrap, dtype=np.float64), percentile))


def find_crossings(
    theta_unwrap: np.ndarray,
    fps: float,
    *,
    theta_ref: float = DEFAULT_THETA_REF,
    min_period_sec: float = DEFAULT_MIN_PERIOD_SEC,
    direction: Direction = "ccw",
) -> List[int]:
    """Frames where the signal crosses ``theta_ref`` (mod 2 pi) in ``direction``.

    Args:
        theta_unwrap: ``[T]`` unwrapped angle.
        fps: Frames per second (sets the minimum spacing in frames).
        theta_ref: Reference azimuth.
        min_period_sec: Minimum spacing between accepted crossings.
        direction: ``"ccw"`` accepts upward crossings with positive angular
            velocity; ``"cw"`` accepts them with negative velocity.

    Returns:
        Sorted list of crossing frames.
    """
    theta = np.asarray(theta_unwrap, dtype=np.float64)
    min_gap = int(round(min_period_sec * fps))
    k = np.round((theta - theta_ref) / (2 * np.pi))
    d = theta - (theta_ref + 2 * np.pi * k)
    sgn = np.sign(d)
    sgn[sgn == 0] = 1
    vel = np.gradient(theta) if len(theta) > 1 else np.zeros_like(theta)
    ok = vel > 0 if direction == "ccw" else vel < 0
    crossing = [t for t in range(1, len(theta)) if ok[t] and sgn[t - 1] < 0 and sgn[t] > 0]
    out: List[int] = []
    last = -(10**9)
    for t in crossing:
        if t - last >= min_gap:
            out.append(t)
            last = t
    return out


def mid_point(theta_unwrap: np.ndarray, start: int, end: int, direction: Direction) -> int:
    """Turn-around frame of one cycle: the extremum of the signal inside it.

    The extremum is searched strictly inside ``(start, end - 1)`` so that the
    middle never coincides with a boundary.
    """
    if end - start < 3:
        raise ValueError("a cycle needs at least three frames to have a middle")
    segment = np.asarray(theta_unwrap[start + 1 : end - 1], dtype=np.float64)
    offset = int(np.argmax(segment) if direction == "ccw" else np.argmin(segment))
    return start + 1 + offset


def spans_from_crossings(theta_unwrap: np.ndarray, crossings: Sequence[int], direction: Direction) -> List[CycleSpan]:
    """Turn consecutive crossings into :class:`CycleSpan` objects with middles."""
    spans: List[CycleSpan] = []
    for start, end in zip(crossings[:-1], crossings[1:]):
        if end - start < 3:
            continue
        spans.append(CycleSpan(int(start), mid_point(theta_unwrap, int(start), int(end), direction), int(end)))
    return spans


def detect_cycles(
    theta_unwrap: np.ndarray,
    fps: float,
    *,
    settings: DetectionSettings = DetectionSettings(),
    both_directions: bool = True,
) -> Tuple[List[CycleSpan], DetectionSettings]:
    """Detect cycles and their middles on an unwrapped angle signal.

    Args:
        theta_unwrap: ``[T]`` smoothed, unwrapped angle.
        fps: Frames per second.
        settings: Detection parameters; with ``theta_ref_mode == "auto_p10"``
            the reference is derived from the signal.
        both_directions: Try clockwise when counter-clockwise yields fewer
            than two crossings.

    Returns:
        Tuple ``(spans, used_settings)`` where ``used_settings`` records the
        reference and direction actually used.
    """
    theta = np.asarray(theta_unwrap, dtype=np.float64)
    used = settings
    if settings.theta_ref_mode == "auto_p10":
        used = DetectionSettings(**{**settings.to_dict(), "theta_ref": auto_theta_ref(theta)})
    if used.theta_ref is None:
        raise ValueError("detect_cycles requires theta_ref (set theta_ref_mode='auto_p10' or a manual value)")
    if used.min_amplitude_rad is not None and len(theta):
        amplitude = float(np.percentile(theta, 95) - np.percentile(theta, 5))
        if amplitude < used.min_amplitude_rad:
            return [], used
    theta_ref = float(used.theta_ref)
    crossings = find_crossings(theta, fps, theta_ref=theta_ref, min_period_sec=used.min_period_sec, direction="ccw")
    direction: Direction = "ccw"
    if both_directions and len(crossings) < 2:
        alternative = find_crossings(theta, fps, theta_ref=theta_ref, min_period_sec=used.min_period_sec, direction="cw")
        if len(alternative) >= 2:
            crossings, direction = alternative, "cw"
    used = DetectionSettings(**{**used.to_dict(), "direction": direction})
    spans = spans_from_crossings(theta, crossings, direction)
    if used.max_period_sec is not None:
        limit = int(round(used.max_period_sec * fps))
        spans = [span for span in spans if span.length <= limit]
    return spans, used


def annotate_mid_points(theta_unwrap: np.ndarray, cycles: Sequence[Tuple[int, int]]) -> List[CycleSpan]:
    """Add middles to existing ``(start, end)`` cycles without re-segmenting.

    The rotation direction of each cycle is inferred from the sign of the
    signal change over its first quarter, so records produced before the
    middle existed can be augmented in place.

    Args:
        theta_unwrap: ``[T]`` smoothed, unwrapped angle on the same timeline.
        cycles: Existing ``(start, end)`` frame ranges.

    Returns:
        One :class:`CycleSpan` per input cycle.
    """
    theta = np.asarray(theta_unwrap, dtype=np.float64)
    spans: List[CycleSpan] = []
    for start, end in cycles:
        start, end = int(start), int(end)
        quarter = max(1, (end - start) // 4)
        direction: Direction = "ccw" if theta[min(end - 1, start + quarter)] >= theta[start] else "cw"
        spans.append(CycleSpan(start, mid_point(theta, start, end, direction), end))
    return spans
