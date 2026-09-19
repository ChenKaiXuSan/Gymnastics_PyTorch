"""Cycle phase: derivation, normalisation, encoding and estimation.

The cycle-aware fusion model conditions its long-term motion branch on the
*phase* of the motion cycle and trains with a periodicity objective that
compares poses at equal phase in consecutive cycles.  This module holds every
phase-related computation so the definition of "phase" is unique.

Definitions:
    A cycle is a half-open frame range ``[start, end)``.  The phase of frame
    ``t`` inside that cycle is

        phi(t) = (t - start) / (end - start)          in [0, 1)

    Frames outside every annotated cycle have no phase (``phase_valid`` is
    false and ``cycle_index`` is -1).

Phase normalisation (resampling):
    Cycles differ in duration.  To give the long-term branch a fixed number of
    samples per cycle (``samples_per_cycle = S``), each complete cycle is
    resampled at phases ``k / S`` for ``k = 0..S-1`` by linear interpolation
    of joint positions in time.  The *physical* timestamp of every resampled
    sample is kept, so the interval ``delta_t`` between samples still measures
    real time and :func:`gymnastics.fusion.cycle_aware.velocity.compute_velocity`
    yields physical velocity.  Sequences without cycle annotations are left
    untouched.

Phase encoding:
    The long-term transformer receives

        [sin(2 pi h phi), cos(2 pi h phi)]   for harmonics h = 1..H

    which is continuous across the cycle boundary (phi = 1 wraps to 0) and
    zero for frames without a valid phase.

Cycle estimation:
    For datasets without cycle annotations (FreeMan, Unity) an optional
    autocorrelation-based estimator finds the dominant period of a 1-D motion
    signal and places cycle boundaries at its successive peaks.  It is a
    convenience for experiments, not part of the method; when it is not used
    those datasets are trained without phase.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .sample import DualViewSample


def phase_from_cycle_bounds(
    num_frames: int,
    cycle_bounds: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive per-frame phase from cycle ranges.

    Args:
        num_frames: Number of frames ``T``.
        cycle_bounds: Increasing, non-overlapping ``(start, end)`` ranges.

    Returns:
        Tuple ``(phase, phase_valid, cycle_index)`` with shapes ``[T]``:
        float32 phase in ``[0, 1)``, bool validity and int64 cycle index
        (``-1`` outside every cycle).
    """
    phase = np.zeros(num_frames, dtype=np.float32)
    valid = np.zeros(num_frames, dtype=bool)
    cycle_index = np.full(num_frames, -1, dtype=np.int64)
    for index, (start, end) in enumerate(cycle_bounds):
        if start < 0 or end > num_frames or end <= start:
            raise ValueError(f"cycle ({start}, {end}) is outside [0, {num_frames})")
        frames = np.arange(start, end)
        phase[start:end] = (frames - start) / float(end - start)
        valid[start:end] = True
        cycle_index[start:end] = index
    return phase, valid, cycle_index


def _interpolate_view(
    points: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    query_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate ``[T, J, 3]`` points at ``query_times``.

    A query is valid when both bracketing frames are valid (or the single
    frame it coincides with).
    """
    index = np.searchsorted(timestamps, query_times, side="right") - 1
    index = np.clip(index, 0, len(timestamps) - 2) if len(timestamps) > 1 else np.zeros_like(index)
    nxt = np.minimum(index + 1, len(timestamps) - 1)
    t0, t1 = timestamps[index], timestamps[nxt]
    span = np.where(t1 > t0, t1 - t0, 1.0)
    weight = np.clip((query_times - t0) / span, 0.0, 1.0).astype(np.float32)
    exact0 = np.isclose(weight, 0.0)
    exact1 = np.isclose(weight, 1.0)
    interpolated = (1.0 - weight)[:, None, None] * points[index] + weight[:, None, None] * points[nxt]
    both = valid[index] & valid[nxt]
    query_valid = np.where(exact0[:, None], valid[index], np.where(exact1[:, None], valid[nxt], both))
    interpolated = np.where(exact0[:, None, None], points[index], interpolated)
    interpolated = np.where(exact1[:, None, None], points[nxt], interpolated)
    return np.where(query_valid[..., None], interpolated, 0.0).astype(np.float32), query_valid


def normalize_sample_to_phase(sample: DualViewSample, samples_per_cycle: int) -> DualViewSample:
    """Resample every complete cycle of ``sample`` to ``samples_per_cycle`` samples.

    Only frames that belong to an annotated cycle are kept; the output
    therefore contains ``len(cycle_bounds) * samples_per_cycle`` samples whose
    ``cycle_bounds`` are exact multiples of ``samples_per_cycle``.  Physical
    timestamps are preserved at the resampled positions.  Samples without
    cycle annotations are returned unchanged.

    Args:
        sample: Input sample in the common representation.
        samples_per_cycle: Number of samples per normalised cycle ``S``.

    Returns:
        The phase-normalised sample (or ``sample`` itself when it has no
        cycles).

    Raises:
        ValueError: If ``samples_per_cycle`` is smaller than two.
    """
    if samples_per_cycle < 2:
        raise ValueError("samples_per_cycle must be at least 2")
    if not sample.cycle_bounds:
        return sample
    timestamps = sample.timestamps
    fps = float(sample.metadata.get("fps", 0.0)) if "fps" in sample.metadata else 0.0
    nominal_dt = 1.0 / fps if fps > 0 else float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    query_times: list[np.ndarray] = []
    new_bounds: list[tuple[int, int]] = []
    for cycle_number, (start, end) in enumerate(sample.cycle_bounds):
        cycle_start = timestamps[start]
        # The cycle ends where the next frame would begin: use the following
        # frame's timestamp when it exists, otherwise extrapolate one interval.
        cycle_end = timestamps[end] if end < len(timestamps) else timestamps[end - 1] + nominal_dt
        phases = np.arange(samples_per_cycle, dtype=np.float64) / samples_per_cycle
        query_times.append(cycle_start + phases * (cycle_end - cycle_start))
        new_bounds.append((cycle_number * samples_per_cycle, (cycle_number + 1) * samples_per_cycle))
    query = np.concatenate(query_times)
    view_a, valid_a = _interpolate_view(sample.view_a, sample.valid_a, timestamps, query)
    view_b, valid_b = _interpolate_view(sample.view_b, sample.valid_b, timestamps, query)
    reference = reference_valid = None
    if sample.reference is not None and sample.reference_valid is not None:
        reference, reference_valid = _interpolate_view(sample.reference, sample.reference_valid, timestamps, query)
    transform = None
    if sample.transform_a is not None:
        nearest = np.clip(np.searchsorted(timestamps, query, side="left"), 0, len(timestamps) - 1)
        transform = replace(
            sample.transform_a,
            rotation=sample.transform_a.rotation[nearest],
            origin=sample.transform_a.origin[nearest],
            valid=sample.transform_a.valid[nearest],
        )
    # Guard against duplicate timestamps produced by extremely short cycles.
    query = np.maximum.accumulate(query + np.arange(len(query)) * 1e-9)
    metadata = dict(sample.metadata)
    metadata.update({"phase_normalized": True, "samples_per_cycle": int(samples_per_cycle)})
    return DualViewSample(
        dataset=sample.dataset,
        subject_id=sample.subject_id,
        sequence_id=sample.sequence_id,
        view_a=view_a,
        view_b=view_b,
        valid_a=valid_a,
        valid_b=valid_b,
        timestamps=query,
        joint_names=sample.joint_names,
        cycle_bounds=tuple(new_bounds),
        reference=reference,
        reference_valid=reference_valid,
        transform_a=transform,
        metadata=metadata,
    )


class PhaseEncoding(nn.Module):
    """Sinusoidal encoding of cycle phase.

    Produces ``[sin(2 pi h phi), cos(2 pi h phi)]`` for ``h = 1..harmonics``
    and zeros where the phase is invalid, so the long-term branch degrades
    gracefully to phase-free operation on unannotated data.

    Attributes:
        harmonics: Number of harmonics ``H``; the output has ``2H`` channels.
    """

    def __init__(self, harmonics: int = 1) -> None:
        super().__init__()
        if harmonics < 1:
            raise ValueError("harmonics must be positive")
        self.harmonics = int(harmonics)

    @property
    def channels(self) -> int:
        """Number of output channels ``2H``."""
        return 2 * self.harmonics

    def forward(self, phase: torch.Tensor, phase_valid: torch.Tensor) -> torch.Tensor:
        """Encode phase.

        Args:
            phase: ``[B, T]`` phase in ``[0, 1)``.
            phase_valid: ``[B, T]`` bool validity.

        Returns:
            ``[B, T, 2H]`` encoding, zero where invalid.
        """
        if phase.shape != phase_valid.shape or phase.ndim != 2:
            raise ValueError("phase and phase_valid must both have shape [B, T]")
        angle = 2.0 * torch.pi * phase[..., None] * torch.arange(1, self.harmonics + 1, device=phase.device, dtype=phase.dtype)
        encoding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.where(phase_valid[..., None], encoding, torch.zeros_like(encoding))


def estimate_cycle_bounds(
    signal: np.ndarray,
    *,
    min_period: int,
    max_period: int,
    min_cycles: int = 2,
) -> tuple[tuple[int, int], ...]:
    """Estimate cycle boundaries of a periodic 1-D signal.

    The dominant period is the lag of the highest autocorrelation peak inside
    ``[min_period, max_period]``.  Cycle boundaries are then the successive
    local maxima of the smoothed signal, each at least ``0.6 * period`` after
    the previous one; consecutive maxima delimit one cycle.

    Args:
        signal: ``[T]`` real-valued motion signal (NaN allowed; filled by
            linear interpolation).
        min_period: Shortest admissible period in frames.
        max_period: Longest admissible period in frames.
        min_cycles: Minimum number of cycles required; otherwise no cycles
            are returned.

    Returns:
        Tuple of ``(start, end)`` ranges, possibly empty.

    Raises:
        ValueError: If the period range is invalid.
    """
    if min_period < 2 or max_period <= min_period:
        raise ValueError("require 2 <= min_period < max_period")
    values = np.asarray(signal, dtype=np.float64).copy()
    frames = len(values)
    if frames < 2 * min_period:
        return ()
    missing = ~np.isfinite(values)
    if missing.all():
        return ()
    if missing.any():
        values[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing), values[~missing])
    values = values - values.mean()
    if not np.any(values):
        return ()
    autocorrelation = np.correlate(values, values, mode="full")[frames - 1 :]
    autocorrelation = autocorrelation / max(autocorrelation[0], 1e-12)
    upper = min(max_period, frames - 1)
    if upper <= min_period:
        return ()
    period = int(min_period + np.argmax(autocorrelation[min_period : upper + 1]))
    if autocorrelation[period] <= 0.0:
        return ()
    kernel = max(3, period // 8)
    smoothed = np.convolve(values, np.ones(kernel) / kernel, mode="same")
    spacing = max(2, int(round(0.6 * period)))
    peaks: list[int] = []
    for index in range(1, frames - 1):
        if smoothed[index] >= smoothed[index - 1] and smoothed[index] > smoothed[index + 1]:
            if peaks and index - peaks[-1] < spacing:
                if smoothed[index] > smoothed[peaks[-1]]:
                    peaks[-1] = index
                continue
            peaks.append(index)
    bounds = tuple((peaks[i], peaks[i + 1]) for i in range(len(peaks) - 1) if min_period <= peaks[i + 1] - peaks[i] <= max_period)
    return bounds if len(bounds) >= min_cycles else ()


def trunk_twist_signal(points: np.ndarray, valid: np.ndarray, left_shoulder: int, right_shoulder: int) -> np.ndarray:
    """Signed shoulder-line rotation about the vertical axis of the body frame.

    In the canonical pelvis frame the hips define the x-axis, so the angle of
    the shoulder line in the x-z plane measures trunk twist, the primary
    periodic signal of the gymnastics exercises studied in this project.

    Args:
        points: ``[T, J, 3]`` canonical keypoints.
        valid: ``[T, J]`` validity.
        left_shoulder: Index of the left shoulder joint.
        right_shoulder: Index of the right shoulder joint.

    Returns:
        ``[T]`` angle in radians (NaN where a shoulder is invalid).
    """
    shoulder = points[:, right_shoulder] - points[:, left_shoulder]
    angle = np.arctan2(shoulder[:, 2], shoulder[:, 0]).astype(np.float64)
    usable = valid[:, left_shoulder] & valid[:, right_shoulder]
    return np.where(usable, angle, np.nan)
