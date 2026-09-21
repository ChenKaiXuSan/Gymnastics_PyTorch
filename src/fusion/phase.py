"""Cycle phase: derivation from records, normalisation and encoding.

The cycle-aware fusion model conditions its long-term motion branch on the
*phase* of the motion cycle and trains with a periodicity objective that
compares poses at equal phase in consecutive cycles.  This module holds every
phase-related computation so the definition of "phase" is unique.

Cycle *detection* is not done here.  Cycles and their middles are detected
offline by :mod:`cycle_alignment.cycles` (``python -m cycle_alignment align`` /
``python -m cycle_alignment cycles``) and read from record files; the training code
only derives phase from those annotations.

Definitions:
    A cycle is a half-open frame range ``[start, end)`` with an optional
    middle frame ``mid`` (the turn-around point, ``start < mid < end``).  The
    phase of frame ``t`` inside that cycle is

        phi(t) = (t - start) / (end - start)          in [0, 1)

    and its *half* is 0 on ``[start, mid)`` (outward motion) and 1 on
    ``[mid, end)`` (return motion).  Frames outside every annotated cycle
    have no phase (``phase_valid`` false, ``cycle_index`` and ``half_index``
    -1).

Phase normalisation (resampling):
    Cycles differ in duration.  To give the long-term branch a fixed number of
    samples per cycle (``samples_per_cycle = S``), each complete cycle is
    resampled by linear interpolation of joint positions in time:

    * with middles, each half is resampled to ``S / 2`` samples, so the
      middle lands exactly at phase 0.5 (sample ``start + S/2``) and the
      time-reversed mirror of sample ``k`` of a cycle is sample ``S - k``
      (needed by the symmetry analysis; ``S`` must be even);
    * without middles, the whole cycle is resampled at phases ``k / S``.

    The *physical* timestamp of every resampled sample is kept, so the
    interval ``delta_t`` between samples still measures real time and
    :func:`fusion.velocity.compute_velocity` yields
    physical velocity.  Sequences without cycle annotations are left
    untouched.

Phase encoding:
    The long-term transformer receives

        [sin(2 pi h phi), cos(2 pi h phi)]   for harmonics h = 1..H

    which is continuous across the cycle boundary (phi = 1 wraps to 0) and
    zero for frames without a valid phase.
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
    cycle_mids: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive per-frame phase, cycle index and half index from annotations.

    Args:
        num_frames: Number of frames ``T``.
        cycle_bounds: Increasing, non-overlapping ``(start, end)`` ranges.
        cycle_mids: Optional middle frame per cycle (same length as
            ``cycle_bounds`` or empty).

    Returns:
        Tuple ``(phase, phase_valid, cycle_index, half_index)`` with shapes
        ``[T]``: float32 phase in ``[0, 1)``, bool validity, int64 cycle
        index (``-1`` outside every cycle) and int64 half index (0 outward,
        1 return, ``-1`` outside cycles or when no middles are known).
    """
    if cycle_mids and len(cycle_mids) != len(cycle_bounds):
        raise ValueError("cycle_mids must be empty or have one entry per cycle")
    phase = np.zeros(num_frames, dtype=np.float32)
    valid = np.zeros(num_frames, dtype=bool)
    cycle_index = np.full(num_frames, -1, dtype=np.int64)
    half_index = np.full(num_frames, -1, dtype=np.int64)
    for index, (start, end) in enumerate(cycle_bounds):
        if start < 0 or end > num_frames or end <= start:
            raise ValueError(f"cycle ({start}, {end}) is outside [0, {num_frames})")
        frames = np.arange(start, end)
        phase[start:end] = (frames - start) / float(end - start)
        valid[start:end] = True
        cycle_index[start:end] = index
        if cycle_mids:
            mid = int(cycle_mids[index])
            if not start < mid < end:
                raise ValueError(f"cycle middle {mid} is outside ({start}, {end})")
            half_index[start:mid] = 0
            half_index[mid:end] = 1
    return phase, valid, cycle_index, half_index


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
    ``cycle_bounds`` are exact multiples of ``samples_per_cycle``.  When the
    sample carries cycle middles, each half is resampled to
    ``samples_per_cycle / 2`` samples so every middle lands at phase 0.5.
    Physical timestamps are preserved at the resampled positions.  Samples
    without cycle annotations are returned unchanged.

    Args:
        sample: Input sample in the common representation.
        samples_per_cycle: Number of samples per normalised cycle ``S``.

    Returns:
        The phase-normalised sample (or ``sample`` itself when it has no
        cycles).

    Raises:
        ValueError: If ``samples_per_cycle`` is smaller than two, or odd
            while the sample has middles.
    """
    if samples_per_cycle < 2:
        raise ValueError("samples_per_cycle must be at least 2")
    if not sample.cycle_bounds:
        return sample
    use_halves = bool(sample.cycle_mids)
    if use_halves and samples_per_cycle % 2:
        raise ValueError("samples_per_cycle must be even when cycles have middles")
    timestamps = sample.timestamps
    fps = float(sample.metadata.get("fps", 0.0)) if "fps" in sample.metadata else 0.0
    nominal_dt = 1.0 / fps if fps > 0 else float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    query_times: list[np.ndarray] = []
    new_bounds: list[tuple[int, int]] = []
    new_mids: list[int] = []
    half = samples_per_cycle // 2
    for cycle_number, (start, end) in enumerate(sample.cycle_bounds):
        cycle_start = timestamps[start]
        # The cycle ends where the next frame would begin: use the following
        # frame's timestamp when it exists, otherwise extrapolate one interval.
        cycle_end = timestamps[end] if end < len(timestamps) else timestamps[end - 1] + nominal_dt
        if use_halves:
            mid_time = timestamps[sample.cycle_mids[cycle_number]]
            outward = cycle_start + np.arange(half, dtype=np.float64) / half * (mid_time - cycle_start)
            back = mid_time + np.arange(half, dtype=np.float64) / half * (cycle_end - mid_time)
            query_times.append(np.concatenate((outward, back)))
            new_mids.append(cycle_number * samples_per_cycle + half)
        else:
            phases = np.arange(samples_per_cycle, dtype=np.float64) / samples_per_cycle
            query_times.append(cycle_start + phases * (cycle_end - cycle_start))
        new_bounds.append((cycle_number * samples_per_cycle, (cycle_number + 1) * samples_per_cycle))
    query = np.concatenate(query_times)
    view_a, valid_a = _interpolate_view(sample.view_a, sample.valid_a, timestamps, query)
    view_b, valid_b = _interpolate_view(sample.view_b, sample.valid_b, timestamps, query)
    reference = reference_valid = None
    if sample.reference is not None and sample.reference_valid is not None:
        reference, reference_valid = _interpolate_view(sample.reference, sample.reference_valid, timestamps, query)
    nearest = np.clip(np.searchsorted(timestamps, query, side="left"), 0, len(timestamps) - 1)

    def resample_transform(transform):
        # Rotations are taken from the nearest original frame (no interpolation).
        if transform is None:
            return None
        return replace(transform, rotation=transform.rotation[nearest], origin=transform.origin[nearest], valid=transform.valid[nearest])

    transform_a = resample_transform(sample.transform_a)
    transform_b = resample_transform(sample.transform_b)
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
        cycle_mids=tuple(new_mids),
        reference=reference,
        reference_valid=reference_valid,
        reference_canonical=sample.reference_canonical,
        transform_a=transform_a,
        transform_b=transform_b,
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
