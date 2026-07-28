"""Cycle quality-control contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CycleQC:
    globally_eligible: bool
    exclusion_reasons: tuple[str, ...]
    joint_valid_fraction: np.ndarray

    def joints_eligible(self, indices: tuple[int, ...]) -> bool:
        if not indices:
            return True
        return bool(
            np.all(self.joint_valid_fraction[np.asarray(indices)] >= 0.80)
        )


def evaluate_cycle_qc(
    frame_valid: np.ndarray,
    joint_valid: np.ndarray,
    timestamps: np.ndarray,
    *,
    minimum_valid_frame_fraction: float = 0.80,
    minimum_valid_frames: int = 60,
) -> CycleQC:
    """Apply prespecified cycle-level QC without consulting cohort labels."""
    frame = np.asarray(frame_valid, dtype=bool)
    joints = np.asarray(joint_valid, dtype=bool)
    time = np.asarray(timestamps, dtype=np.float64)
    if frame.ndim != 1 or joints.ndim != 2 or time.ndim != 1:
        raise ValueError("QC arrays have invalid dimensions")
    if len(frame) != len(time) or joints.shape[0] != len(frame):
        raise ValueError("QC arrays have inconsistent frame counts")
    reasons: list[str] = []
    valid_count = int(frame.sum())
    if valid_count / max(len(frame), 1) < minimum_valid_frame_fraction:
        reasons.append("valid_frame_fraction")
    if valid_count < minimum_valid_frames:
        reasons.append("minimum_valid_frames")
    if not np.all(np.isfinite(time)):
        reasons.append("timestamps_not_finite")
    elif len(time) < 2 or not np.all(np.diff(time) > 0):
        reasons.append("timestamps_not_strictly_increasing")
    joint_fraction = (joints & frame[:, None]).mean(axis=0)
    return CycleQC(
        globally_eligible=not reasons,
        exclusion_reasons=tuple(reasons),
        joint_valid_fraction=joint_fraction,
    )


def interpolate_short_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    maximum_gap_fraction: float = 0.10,
) -> np.ndarray:
    """Linearly bridge internal gaps no longer than the declared fraction."""
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if data.ndim < 1 or mask.shape != (len(data),):
        raise ValueError("valid mask must match the first value dimension")
    if mask.all():
        return data.copy()
    missing = ~mask
    starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
    ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
    for start, end in zip(starts, ends, strict=True):
        if start == 0 or end == len(data) - 1:
            raise ValueError("edge gap cannot be interpolated")
        if (end - start + 1) / len(data) > maximum_gap_fraction:
            raise ValueError("internal gap exceeds interpolation limit")
    flat = data.reshape(len(data), -1)
    output = flat.copy()
    positions = np.arange(len(data))
    for column in range(flat.shape[1]):
        output[missing, column] = np.interp(
            positions[missing],
            positions[mask],
            flat[mask, column],
        )
    return output.reshape(data.shape)
