"""Cycle preprocessing contracts."""

from __future__ import annotations

import numpy as np


def phase_normalize(
    values: np.ndarray,
    timestamps: np.ndarray,
    *,
    points: int = 101,
) -> np.ndarray:
    """Linearly resample a complete cycle onto an inclusive phase grid."""
    data = np.asarray(values, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    if data.ndim < 1 or len(data) != len(time) or len(time) < 2:
        raise ValueError("phase inputs require at least two matching frames")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        raise ValueError("phase timestamps must be finite and increasing")
    if points < 2:
        raise ValueError("phase grid requires at least two points")
    source_phase = (time - time[0]) / (time[-1] - time[0])
    target_phase = np.linspace(0.0, 1.0, points)
    flat = data.reshape(len(data), -1)
    normalized = np.empty((points, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        normalized[:, column] = np.interp(
            target_phase,
            source_phase,
            flat[:, column],
        )
    return normalized.reshape((points, *data.shape[1:]))


def align_rotation_direction(
    theta: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Orient the dominant median-centred axial excursion positively."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("axial rotation must be one finite trajectory")
    centred = values - np.median(values)
    positive = float(np.max(centred))
    negative = abs(float(np.min(centred)))
    sign = 1 if positive >= negative else -1
    return values * sign, sign


def normalized_cycle_positions(count: int) -> np.ndarray:
    """Map ordered repetitions from first=0 to last=1."""
    if count < 1:
        raise ValueError("a person must have at least one cycle")
    if count == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, count)
