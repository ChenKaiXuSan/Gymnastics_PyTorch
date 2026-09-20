"""Multiple-comparison correction contracts."""

from __future__ import annotations

import numpy as np


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm family-wise adjusted p-values in original order."""
    values = _validate(p_values)
    count = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        ranked * np.arange(count, 0, -1)
    )
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg FDR adjusted p-values in original order."""
    values = _validate(p_values)
    count = len(values)
    order = np.argsort(values)
    ranked = values[order]
    raw = ranked * count / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _validate(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("p-values must be one finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    return values
