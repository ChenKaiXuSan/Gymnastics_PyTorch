from __future__ import annotations

import numpy as np
import pytest

from gymnastics.analysis.cohort_cycle.preprocess import (
    align_rotation_direction,
    normalized_cycle_positions,
    phase_normalize,
)


def test_phase_normalize_uses_linear_101_point_grid():
    """Changing endpoints or using overshooting interpolation breaks comparability."""
    timestamps = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values = 2.0 * timestamps

    normalized = phase_normalize(values, timestamps, points=101)

    assert normalized.shape == (101,)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[50] == pytest.approx(1.0)
    assert normalized[-1] == pytest.approx(2.0)


def test_direction_alignment_flips_negative_dominant_excursion():
    """Clockwise and counter-clockwise cycles must share one signed convention."""
    positive = np.array([0.0, 0.2, 1.0, 0.1])
    negative = -positive

    aligned_positive, positive_sign = align_rotation_direction(positive)
    aligned_negative, negative_sign = align_rotation_direction(negative)

    assert positive_sign == 1
    assert negative_sign == -1
    np.testing.assert_allclose(aligned_positive, aligned_negative)


def test_normalized_cycle_positions_span_first_to_last():
    """Cycle order must be comparable for people with different cycle counts."""
    np.testing.assert_allclose(
        normalized_cycle_positions(4),
        np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
    )
    np.testing.assert_allclose(normalized_cycle_positions(1), np.array([0.0]))
