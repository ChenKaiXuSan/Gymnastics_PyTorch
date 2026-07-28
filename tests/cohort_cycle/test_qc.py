from __future__ import annotations

import numpy as np
import pytest

from gymnastics.analysis.cohort_cycle.qc import (
    evaluate_cycle_qc,
    interpolate_short_gaps,
)


def test_cycle_qc_enforces_global_frame_and_timestamp_contract():
    """Too few valid frames or non-increasing time must exclude a cycle."""
    timestamps = np.linspace(0.0, 2.0, 100)
    frame_valid = np.ones(100, dtype=bool)
    joint_valid = np.ones((100, 70), dtype=bool)

    accepted = evaluate_cycle_qc(frame_valid, joint_valid, timestamps)
    assert accepted.globally_eligible is True

    frame_valid[:21] = False
    rejected = evaluate_cycle_qc(frame_valid, joint_valid, timestamps)
    assert rejected.globally_eligible is False
    assert "valid_frame_fraction" in rejected.exclusion_reasons

    timestamps[50] = timestamps[49]
    rejected_time = evaluate_cycle_qc(
        np.ones(100, dtype=bool),
        joint_valid,
        timestamps,
    )
    assert rejected_time.globally_eligible is False
    assert "timestamps_not_strictly_increasing" in rejected_time.exclusion_reasons


def test_cycle_qc_tracks_metric_joint_eligibility():
    """A metric must not use a joint valid in less than 80% of frames."""
    joint_valid = np.ones((100, 70), dtype=bool)
    joint_valid[:21, 5] = False
    qc = evaluate_cycle_qc(
        np.ones(100, dtype=bool),
        joint_valid,
        np.linspace(0.0, 1.0, 100),
    )

    assert qc.joints_eligible((6, 9, 10)) is True
    assert qc.joints_eligible((5, 6, 9, 10)) is False


def test_interpolation_bridges_only_short_internal_gaps():
    """Long or edge gaps must not be silently synthesized."""
    values = np.arange(10, dtype=np.float64)
    valid = np.ones(10, dtype=bool)
    valid[4] = False

    interpolated = interpolate_short_gaps(
        values,
        valid,
        maximum_gap_fraction=0.20,
    )
    assert interpolated[4] == pytest.approx(4.0)

    valid[4:7] = False
    with pytest.raises(ValueError, match="gap exceeds"):
        interpolate_short_gaps(
            values,
            valid,
            maximum_gap_fraction=0.20,
        )

    edge_valid = np.ones(10, dtype=bool)
    edge_valid[0] = False
    with pytest.raises(ValueError, match="edge gap"):
        interpolate_short_gaps(
            values,
            edge_valid,
            maximum_gap_fraction=0.20,
        )
