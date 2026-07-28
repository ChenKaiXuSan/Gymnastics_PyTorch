from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gymnastics.analysis.cohort_cycle.features import (
    angular_jerk,
    compute_core_scalars,
    extract_publication_features,
    leave_one_cycle_out_repeatability,
    peak_rotation_phase,
)


def _upright_pose(frames: int) -> np.ndarray:
    points = np.zeros((frames, 70, 3), dtype=np.float64)
    points[:, 9] = (-0.2, 0.0, 0.0)
    points[:, 10] = (0.2, 0.0, 0.0)
    points[:, 5] = (-0.3, 1.0, 0.0)
    points[:, 6] = (0.3, 1.0, 0.0)
    points[:, 41] = (0.7, 0.5, 0.7)
    points[:, 62] = (-0.7, 0.5, -0.7)
    return points


def test_core_scalars_recover_known_rom_speed_tilt_and_duration():
    """Wrong units or trunk geometry must fail on an analytic pose."""
    timestamps = np.linspace(0.0, 2.0, 101)
    theta = np.linspace(0.0, 1.0, 101)
    omega = np.full(101, 0.5)
    values = compute_core_scalars(
        theta,
        omega,
        timestamps,
        _upright_pose(101),
        direction_sign=1,
    )

    assert values["trunk_axial_rotation_rom"] == pytest.approx(0.9)
    assert values["angular_speed_p95"] == pytest.approx(0.5)
    assert values["trunk_tilt_p95"] == pytest.approx(0.0, abs=1e-7)
    assert values["cycle_duration"] == pytest.approx(2.0)
    assert values["wrist_lead_p95"] == pytest.approx(np.pi / 4.0)


def test_peak_rotation_phase_uses_absolute_deviation_from_cycle_median():
    """Peak phase must represent the dominant excursion, not signed maximum."""
    theta = np.zeros(101)
    theta[73] = -2.0
    assert peak_rotation_phase(theta) == pytest.approx(0.73)


def test_angular_jerk_rejects_negligible_rom_and_is_finite_otherwise():
    """The normalized jerk denominator must not explode for static cycles."""
    timestamps = np.linspace(0.0, 1.0, 101)
    with pytest.raises(ValueError, match="negligible"):
        angular_jerk(np.zeros(101), timestamps)

    value = angular_jerk(np.sin(2.0 * np.pi * timestamps), timestamps)
    assert np.isfinite(value)


def test_leave_one_cycle_out_repeatability_matches_known_displacement():
    """Using the evaluated cycle in its own template would bias error downward."""
    base = np.zeros((101, 20, 3), dtype=np.float64)
    shifted = np.ones_like(base)
    identical_errors = leave_one_cycle_out_repeatability(
        np.stack([base, base, base])
    )
    np.testing.assert_allclose(identical_errors, 0.0)

    errors = leave_one_cycle_out_repeatability(
        np.stack([base, base, shifted])
    )
    assert errors[2] == pytest.approx(1.0)
    assert errors[0] == pytest.approx(0.5)


def test_extract_publication_writes_cycle_person_qc_and_phase_artifacts(
    tmp_path: Path,
):
    """Dropping cycle order, repeatability, or QC provenance breaks analysis."""
    publication = tmp_path / "oof"
    rows = []
    timestamps = np.linspace(0.0, 2.0, 101)
    theta = np.sin(2.0 * np.pi * timestamps / 2.0)
    omega = np.gradient(theta, timestamps)
    for cycle_index in range(4):
        cycle_id = f"cycle_{cycle_index:03d}"
        cycle_root = publication / "person_1" / cycle_id
        cycle_root.mkdir(parents=True)
        np.savez_compressed(
            cycle_root / "prediction.npz",
            kpts_body=_upright_pose(101),
            theta_fused_rad=theta,
            omega_fused_rad_s=omega,
            timestamps=timestamps,
            frame_valid=np.ones(101, dtype=bool),
            joint_valid=np.ones((101, 70), dtype=bool),
        )
        rows.append(
            {
                "person_id": "1",
                "cohort": "elderly",
                "outer_fold": "0",
                "cycle_id": cycle_id,
                "prediction_path": (
                    f"person_1/{cycle_id}/prediction.npz"
                ),
            }
        )
    with (publication / "oof_provenance.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "features"

    summary = extract_publication_features(publication, output)

    assert summary["cycles"] == 4
    assert summary["people"] == 1
    cycle_table = pd.read_csv(output / "cycle_features.csv")
    assert list(cycle_table["normalized_cycle_position"]) == pytest.approx(
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    assert list(cycle_table["whole_body_repeatability"]) == pytest.approx(
        [0.0, 0.0, 0.0, 0.0]
    )
    person_table = pd.read_csv(output / "person_features.csv")
    assert len(person_table) == 1
    assert person_table.loc[0, "eligible_cycles"] == 4
    assert (output / "qc_exclusions.csv").is_file()
    phase = np.load(output / "phase_curves.npz")
    assert phase["theta"].shape == (4, 101)
