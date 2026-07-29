from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gymnastics.analysis.cohort_cycle.statistics import (
    adjust_phase_cluster_families,
    analyze_feature_artifacts,
    bootstrap_icc,
    cliffs_delta,
    fit_mixed_effect,
    hedges_g,
    intraclass_correlation,
    person_label_permutation,
)


def _synthetic_cycles() -> pd.DataFrame:
    rows = []
    for person_index in range(30):
        cohort = "student" if person_index < 15 else "elderly"
        cohort_value = float(cohort == "elderly")
        person_offset = ((person_index * 7) % 11 - 5) * 0.08
        for cycle_index, position in enumerate(np.linspace(0.0, 1.0, 6)):
            rows.append(
                {
                    "person_id": str(person_index),
                    "cohort": cohort,
                    "outer_fold": person_index % 5,
                    "normalized_cycle_position": position,
                    "outcome": (
                        3.0
                        + 2.0 * cohort_value
                        + 0.5 * position
                        + 1.0 * cohort_value * position
                        + person_offset
                        + 0.01 * np.sin(cycle_index)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_synthetic_feature_artifacts(feature_root):
    feature_root.mkdir()
    cycles = _synthetic_cycles()
    outcomes = (
        "trunk_axial_rotation_rom",
        "angular_speed_p95",
        "peak_rotation_phase",
        "trunk_tilt_p95",
        "wrist_lead_p95",
        "cycle_duration",
        "log_dimensionless_angular_jerk",
        "whole_body_repeatability",
    )
    for index, outcome in enumerate(outcomes, start=1):
        cycles[outcome] = cycles["outcome"] + index
    cycles["eligible"] = True
    cycles.to_csv(feature_root / "cycle_features.csv", index=False)

    person_rows = []
    for person_id, group in cycles.groupby("person_id"):
        row = {
            "person_id": person_id,
            "cohort": group["cohort"].iloc[0],
            "outer_fold": group["outer_fold"].iloc[0],
        }
        for outcome in outcomes:
            values = group[outcome].to_numpy()
            row[f"{outcome}_median"] = np.median(values)
            row[f"{outcome}_mad"] = np.median(
                np.abs(values - np.median(values))
            )
        person_rows.append(row)
    pd.DataFrame(person_rows).to_csv(
        feature_root / "person_features.csv",
        index=False,
    )
    person_ids = np.asarray(
        cycles.drop_duplicates("person_id")["person_id"].tolist(),
        dtype="U",
    )
    cohorts = np.asarray(
        cycles.drop_duplicates("person_id")["cohort"].tolist(),
        dtype="U",
    )
    phase = np.zeros((len(person_ids), 101))
    np.savez_compressed(
        feature_root / "phase_curves.npz",
        person_id=person_ids,
        cohort=cohorts,
        theta=phase,
        omega=phase,
        tilt=phase,
        wrist=phase,
    )


def test_mixed_effect_cohort_effect_is_reported_at_mid_repetition():
    """The primary cohort contrast must describe a representative repetition."""
    result = fit_mixed_effect(
        _synthetic_cycles(),
        "outcome",
        try_random_slope=False,
        cycle_reference=0.5,
        include_outer_fold=True,
    )

    assert result["converged"] is True
    assert result["cycle_reference"] == 0.5
    assert result["include_outer_fold"] is True
    assert result["cohort_effect"] == pytest.approx(2.5, abs=0.15)
    assert result["cycle_effect"] == pytest.approx(0.5, abs=0.10)
    assert result["interaction_effect"] == pytest.approx(1.0, abs=0.10)
    assert result["random_intercept_variance"] >= 0.0
    assert result["random_slope_variance"] >= 0.0
    assert result["cohort_effect_se"] > 0.0
    assert result["maximum_absolute_standardized_residual"] >= 0.0


def test_mixed_effect_can_omit_artificial_outer_fold_adjustment():
    """Outer-fold fixed effects must be auditable as a sensitivity choice."""
    result = fit_mixed_effect(
        _synthetic_cycles(),
        "outcome",
        try_random_slope=False,
        cycle_reference=0.5,
        include_outer_fold=False,
    )

    assert result["include_outer_fold"] is False
    assert "C(outer_fold)" not in result["formula"]
    assert result["cohort_effect"] == pytest.approx(2.5, abs=0.15)


def test_person_label_permutation_and_effect_sizes_detect_separation():
    """Permutation must operate on people and preserve the declared direction."""
    student = np.arange(12, dtype=float)
    elderly = student + 20.0
    table = pd.DataFrame(
        {
            "person_id": [f"s{i}" for i in range(12)]
            + [f"e{i}" for i in range(12)],
            "cohort": ["student"] * 12 + ["elderly"] * 12,
            "value": np.concatenate([student, elderly]),
        }
    )

    result = person_label_permutation(
        table,
        "value",
        permutations=999,
        seed=7,
    )

    assert result["median_difference"] == 20.0
    assert result["median_difference_ci_low"] <= 20.0
    assert result["median_difference_ci_high"] >= 20.0
    assert result["elderly_q25"] < result["elderly_q75"]
    assert result["student_q25"] < result["student_q75"]
    assert result["p_value"] < 0.01
    assert hedges_g(elderly, student) > 2.0
    assert cliffs_delta(elderly, student) == 1.0


def test_intraclass_correlation_is_high_for_stable_person_signatures():
    """Treating repeated cycles as independent would erase person repeatability."""
    rows = []
    for person_id in range(8):
        for cycle in range(5):
            rows.append(
                {
                    "person_id": str(person_id),
                    "value": person_id + 0.001 * cycle,
                }
            )
    table = pd.DataFrame(rows)
    assert intraclass_correlation(table, "value") > 0.99
    low, high = bootstrap_icc(
        table,
        "value",
        samples=199,
        seed=4,
    )
    assert low > 0.99
    assert high <= 1.0


def test_analysis_writes_eight_core_models_and_corrected_families(
    tmp_path,
):
    """A partial or uncorrected result publication must not reach reporting."""
    feature_root = tmp_path / "features"
    _write_synthetic_feature_artifacts(feature_root)
    output = tmp_path / "analysis"

    summary = analyze_feature_artifacts(
        feature_root,
        output,
        permutations=99,
        seed=5,
        try_random_slope=False,
        sensitivity_sources={"face": feature_root},
    )

    assert summary["core_outcomes"] == 8
    core = pd.read_csv(output / "core_mixed_models.csv")
    no_fold = pd.read_csv(output / "core_mixed_models_no_fold.csv")
    variability = pd.read_csv(output / "variability_results.csv")
    assert len(core) == 8
    assert len(no_fold) == 8
    assert set(core["cycle_reference"]) == {0.5}
    assert set(core["include_outer_fold"]) == {True}
    assert set(no_fold["include_outer_fold"]) == {False}
    assert core["cohort_p_holm"].notna().all()
    assert core["interaction_p_holm"].notna().all()
    assert len(variability) == 8
    assert variability["p_holm"].notna().all()
    sensitivity = pd.read_csv(output / "sensitivity_mixed_models.csv")
    person_medians = pd.read_csv(output / "sensitivity_person_medians.csv")
    assert len(sensitivity) == 16
    assert len(person_medians) == 16
    assert set(sensitivity["source"]) == {"oof_a6", "face"}
    assert set(sensitivity["cycle_reference"]) == {0.5}
    assert set(sensitivity["estimand"]) == {
        "mixed_model_mid_repetition_cohort_effect"
    }
    assert sensitivity["cohort_effect"].gt(0).all()
    phase = pd.read_csv(output / "phase_clusters.csv")
    assert "p_holm_across_metrics" in phase.columns


def test_phase_clusters_are_corrected_across_descriptor_families():
    """Cluster-wise p-values must also account for four descriptor families."""
    rows = [
        {"metric": "theta", "p_value": 0.01},
        {"metric": "theta", "p_value": 0.03},
        {"metric": "omega", "p_value": 0.02},
        {"metric": "tilt", "p_value": 0.40},
    ]

    corrected = adjust_phase_cluster_families(
        rows,
        metrics=("theta", "omega", "tilt", "wrist"),
    )

    frame = pd.DataFrame(corrected)
    assert "p_holm_across_metrics" in frame
    assert frame["p_holm_across_metrics"].ge(frame["p_value"]).all()
    theta = frame.loc[frame["metric"] == "theta"]
    assert theta["p_holm_across_metrics"].nunique() == 1
    assert theta["p_holm_across_metrics"].iloc[0] == pytest.approx(0.04)
