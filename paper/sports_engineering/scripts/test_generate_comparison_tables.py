from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.deterministic.experiment_matrix import joint_errors

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_comparison_tables import (
    MAJOR_JOINT_INDICES,
    build_extrinsic_summary,
    build_joint_summary,
    evaluate_matched_joint_metrics,
    load_test_people,
    render_all_joint_table,
    render_main_joint_table,
)


LEARNED_METHODS = ("A0", "A1", "A2", "A6")
EXTRINSIC_METHODS = (
    "extrinsic_r_average",
    "extrinsic_r_quality_average",
)


def _joint_rows(
    people: tuple[str, ...],
    methods: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for person_id in people:
        person_value = 0.010 if person_id == people[0] else 0.030
        valid_points = 1 if person_id == people[0] else 10_000
        for method_index, method in enumerate(methods):
            for joint in range(70):
                rows.append(
                    {
                        "person_id": person_id,
                        "method": method,
                        "joint": joint,
                        "valid_points": valid_points,
                        "mpjpe": person_value + method_index * 0.001,
                        "evaluation_protocol": "similarity_plus_hip_centering",
                    }
                )
    return pd.DataFrame(rows)


def test_load_test_people_rejects_noncanonical_test_size(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": ["90"], "val": ["91"], "test": [str(i) for i in range(13)]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 14 test people"):
        load_test_people(split_path)


def test_build_joint_summary_averages_people_not_valid_points() -> None:
    people = ("1", "2")
    learned = _joint_rows(people, LEARNED_METHODS)
    extrinsic = _joint_rows(people, EXTRINSIC_METHODS)

    summary = build_joint_summary(learned, extrinsic, people)

    assert len(summary) == 70
    assert summary.loc[summary["joint"] == 0, "A0"].item() == pytest.approx(20.0)
    assert summary.loc[
        summary["joint"] == 0, "extrinsic_r_average"
    ].item() == pytest.approx(20.0)


def test_build_joint_summary_rejects_mixed_evaluation_protocols() -> None:
    people = ("1", "2")
    learned = _joint_rows(people, LEARNED_METHODS)
    extrinsic = _joint_rows(people, EXTRINSIC_METHODS)
    extrinsic["evaluation_protocol"] = "similarity_only"

    with pytest.raises(ValueError, match="same evaluation protocol"):
        build_joint_summary(learned, extrinsic, people)


def test_matched_joint_evaluator_removes_framewise_root_translation() -> None:
    skeleton = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))
    rng = np.random.default_rng(7)
    base_pose = rng.normal(size=(1, 70, 3)).astype("float64") + 2.0
    reference = np.repeat(base_pose, 5, axis=0)
    translation = np.arange(5, dtype="float64")[:, None, None]
    candidate = reference + translation * np.array([0.1, -0.2, 0.05])

    rows = evaluate_matched_joint_metrics(
        "1", "extrinsic_r_average", [(candidate, reference)], skeleton
    )
    similarity_only, valid = joint_errors(candidate, reference, alignment="similarity")

    assert len(rows) == 70
    assert rows["mpjpe"].mean() < similarity_only[valid].mean() * 0.5
    assert set(rows["evaluation_protocol"]) == {"similarity_plus_hip_centering"}


def test_build_extrinsic_summary_uses_paired_person_differences() -> None:
    deterministic = pd.DataFrame(
        {
            "person_id": ["1", "2", "3", "4"],
            "method": ["avg_body_current"] * 4,
            "mpjpe": [0.10, 0.20, 0.30, 0.40],
        }
    )
    extrinsic = pd.DataFrame(
        {
            "person_id": ["1", "2", "3", "4"] * 2,
            "method": ["extrinsic_r_average"] * 4
            + ["extrinsic_r_quality_average"] * 4,
            "mpjpe": [0.09, 0.19, 0.31, 0.35, 0.11, 0.18, 0.29, 0.45],
        }
    )

    summary = build_extrinsic_summary(
        deterministic, extrinsic, bootstrap_repetitions=200
    )
    row = summary.loc[summary["method"] == "extrinsic_r_average"].iloc[0]

    assert row["mean_mm"] == pytest.approx(235.0)
    assert row["delta_mm"] == pytest.approx(-15.0)
    assert row["improved_people"] == 3
    assert row["ci_low_mm"] <= row["delta_mm"] <= row["ci_high_mm"]
    assert 0.0 <= row["p_holm"] <= 1.0


def test_joint_latex_has_expected_rows_and_bolds_row_minimum() -> None:
    rows = []
    for joint in range(70):
        rows.append(
            {
                "joint": joint,
                "joint_name": f"joint_{joint}",
                "A0": 50.0,
                "A1": 45.0,
                "A2": 40.0,
                "A6": 35.0,
                "extrinsic_r_average": 30.0,
                "extrinsic_r_quality_average": 32.0,
            }
        )
    summary = pd.DataFrame(rows)

    main_latex = render_main_joint_table(summary)
    all_latex = render_all_joint_table(summary)

    assert main_latex.count("% joint-row") == len(MAJOR_JOINT_INDICES)
    assert main_latex.count(r"\textbf{") == len(MAJOR_JOINT_INDICES) + 1
    assert r"joint\_0" in main_latex
    assert all_latex.count("% joint-row") == 70
    assert r"\begin{longtable}" in all_latex
    assert "Extrinsic-R quality" in all_latex
