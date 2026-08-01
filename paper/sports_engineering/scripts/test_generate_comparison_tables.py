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

import generate_comparison_tables as comparison_tables

from generate_comparison_tables import (
    MAJOR_JOINT_INDICES,
    build_calibration_association,
    build_coordinate_summary,
    build_deterministic_summary,
    build_extrinsic_summary,
    build_joint_summary,
    evaluate_matched_metrics,
    evaluate_matched_joint_metrics,
    load_cached_person_metrics,
    load_all_people,
    load_test_people,
    reevaluate_compact_metrics,
    render_all_joint_table,
    render_deterministic_table,
    render_extrinsic_table,
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


def test_load_all_people_rejects_overlap_between_partitions(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": [str(i) for i in range(96)],
                "val": [str(i) for i in range(96, 123)],
                "test": ["0", *[str(i) for i in range(124, 137)]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="137 unique people"):
        load_all_people(split_path)


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


def test_matched_evaluator_returns_pooled_person_and_joint_metrics() -> None:
    skeleton = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))
    rng = np.random.default_rng(11)
    base_pose = rng.normal(size=(1, 70, 3)).astype("float64") + 2.0
    reference = np.repeat(base_pose, 4, axis=0)
    candidate = reference.copy()

    person, joints = evaluate_matched_metrics(
        "1", "avg_body_current", [(candidate, reference)], skeleton
    )

    assert person["person_id"] == "1"
    assert person["method"] == "avg_body_current"
    assert person["valid_points"] == 4 * 70
    assert person["mpjpe"] == pytest.approx(0.0, abs=1e-10)
    assert person["evaluation_protocol"] == "similarity_plus_hip_centering"
    assert len(joints) == 70
    assert joints["mpjpe"].max() == pytest.approx(0.0, abs=1e-10)


def test_compact_reevaluation_emits_person_and_joint_rows(tmp_path: Path) -> None:
    skeleton = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))
    method_root = tmp_path / "avg_body_current"
    person_root = method_root / "person_1"
    person_root.mkdir(parents=True)
    rng = np.random.default_rng(17)
    reference = rng.normal(size=(2, 70, 3)).astype("float32") + 2.0
    np.savez_compressed(
        person_root / "fused_sequence.npz",
        kpts_world=reference,
        face_map=np.array([0, 1]),
        side_map=np.array([0, 1]),
    )
    cycle_root = tmp_path / "triangulated" / "person_1" / "cycle_000"
    cycle_root.mkdir(parents=True)
    np.savez_compressed(cycle_root / "joints_3d_sequence.npz", joints_3d=reference)
    (cycle_root / "summary.json").write_text(
        json.dumps(
            {
                "face_video_frames": {"start": 0},
                "side_video_frames": {"start": 0},
                "processed_frames": 2,
            }
        ),
        encoding="utf-8",
    )

    people, joints = reevaluate_compact_metrics(
        {"avg_body_current": method_root},
        ("1",),
        tmp_path / "triangulated",
        skeleton,
    )

    assert len(people) == 1
    assert people.iloc[0]["valid_points"] == 2 * 70
    assert people.iloc[0]["evaluation_protocol"] == "similarity_plus_hip_centering"
    assert len(joints) == 70
    assert set(joints["evaluation_protocol"]) == {"similarity_plus_hip_centering"}


def test_cached_person_metrics_require_complete_method_person_coverage(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "person_metrics.csv"
    frame = pd.DataFrame(
        {
            "person_id": ["1", "2", "1", "2"],
            "method": ["a", "a", "b", "b"],
            "mpjpe": [0.1, 0.2, 0.3, 0.4],
            "evaluation_protocol": ["similarity_plus_hip_centering"] * 4,
        }
    )
    frame.to_csv(cache_path, index=False)

    loaded = load_cached_person_metrics(cache_path, ("a", "b"), ("1", "2"))

    assert len(loaded) == 4
    frame.loc[frame["person_id"] != "2"].to_csv(cache_path, index=False)
    with pytest.raises(ValueError, match="complete method-person coverage"):
        load_cached_person_metrics(cache_path, ("a", "b"), ("1", "2"))


def test_build_extrinsic_summary_uses_paired_person_differences() -> None:
    deterministic = pd.DataFrame(
        {
            "person_id": ["1", "2", "3", "4"],
            "method": ["avg_body_current"] * 4,
            "mpjpe": [0.10, 0.20, 0.30, 0.40],
            "evaluation_protocol": ["similarity_plus_hip_centering"] * 4,
        }
    )
    extrinsic = pd.DataFrame(
        {
            "person_id": ["1", "2", "3", "4"] * 2,
            "method": ["extrinsic_r_average"] * 4
            + ["extrinsic_r_quality_average"] * 4,
            "mpjpe": [0.09, 0.19, 0.31, 0.35, 0.11, 0.18, 0.29, 0.45],
            "evaluation_protocol": ["similarity_plus_hip_centering"] * 8,
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


def test_build_extrinsic_summary_rejects_similarity_only_rows() -> None:
    deterministic = pd.DataFrame(
        {
            "person_id": ["1", "2"],
            "method": ["avg_body_current"] * 2,
            "mpjpe": [0.10, 0.20],
            "evaluation_protocol": ["similarity_only"] * 2,
        }
    )
    extrinsic = pd.DataFrame(
        {
            "person_id": ["1", "2"] * 2,
            "method": ["extrinsic_r_average"] * 2
            + ["extrinsic_r_quality_average"] * 2,
            "mpjpe": [0.09, 0.19, 0.11, 0.18],
            "evaluation_protocol": ["similarity_plus_hip_centering"] * 4,
        }
    )

    with pytest.raises(ValueError, match="same evaluation protocol"):
        build_extrinsic_summary(deterministic, extrinsic, bootstrap_repetitions=20)


def test_extrinsic_summaries_separate_heldout_and_all_participants() -> None:
    people = ("1", "2", "3", "4")
    rows = []
    for method_index, method in enumerate(
        (
            "avg_body_current",
            "extrinsic_r_average",
            "extrinsic_r_quality_average",
        )
    ):
        for person_index, person_id in enumerate(people):
            rows.append(
                {
                    "person_id": person_id,
                    "method": method,
                    "mpjpe": 0.10 + 0.01 * method_index + 0.001 * person_index,
                    "evaluation_protocol": "similarity_plus_hip_centering",
                }
            )
    metrics = pd.DataFrame(rows)

    heldout, all_participants = comparison_tables.build_extrinsic_summaries(
        metrics,
        ("2", "4"),
        bootstrap_repetitions=100,
    )

    assert set(heldout["n"]) == {2}
    assert set(all_participants["n"]) == {4}
    with pytest.raises(ValueError, match="test people must be present"):
        comparison_tables.build_extrinsic_summaries(
            metrics,
            ("2", "missing"),
            bootstrap_repetitions=20,
        )


def test_calibration_association_uses_unified_extrinsic_person_errors() -> None:
    person_metrics = pd.DataFrame(
        {
            "person_id": ["1", "2", "3", "4"],
            "method": ["extrinsic_r_average"] * 4,
            "mpjpe": [0.01, 0.02, 0.03, 0.04],
            "evaluation_protocol": ["similarity_plus_hip_centering"] * 4,
        }
    )
    extrinsics = {
        "persons": {
            str(index): {"holdout_reproj_px": float(index)}
            for index in range(1, 5)
        }
    }

    result = build_calibration_association(person_metrics, extrinsics)

    assert result["n"] == 4
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["evaluation_protocol"] == "similarity_plus_hip_centering"


def test_coordinate_summary_uses_same_people_and_protocol() -> None:
    metrics = pd.DataFrame(
        {
            "person_id": ["1", "2", "1", "2"],
            "method": [
                "avg_world_face_ref",
                "avg_world_face_ref",
                "avg_body_current",
                "avg_body_current",
            ],
            "mpjpe": [0.2, 0.4, 0.1, 0.2],
            "evaluation_protocol": ["similarity_plus_hip_centering"] * 4,
        }
    )

    summary = build_coordinate_summary(metrics)

    body = summary.loc[summary["method"] == "avg_body_current"].iloc[0]
    assert body["mean_mm"] == pytest.approx(150.0)
    assert body["reduction_vs_world_pct"] == pytest.approx(50.0)
    assert set(summary["evaluation_protocol"]) == {
        "similarity_plus_hip_centering"
    }


def test_deterministic_summary_and_latex_use_unified_protocol() -> None:
    methods = ("avg_world_face_ref", "avg_body_current", "root_face_stable")
    rows = []
    for method_index, method in enumerate(methods):
        for person_index, person_id in enumerate(("1", "2", "3", "4")):
            rows.append(
                {
                    "person_id": person_id,
                    "method": method,
                    "mpjpe": 0.1 + 0.01 * method_index + 0.001 * person_index,
                    "evaluation_protocol": "similarity_plus_hip_centering",
                }
            )
    summary = build_deterministic_summary(
        pd.DataFrame(rows), methods=methods, bootstrap_repetitions=100
    )
    latex = render_deterministic_table(summary)

    assert len(summary) == 3
    assert summary.iloc[0]["mean_mm"] == pytest.approx(101.5)
    assert set(summary["evaluation_protocol"]) == {
        "similarity_plus_hip_centering"
    }
    assert latex.count("% deterministic-row") == 3
    assert "framewise hip centring" in latex
    assert r"\begin{tabular}{p{0.42\linewidth}rrr}" in latex
    assert "Method & Mean & SD & 95\\% CI" in latex


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


def test_extrinsic_latex_uses_compact_full_width_layout() -> None:
    summary = pd.DataFrame(
        [
            {
                "method": "avg_body_current",
                "n": 14,
                "mean_mm": 64.045,
                "std_mm": 16.092,
                "delta_mm": 0.0,
                "ci_low_mm": np.nan,
                "ci_high_mm": np.nan,
                "p_holm": np.nan,
                "improved_people": 0,
            },
            {
                "method": "extrinsic_r_average",
                "n": 14,
                "mean_mm": 63.074,
                "std_mm": 16.571,
                "delta_mm": -2.175,
                "ci_low_mm": -2.385,
                "ci_high_mm": -1.647,
                "p_holm": 1.24e-14,
                "improved_people": 10,
            },
            {
                "method": "extrinsic_r_quality_average",
                "n": 14,
                "mean_mm": 63.251,
                "std_mm": 16.794,
                "delta_mm": -1.116,
                "ci_low_mm": -1.204,
                "ci_high_mm": -0.395,
                "p_holm": 4.92e-5,
                "improved_people": 9,
            },
        ]
    )

    latex = render_extrinsic_table(summary, scope="heldout")

    assert r"\scriptsize" in latex
    assert r"\setlength{\tabcolsep}{3pt}" in latex
    assert "framewise hip centring" in latex
    assert "same 14 held-out participants used in Table~1" in latex
    assert "10/14" in latex
    assert "/137" not in latex
