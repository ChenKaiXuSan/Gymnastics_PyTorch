from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gymnastics.benchmarks.unity.evaluation import (
    EvaluationBundle,
    angular_residual_deg,
    build_reference_sequence,
    evaluate_method_sequence,
    sequence_joint_errors,
    summarize_results,
    to_evaluation_sequence,
)
from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.mapping import EVALUATION_JOINT_NAMES
from gymnastics.benchmarks.unity.report import write_report
from gymnastics.benchmarks.unity.schema import MethodSequence


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float32)


def _pose(frames: int = 2) -> np.ndarray:
    rng = np.random.default_rng(12)
    points = rng.normal(size=(frames, 16, 3)).astype(np.float32)
    points[:, 0] = (0.0, 1.0, 0.0)
    points[:, 2] = (-0.3, 1.4, 0.0)
    points[:, 5] = (0.3, 1.4, 0.0)
    points[:, 8] = (-0.2, 0.9, 0.0)
    points[:, 12] = (0.2, 0.9, 0.0)
    return points


def test_similarity_alignment_is_one_transform_per_sequence() -> None:
    target = _pose()
    rotation = _rotation_z(np.deg2rad(30))
    candidate = 1.7 * (target @ rotation) + np.asarray((2.0, -1.0, 0.4))
    valid = np.ones((2, 16), dtype=bool)

    exact = sequence_joint_errors(candidate, valid, target, valid)

    assert exact.valid.all()
    assert exact.errors_m.max() < 1e-5

    candidate[1] = candidate[1] @ _rotation_z(np.deg2rad(20))
    changed = sequence_joint_errors(candidate, valid, target, valid)
    assert changed.errors_m[1].mean() > 1e-2


def test_angular_residual_wraps_at_180_degrees() -> None:
    residual = angular_residual_deg(
        np.asarray([179.0]), np.asarray([-179.0])
    )
    np.testing.assert_allclose(residual, np.asarray([-2.0]))


def test_evaluation_reports_millimetres_and_report_separates_diagnostics(
    tmp_path: Path,
) -> None:
    target = _pose(frames=3)
    candidate = target.copy()
    candidate[:, 3, 0] += 0.01
    valid = np.ones((3, 16), dtype=bool)
    sample_ids = np.arange(3)
    reference = MethodSequence(
        "unity_gt",
        "sequence",
        sample_ids,
        target,
        valid,
        EVALUATION_JOINT_NAMES,
        {"ranking_group": "reference"},
    )
    method = MethodSequence(
        "cam0",
        "sequence",
        sample_ids,
        candidate,
        valid,
        EVALUATION_JOINT_NAMES,
        {"ranking_group": "valid"},
    )
    diagnostic = MethodSequence(
        "triangulation_oracle2d",
        "sequence",
        sample_ids,
        target,
        valid,
        EVALUATION_JOINT_NAMES,
        {"ranking_group": "diagnostic"},
    )
    visibility = {
        "cam0": np.ones((3, 16), dtype=bool),
        "cam1": np.zeros((3, 16), dtype=bool),
    }

    result = evaluate_method_sequence(
        method,
        reference,
        visibility=visibility,
        actual_angles_deg=np.zeros((3,), dtype=np.float32),
    )
    oracle = evaluate_method_sequence(
        diagnostic,
        reference,
        visibility=visibility,
        actual_angles_deg=np.zeros((3,), dtype=np.float32),
    )
    bundle = summarize_results((result, oracle), failures=())
    report = write_report(bundle, tmp_path, provenance={"commit": "abc"})

    assert result.summary["mpjpe_mm"] > 0
    assert result.summary["mpjpe_mm"] < 10
    assert [row["method"] for row in bundle.valid_ranking] == ["cam0"]
    assert [row["method"] for row in bundle.diagnostics] == [
        "triangulation_oracle2d"
    ]
    assert report.is_file()
    assert (tmp_path / "evaluation/metrics_summary.csv").is_file()
    assert (tmp_path / "report/results.json").is_file()
    text = report.read_text(encoding="utf-8")
    assert "Executive Conclusions" in text
    assert "Valid Method Ranking" in text
    assert "Selected Per-Sequence Results" in text
    assert "Visibility Breakdown" in text
    assert "Diagnostic Methods" in text


def test_converts_mhr70_candidate_and_builds_matching_reference() -> None:
    benchmark = load_unity_benchmark(
        "/home/data/xchen/gymnastics/unity_benchmark"
    )
    frames = benchmark.frames[:5]
    mhr = np.ones((5, 70, 3), dtype=np.float32)
    candidate = MethodSequence(
        "cam0",
        "static_sweep",
        np.asarray([frame.sample_id for frame in frames]),
        mhr,
        np.ones((5, 70), dtype=bool),
        tuple(f"joint_{index}" for index in range(70)),
        {"ranking_group": "valid"},
    )

    converted = to_evaluation_sequence(candidate)
    reference = build_reference_sequence("static_sweep", frames)

    assert converted.points.shape == (5, 16, 3)
    assert converted.joint_names == EVALUATION_JOINT_NAMES
    assert reference.sample_ids.tolist() == converted.sample_ids.tolist()
    assert reference.method == "unity_gt"
