from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gymnastics.benchmarks.freeman.evaluation import (
    SessionMetrics,
    aggregate_metrics,
    evaluate_session,
    paired_method_tests,
)
from gymnastics.benchmarks.freeman.mapping import FREEMAN_COCO17_NAMES
from gymnastics.benchmarks.freeman.report import (
    ReportContext,
    write_report,
)
from gymnastics.benchmarks.freeman.schema import MethodPrediction, ReferenceSequence
from gymnastics.common.skeletons.mhr70 import MHR70_INDEX


def _reference(frames: int = 4) -> ReferenceSequence:
    rng = np.random.default_rng(101)
    points = rng.normal(scale=0.2, size=(frames, 17, 3)).astype(np.float32)
    points[..., 2] += 1.5
    return ReferenceSequence(
        session_id="eval_subj01",
        subject_id=1,
        fps=30,
        split="test",
        scenario="park",
        action=None,
        reference_scale_to_m=1.0,
        points_m=points,
        valid=np.ones((frames, 17), dtype=bool),
        frame_ids=np.arange(frames),
        joint_names=FREEMAN_COCO17_NAMES,
    )


def _prediction_from_coco17(
    points: np.ndarray,
    *,
    frame_ids: np.ndarray | None = None,
    method: str = "candidate",
) -> MethodPrediction:
    frames = points.shape[0]
    mhr = np.ones((frames, 70, 3), dtype=np.float32)
    valid = np.ones((frames, 70), dtype=bool)
    for index, name in enumerate(FREEMAN_COCO17_NAMES):
        mhr[:, MHR70_INDEX[name]] = points[:, index]
    return MethodPrediction(
        method=method,
        session_id="eval_subj01",
        subject_id=1,
        fps=30,
        points=mhr,
        valid=valid,
        frame_ids=np.arange(frames) if frame_ids is None else frame_ids,
        metadata={"classification": "VALID"},
    )


def test_sequence_sim3_removes_one_static_coordinate_transform() -> None:
    reference = _reference()
    transformed = reference.points_m * 1.7 + np.array([0.5, -0.3, 2.0])
    prediction = _prediction_from_coco17(transformed)

    metrics = evaluate_session(prediction, reference, thresholds_mm=(50, 100))

    assert metrics.sim3_mpjpe_mm < 1e-3
    assert metrics.median_mpjpe_mm < 1e-3
    assert metrics.p95_mpjpe_mm < 1e-3
    assert metrics.max_mpjpe_mm < 1e-3
    assert metrics.root_mpjpe_mm < 1e-3
    assert metrics.velocity_error_mm_s < 1e-2
    assert metrics.acceleration_error_mm_s2 < 1.0
    assert metrics.pck[50] == pytest.approx(1.0)
    assert metrics.auc == pytest.approx(1.0, abs=1e-3)


def test_sequence_alignment_preserves_frame_dependent_rotation_error() -> None:
    reference = _reference(frames=3)
    angles = np.deg2rad([0.0, 35.0, -35.0])
    rotated = np.empty_like(reference.points_m)
    for frame, angle in enumerate(angles):
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated[frame] = reference.points_m[frame] @ rotation
    prediction = _prediction_from_coco17(rotated)

    metrics = evaluate_session(prediction, reference, thresholds_mm=(50, 100))

    assert metrics.sim3_mpjpe_mm > 20.0
    assert metrics.pa_mpjpe_mm < 1e-3


def test_evaluation_rejects_frame_ids_missing_from_reference() -> None:
    reference = _reference(frames=3)
    prediction = _prediction_from_coco17(
        np.array(reference.points_m, copy=True),
        frame_ids=np.array([0, 1, 99]),
    )

    with pytest.raises(ValueError, match="reference frame IDs"):
        evaluate_session(prediction, reference, thresholds_mm=(50,))


def _session_metric(
    *,
    subject: int,
    session: str,
    method: str,
    mpjpe: float,
    classification: str = "VALID",
) -> SessionMetrics:
    return SessionMetrics(
        subject_id=subject,
        session_id=session,
        fps=30,
        split="test",
        scenario="park",
        action=None,
        method=method,
        classification=classification,
        frames_total=10,
        frames_valid=10,
        valid_points=170,
        sim3_mpjpe_mm=mpjpe,
        median_mpjpe_mm=mpjpe,
        p95_mpjpe_mm=mpjpe,
        max_mpjpe_mm=mpjpe,
        root_mpjpe_mm=mpjpe + 1,
        pa_mpjpe_mm=mpjpe - 1,
        velocity_error_mm_s=2.0,
        acceleration_error_mm_s2=3.0,
        pck={50: 0.5, 100: 0.9},
        auc=0.7,
        coverage=1.0,
        per_joint_mpjpe_mm=tuple([mpjpe] * 17),
    )


def test_aggregation_uses_subject_means_not_session_pooling() -> None:
    rows = [
        _session_metric(subject=1, session="long", method="candidate", mpjpe=0.0),
        _session_metric(subject=1, session="short", method="candidate", mpjpe=100.0),
        _session_metric(subject=2, session="only", method="candidate", mpjpe=200.0),
    ]

    tables = aggregate_metrics(rows)

    by_subject = tables.by_subject.sort_values("subject_id")
    np.testing.assert_allclose(by_subject["sim3_mpjpe_mm"], [50.0, 200.0])
    assert tables.by_method.iloc[0]["sim3_mpjpe_mm"] == pytest.approx(125.0)
    assert set(tables.by_session["scenario"]) == {"park"}
    assert len(tables.by_joint) == 3 * 17


def test_paired_tests_match_subjects_and_apply_holm_correction() -> None:
    rows = []
    for subject in range(1, 13):
        rows.extend(
            [
                {
                    "subject_id": subject,
                    "method": "view_a",
                    "classification": "VALID",
                    "sim3_mpjpe_mm": 100.0 + subject,
                },
                {
                    "subject_id": subject,
                    "method": "candidate",
                    "classification": "VALID",
                    "sim3_mpjpe_mm": 80.0 + subject,
                },
            ]
        )

    result = paired_method_tests(
        pd.DataFrame(rows),
        seed=20260726,
        bootstrap_samples=500,
    )

    candidate = result[
        (result["method"] == "candidate") & (result["baseline"] == "view_a")
    ].iloc[0]
    assert candidate["matched_subjects"] == 12
    assert candidate["mean_difference_mm"] == pytest.approx(-20.0)
    assert candidate["holm_p_value"] >= candidate["p_value"]
    assert candidate["status"] == "measured"


def _report_tables():
    rows = []
    for subject in range(1, 41):
        rows.extend(
            [
                _session_metric(
                    subject=subject,
                    session=f"session_{subject:02d}",
                    method="view_a",
                    mpjpe=100.0 + subject,
                ),
                _session_metric(
                    subject=subject,
                    session=f"session_{subject:02d}",
                    method="candidate",
                    mpjpe=80.0 + subject,
                ),
                _session_metric(
                    subject=subject,
                    session=f"session_{subject:02d}",
                    method="sim3_face_stable_joint_weight",
                    mpjpe=70.0 + subject,
                    classification="GT_LEAKY_DIAGNOSTIC",
                ),
            ]
        )
    return aggregate_metrics(rows)


def _report_context() -> ReportContext:
    return ReportContext(
        resolved_config={
            "repository": {"repo_id": "wjwow/FreeMan", "revision": "main"},
            "dataset": {
                "subjects": list(range(1, 41)),
                "fps_subsets": [30, 60],
                "reference_scale_to_m": 1.0,
            },
            "evaluation": {"minimum_subject_coverage": 0.95},
        },
        dataset_manifest={
            "processed_subjects": list(range(1, 41)),
            "processed_sessions": 40,
            "fps_session_counts": {"30": 40, "60": 0},
        },
        download_manifest={"inventory_sha256": "inventory-hash"},
        camera_pairs=pd.DataFrame(
            [
                {
                    "subject_id": subject,
                    "session_id": f"session_{subject:02d}",
                    "fps": 30,
                    "view_a": "c01",
                    "view_b": "c05",
                    "separation_deg": 89.5,
                    "target_error_deg": 0.5,
                }
                for subject in range(1, 41)
            ]
        ),
        checkpoint_metadata={
            "sam3d": {"checkpoint_id": "sam3d-body"},
            "rotation_aware": {"paper_a6": {"sha256": "checkpoint-hash"}},
        },
        code_commit="0123456789abcdef",
    )


def test_report_writes_machine_readable_outputs_and_separates_diagnostics(
    tmp_path,
) -> None:
    outputs = write_report(
        _report_tables(),
        _report_context(),
        tmp_path,
    )

    expected_csvs = {
        "metrics_by_session",
        "metrics_by_subject",
        "metrics_by_method",
        "metrics_by_joint",
        "metrics_by_split",
        "metrics_by_scenario",
        "paired_statistics",
        "failures",
        "camera_pairs",
    }
    assert set(outputs.csv_paths) == expected_csvs
    assert all(path.is_file() for path in outputs.csv_paths.values())
    assert outputs.results_json.is_file()
    text = outputs.markdown.read_text(encoding="utf-8")
    assert "public markerless multi-view reference" in text
    assert "independent marker-based motion capture" in text
    assert "sim3_face_stable_joint_weight" in text
    assert "excluded from valid ranking" in text
    assert "all 40 subjects" in text
    assert "Complete: yes" in text

    raw_json = outputs.results_json.read_text(encoding="utf-8")
    assert "NaN" not in raw_json
    payload = __import__("json").loads(raw_json)
    classifications = payload["method_classification"]
    assert classifications["candidate"] == "VALID"
    assert (
        classifications["sim3_face_stable_joint_weight"]
        == "GT_LEAKY_DIAGNOSTIC"
    )


def test_report_refuses_complete_label_below_subject_coverage(tmp_path) -> None:
    context = _report_context()
    incomplete = ReportContext(
        resolved_config=context.resolved_config,
        dataset_manifest={
            **context.dataset_manifest,
            "processed_subjects": list(range(1, 21)),
        },
        download_manifest=context.download_manifest,
        camera_pairs=context.camera_pairs.iloc[:20],
        checkpoint_metadata=context.checkpoint_metadata,
        code_commit=context.code_commit,
    )

    outputs = write_report(_report_tables(), incomplete, tmp_path)

    text = outputs.markdown.read_text(encoding="utf-8")
    assert "Complete: no" in text
    assert "all 40 subjects" not in text


def test_report_requires_evaluated_subjects_not_only_manifest_claim(tmp_path) -> None:
    rows = [
        _session_metric(
            subject=subject,
            session=f"session_{subject:02d}",
            method="candidate",
            mpjpe=80.0 + subject,
        )
        for subject in range(1, 21)
    ]

    outputs = write_report(
        aggregate_metrics(rows),
        _report_context(),
        tmp_path,
    )

    assert "Complete: no" in outputs.markdown.read_text(encoding="utf-8")
