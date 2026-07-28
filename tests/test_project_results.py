from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gymnastics.analysis.project_results import (
    generate_project_results,
    holm_adjust,
    load_split_manifest,
    paired_comparisons,
    summarize_classification,
    summarize_learned_by_split,
)


def test_split_aggregation_keeps_generalization_and_validation_diagnostics_separate(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps({"train": ["1", "2"], "val": ["3"], "test": ["4"]}),
        encoding="utf-8",
    )
    rows = [
        {
            "person_id": str(person_id),
            "method": method,
            "mpjpe": str(mpjpe),
            "fixed_corruption_recovery": corruption,
        }
        for method, values in {
            "A0": [(1, 0.10, "nan"), (2, 0.20, "nan"), (3, 0.30, "nan"), (4, 0.40, "nan")],
            "A6": [(1, 0.05, "nan"), (2, 0.10, "nan"), (3, 0.15, "0.60"), (4, 0.20, "nan")],
        }.items()
        for person_id, mpjpe, corruption in values
    ]

    splits = load_split_manifest(manifest_path)
    summary = summarize_learned_by_split(
        rows,
        splits,
        metrics=("mpjpe", "fixed_corruption_recovery"),
    )

    lookup = {
        (row["method"], row["split"], row["metric"]): row for row in summary
    }
    assert lookup[("A6", "test", "mpjpe")] == {
        "method": "A6",
        "split": "test",
        "metric": "mpjpe",
        "n_people": 1,
        "n_measured": 1,
        "mean": pytest.approx(0.20),
        "std": pytest.approx(0.0),
    }
    assert lookup[("A6", "all", "mpjpe")]["n_people"] == 4
    assert lookup[("A6", "all", "mpjpe")]["mean"] == pytest.approx(0.125)
    assert lookup[("A6", "val", "fixed_corruption_recovery")]["n_measured"] == 1
    assert lookup[("A6", "val", "fixed_corruption_recovery")]["mean"] == pytest.approx(
        0.60
    )
    assert lookup[("A6", "test", "fixed_corruption_recovery")]["n_measured"] == 0
    assert np.isnan(lookup[("A6", "test", "fixed_corruption_recovery")]["mean"])


def test_split_manifest_rejects_duplicate_person_membership(tmp_path: Path) -> None:
    path = tmp_path / "split_manifest.json"
    path.write_text(
        json.dumps({"train": ["1"], "val": ["1"], "test": ["2"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="more than one split"):
        load_split_manifest(path)


def test_holm_adjust_preserves_input_order_and_controls_familywise_error() -> None:
    adjusted = holm_adjust([0.03, 0.001, 0.04, 0.20])

    assert adjusted == pytest.approx([0.09, 0.004, 0.09, 0.20])


def test_paired_comparisons_use_only_requested_people_and_report_reproducible_ci() -> None:
    rows = [
        {"person_id": str(pid), "method": method, "mpjpe": str(value)}
        for method, values in {
            "A0": [(1, 0.30), (2, 0.40), (3, 0.50), (4, 0.60)],
            "A5": [(1, 0.20), (2, 0.30), (3, 0.30), (4, 0.40)],
            "A6": [(1, 0.10), (2, 0.20), (3, 0.20), (4, 0.30)],
        }.items()
        for pid, value in values
    ]

    result = paired_comparisons(
        rows,
        reference_method="A6",
        metric="mpjpe",
        person_ids={"2", "3", "4"},
        seed=7,
        bootstrap_samples=2000,
    )

    by_method = {row["method"]: row for row in result}
    assert set(by_method) == {"A0", "A5"}
    assert by_method["A0"]["n_pairs"] == 3
    assert by_method["A0"]["mean_difference"] == pytest.approx(0.2666666667)
    assert by_method["A5"]["mean_difference"] == pytest.approx(0.10)
    assert by_method["A0"]["ci_low"] <= by_method["A0"]["mean_difference"]
    assert by_method["A0"]["ci_high"] >= by_method["A0"]["mean_difference"]
    assert all(0.0 <= row["holm_p"] <= 1.0 for row in result)


def _write_fold_metric(
    root: Path,
    run_name: str,
    fold: int,
    payload: dict[str, float],
) -> Path:
    path = (
        root
        / run_name
        / "2026-05-19"
        / "09-18-17"
        / "metrics"
        / f"fold_{fold}_test_metrics.txt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([payload]), encoding="utf-8")
    return path


def test_classification_summary_aggregates_matching_run_over_person_folds(
    tmp_path: Path,
) -> None:
    paths = [
        _write_fold_metric(
            tmp_path,
            "st_gcn_['posture', 'total']",
            fold,
            {
                "test/acc_posture": acc,
                "test/f1_posture": f1,
                "test/loss": 1.0,
            },
        )
        for fold, acc, f1 in [
            (0, 0.50, 0.40),
            (1, 0.70, 0.60),
            (2, 0.90, 0.80),
        ]
    ]

    summary = summarize_classification(paths)

    by_metric = {row["metric"]: row for row in summary}
    assert by_metric["test/acc_posture"] == {
        "model": "st_gcn",
        "targets": "posture,total",
        "metric": "test/acc_posture",
        "n_folds": 3,
        "mean": pytest.approx(0.70),
        "std": pytest.approx(0.20),
    }
    assert by_metric["test/f1_posture"]["mean"] == pytest.approx(0.60)
    assert "test/loss" not in by_metric


def test_generator_writes_machine_readable_outputs_with_cohort_labels(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps({"train": ["1", "2"], "val": ["3"], "test": ["4"]}),
        encoding="utf-8",
    )
    learned_path = tmp_path / "metrics_by_person.csv"
    learned_path.write_text(
        "\n".join(
            [
                "person_id,method,mpjpe,fixed_corruption_recovery",
                "1,A0,0.30,nan",
                "2,A0,0.40,nan",
                "3,A0,0.50,nan",
                "4,A0,0.60,nan",
                "1,A6,0.10,nan",
                "2,A6,0.20,nan",
                "3,A6,0.30,0.75",
                "4,A6,0.40,nan",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    classification_root = tmp_path / "classification"
    metric_paths = [
        _write_fold_metric(
            classification_root,
            "tcn_['posture', 'relax', 'twist', 'total']",
            fold,
            {"test/acc_total": value, "test/f1_total": value - 0.1},
        )
        for fold, value in [(0, 0.50), (1, 0.60), (2, 0.70)]
    ]
    output_dir = tmp_path / "output"

    outputs = generate_project_results(
        learned_metrics_path=learned_path,
        split_manifest_path=manifest_path,
        classification_metric_paths=metric_paths,
        output_dir=output_dir,
        reference_method="A6",
        bootstrap_samples=500,
    )

    assert set(outputs) == {
        "learned_by_split",
        "learned_test_comparisons",
        "classification_summary",
        "markdown_summary",
    }
    assert all(path.is_file() for path in outputs.values())
    report = outputs["markdown_summary"].read_text(encoding="utf-8")
    assert "held-out test (`N=1`)" in report
    assert "descriptive all-person (`N=4`)" in report
    assert "validation-only (`N=1`)" in report
    assert "not repeated-seed uncertainty" in report
