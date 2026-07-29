from __future__ import annotations

from pathlib import Path

from gymnastics.fusion.rotation_aware.real_camera_evaluation import (
    aggregate_camera_metrics,
    write_real_camera_report,
)


def _rows(*, g4: float, g5: float) -> list[dict[str, object]]:
    values = {"G0": 100.0, "G1": 102.0, "G2": 99.0, "G3": 98.0, "G4": g4, "G5": g5}
    return [
        {
            "person_id": str(person),
            "seed": seed,
            "method": method,
            "mpjpe": value + person * 0.01 + seed * 0.001,
            "cycles": 2,
            "external_valid_points": 1000,
        }
        for person in range(14)
        for seed in range(3)
        for method, value in values.items()
    ]


def test_aggregation_pairs_person_and_seed_before_comparison() -> None:
    summary = aggregate_camera_metrics(_rows(g4=90.0, g5=95.0), bootstrap_samples=200)

    g4 = next(row for row in summary.method_rows if row["method"] == "G4")
    comparison = next(
        row for row in summary.paired_rows
        if row["method"] == "G4" and row["baseline"] == "G0"
    )
    negative_control = next(
        row for row in summary.paired_rows
        if row["method"] == "G4" and row["baseline"] == "G5"
    )

    assert g4["person_seed_pairs"] == 42
    assert comparison["paired_samples"] == 42
    assert comparison["paired_people"] == 14
    assert comparison["mean_delta_mpjpe"] < 0
    assert negative_control["mean_delta_mpjpe"] < 0
    assert summary.camera_claim_supported is True


def test_report_refuses_claim_when_wrong_camera_matches_correct_camera(
    tmp_path: Path,
) -> None:
    summary = aggregate_camera_metrics(_rows(g4=95.0, g5=90.0), bootstrap_samples=200)
    target = write_real_camera_report(
        summary,
        tmp_path / "report.md",
        camera_audit={"people": 137, "median_holdout_reprojection_px": 6.27},
    )
    text = target.read_text(encoding="utf-8")

    assert summary.camera_claim_supported is False
    assert "not supported" in text
    assert "G4" in text and "G5" in text

