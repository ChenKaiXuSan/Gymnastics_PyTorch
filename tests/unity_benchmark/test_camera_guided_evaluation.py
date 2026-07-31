from __future__ import annotations

from pathlib import Path

import pytest

from gymnastics.benchmarks.unity.camera_guided_evaluation import (
    aggregate_camera_guided_results,
    paired_comparisons_vs_g0,
    write_camera_guided_report,
)


def _complete_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ablation, offset in (("G0", 0.0), ("G4", -10.0), ("G5", 5.0)):
        for fold, base in (
            ("left_to_right", 100.0),
            ("right_to_left", 200.0),
        ):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "ablation": ablation,
                        "fold": fold,
                        "seed": seed,
                        "split_kind": "heldout_continuous",
                        "mpjpe_mm": base + seed + offset,
                        "angle_mae_deg": 20.0 + 0.1 * seed + offset / 10,
                        "rom_error_deg": 5.0 + seed,
                        "peak_timing_error_frames": 2.0 + seed,
                    }
                )
    return rows


def test_camera_guided_aggregation_averages_seeds_then_folds() -> None:
    summary = aggregate_camera_guided_results(_complete_rows())
    by_ablation = {row["ablation"]: row for row in summary}

    assert by_ablation["G0"]["macro_mpjpe_mm"] == pytest.approx(151.0)
    assert by_ablation["G4"]["macro_mpjpe_mm"] == pytest.approx(141.0)
    assert by_ablation["G5"]["macro_mpjpe_mm"] == pytest.approx(156.0)
    assert by_ablation["G4"]["runs"] == 6


def test_camera_guided_aggregation_rejects_incomplete_matrix() -> None:
    with pytest.raises(ValueError, match="incomplete 2x3 matrix"):
        aggregate_camera_guided_results(_complete_rows()[:-1])


def test_paired_comparison_reports_negative_control_against_g0() -> None:
    comparisons = paired_comparisons_vs_g0(_complete_rows(), bootstrap_seed=3)
    by_ablation = {row["ablation"]: row for row in comparisons}

    assert by_ablation["G4"]["mean_delta_mpjpe_mm"] == pytest.approx(-10.0)
    assert by_ablation["G5"]["mean_delta_mpjpe_mm"] == pytest.approx(5.0)
    assert by_ablation["G4"]["improved_cells"] == 6
    assert by_ablation["G5"]["improved_cells"] == 0
    assert by_ablation["G4"]["ci95_high_delta_mpjpe_mm"] < 0


def test_report_writes_ranked_markdown_and_csv_tables(tmp_path: Path) -> None:
    outputs = write_camera_guided_report(
        tmp_path,
        run_rows=_complete_rows(),
        provenance={"matrix": "fixture"},
    )

    report = outputs["report"].read_text(encoding="utf-8")
    assert "G4" in report
    assert "negative control" in report
    assert "supports a correct-camera geometry claim" in report
    assert outputs["by_method"].is_file()
    assert outputs["comparisons_vs_g0"].is_file()
