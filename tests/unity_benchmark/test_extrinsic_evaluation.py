from __future__ import annotations

from pathlib import Path

import pytest

from gymnastics.benchmarks.unity.extrinsic_evaluation import (
    aggregate_extrinsic_results,
    write_extrinsic_report,
)


def _rows(method: str, base: float) -> list[dict[str, object]]:
    regime = (
        "calibrated_2d_to_3d"
        if method == "learnable_triangulation"
        else "calibrated_3d_to_3d"
    )
    rows = []
    for fold_index, fold in enumerate(("left_to_right", "right_to_left")):
        for seed in (0, 1, 2):
            rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "input_regime": regime,
                    "mpjpe_mm": base + 10.0 * fold_index + seed,
                    "median_mm": base / 2 + seed,
                    "p95_mm": base * 2 + seed,
                    "angle_mae_deg": 20.0 + seed,
                    "angle_rmse_deg": 30.0 + seed,
                }
            )
    return rows


def test_aggregate_extrinsic_results_macro_averages_directions_after_seeds() -> None:
    summary = aggregate_extrinsic_results(_rows("extrinsic_gate", 100.0))
    assert len(summary) == 1
    row = summary[0]
    assert row["method"] == "extrinsic_gate"
    assert row["input_regime"] == "calibrated_3d_to_3d"
    assert row["folds"] == 2
    assert row["seeds"] == 3
    assert row["runs"] == 6
    assert row["macro_mpjpe_mm"] == pytest.approx(106.0)
    assert row["seed_std_mpjpe_mm"] == pytest.approx(1.0)


def test_aggregate_extrinsic_results_rejects_incomplete_matrix() -> None:
    rows = _rows("extrinsic_gate", 100.0)[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_extrinsic_results(rows)


def test_aggregate_keeps_2d_and_3d_input_regimes_separate() -> None:
    summary = aggregate_extrinsic_results(
        _rows("extrinsic_gate", 100.0)
        + _rows("learnable_triangulation", 30.0)
    )
    by_method = {str(row["method"]): row for row in summary}
    assert by_method["extrinsic_gate"]["input_regime"] == "calibrated_3d_to_3d"
    assert (
        by_method["learnable_triangulation"]["input_regime"]
        == "calibrated_2d_to_3d"
    )


def test_write_extrinsic_report_orders_each_regime_by_mpjpe(
    tmp_path: Path,
) -> None:
    rows = (
        _rows("extrinsic_gate", 100.0)
        + _rows("extrinsic_residual_tcn", 90.0)
        + _rows("learnable_triangulation", 30.0)
    )
    report = write_extrinsic_report(
        rows,
        static_rows=(),
        output_root=tmp_path,
        provenance={"protocol": "direction-held-out-2x3"},
    )
    text = report.read_text(encoding="utf-8")
    assert text.index("extrinsic_residual_tcn") < text.index("extrinsic_gate")
    assert "Calibrated 2D-to-3D" in text
    assert (tmp_path / "evaluation/by_method.csv").is_file()
    assert (tmp_path / "report/results.json").is_file()

