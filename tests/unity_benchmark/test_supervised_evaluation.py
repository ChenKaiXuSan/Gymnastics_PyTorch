from __future__ import annotations

import numpy as np
import pytest

from gymnastics.benchmarks.unity.supervised_evaluation import (
    aggregate_finetuned_results,
    evaluate_finetuned_sequence,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def test_finetuned_evaluation_uses_common_sequence_sim3() -> None:
    rng = np.random.default_rng(21)
    reference = rng.normal(size=(4, 16, 3)).astype(np.float32)
    rotation = _rotation_z(np.deg2rad(35.0))
    candidate = (
        1.8 * (reference @ rotation)
        + np.asarray((2.0, -1.0, 0.4), dtype=np.float32)
    )
    visibility = {
        "cam0": np.ones((4, 16), dtype=bool),
        "cam1": np.ones((4, 16), dtype=bool),
    }
    angles = np.zeros((4,), dtype=np.float32)

    exact = evaluate_finetuned_sequence(
        candidate,
        reference,
        visibility=visibility,
        actual_angles_deg=angles,
        fold="left_to_right",
        ablation="A4",
        seed=0,
    )

    assert exact.summary["mpjpe_mm"] < 1e-3
    assert exact.metadata["ranking_group"] == "unity_supervised"

    candidate[1] = candidate[1] @ _rotation_z(np.deg2rad(20.0))
    changed = evaluate_finetuned_sequence(
        candidate,
        reference,
        visibility=visibility,
        actual_angles_deg=angles,
        fold="left_to_right",
        ablation="A4",
        seed=0,
    )
    assert changed.errors_m[1].mean() > 1e-2


def _complete_rows() -> list[dict[str, object]]:
    return [
        {
            "ablation": "A4",
            "fold": "left_to_right",
            "seed": 0,
            "mpjpe_mm": 100.0,
        },
        {
            "ablation": "A4",
            "fold": "left_to_right",
            "seed": 1,
            "mpjpe_mm": 110.0,
        },
        {
            "ablation": "A4",
            "fold": "left_to_right",
            "seed": 2,
            "mpjpe_mm": 120.0,
        },
        {
            "ablation": "A4",
            "fold": "right_to_left",
            "seed": 0,
            "mpjpe_mm": 200.0,
        },
        {
            "ablation": "A4",
            "fold": "right_to_left",
            "seed": 1,
            "mpjpe_mm": 210.0,
        },
        {
            "ablation": "A4",
            "fold": "right_to_left",
            "seed": 2,
            "mpjpe_mm": 220.0,
        },
    ]


def test_aggregation_macro_averages_folds_after_seed_means() -> None:
    summary = aggregate_finetuned_results(_complete_rows())

    assert len(summary) == 1
    row = summary[0]
    assert row["ablation"] == "A4"
    assert row["folds"] == 2
    assert row["seeds"] == 3
    assert row["runs"] == 6
    assert row["macro_mpjpe_mm"] == pytest.approx(160.0)
    assert row["seed_std_mpjpe_mm"] == pytest.approx(10.0)


def test_aggregation_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="incomplete 2x3 matrix"):
        aggregate_finetuned_results(_complete_rows()[:-1])
