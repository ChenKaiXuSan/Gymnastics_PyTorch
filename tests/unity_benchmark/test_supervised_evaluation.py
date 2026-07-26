from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gymnastics.benchmarks.unity.supervised_evaluation import (
    FineTunedRunEvaluation,
    aggregate_finetuned_results,
    build_finetuned_bundle,
    evaluate_finetuned_sequence,
    write_finetuned_report,
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


def test_report_separates_supervised_zero_shot_and_diagnostics(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(31)
    target = rng.normal(size=(4, 16, 3)).astype(np.float32)
    visibility = {
        "cam0": np.ones((4, 16), dtype=bool),
        "cam1": np.ones((4, 16), dtype=bool),
    }
    results = []
    for fold in ("left_to_right", "right_to_left"):
        for seed in (0, 1, 2):
            candidate = target.copy()
            candidate[:, 3, 0] += 0.08 + seed * 0.005
            continuous = evaluate_finetuned_sequence(
                candidate,
                target,
                visibility=visibility,
                actual_angles_deg=np.zeros((4,), dtype=np.float32),
                fold=fold,
                ablation="A4",
                seed=seed,
            )
            static = evaluate_finetuned_sequence(
                candidate + 0.01,
                target,
                visibility=visibility,
                actual_angles_deg=np.zeros((4,), dtype=np.float32),
                fold=fold,
                ablation="A4",
                seed=seed,
            )
            results.extend(
                (
                    FineTunedRunEvaluation(
                        "A4", fold, seed, "heldout_continuous", continuous
                    ),
                    FineTunedRunEvaluation(
                        "A4", fold, seed, "static_ood", static
                    ),
                )
            )
    bundle = build_finetuned_bundle(
        results,
        failures=(),
        provenance={"protocol": "direction-held-out"},
    )
    baseline = {
        "valid_ranking": [
            {
                "method": "triangulation_sam3d2d",
                "mpjpe_mm": 30.259,
                "ranking_group": "valid",
            },
            {
                "method": "avg_world_face_ref",
                "mpjpe_mm": 166.537,
                "ranking_group": "valid",
            },
            {
                "method": "A4",
                "mpjpe_mm": 180.0,
                "ranking_group": "valid",
            },
        ],
        "diagnostics": [
            {
                "method": "triangulation_oracle2d",
                "mpjpe_mm": 0.0002,
                "ranking_group": "diagnostic",
            },
            {
                "method": "sim3_face_stable_joint_weight",
                "mpjpe_mm": 175.0,
                "ranking_group": "diagnostic",
            },
        ],
        "tables": {
            "by_sequence": [
                {
                    "method": "A4",
                    "sequence_id": "continuous_left_060_r00",
                    "mpjpe_mm": 160.0,
                },
                {
                    "method": "A4",
                    "sequence_id": "continuous_right_060_r00",
                    "mpjpe_mm": 180.0,
                },
            ]
        },
    }

    report = write_finetuned_report(
        bundle,
        tmp_path,
        baseline_results=baseline,
    )

    for relative in (
        "evaluation/run_results.csv",
        "evaluation/by_fold.csv",
        "evaluation/by_ablation.csv",
        "evaluation/by_joint.csv",
        "evaluation/by_visibility.csv",
        "report/results.json",
        "report/unity_supervised_finetune_report.md",
        "report/figures/zero_shot_vs_supervised_mpjpe.png",
    ):
        assert (tmp_path / relative).is_file()
    text = report.read_text(encoding="utf-8")
    for heading in (
        "Unity-Supervised Training",
        "Direction-Held-Out Results",
        "Zero-Shot vs Fine-Tuned",
        "Static OOD Diagnostic",
        "Triangulation Comparison",
        "Interpretation Boundary",
    ):
        assert heading in text
    machine = json.loads(
        (tmp_path / "report/results.json").read_text(encoding="utf-8")
    )
    forbidden = {
        "triangulation_oracle2d",
        "sim3_face_stable_joint_weight",
    }
    assert not forbidden & {
        row["method"] for row in machine["zero_shot_ranking"]
    }
    assert not forbidden & {
        row["ablation"] for row in machine["supervised_ranking"]
    }
    assert all(
        row["training_supervision"] == "Unity GT used for training"
        for row in machine["supervised_ranking"]
    )
    assert machine["zero_shot_ranking"][0]["mpjpe_mm"] == pytest.approx(170.0)
    assert (
        machine["zero_shot_ranking"][0]["comparison_scope"]
        == "continuous_direction_macro"
    )
