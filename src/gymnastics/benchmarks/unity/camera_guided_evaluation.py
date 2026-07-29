"""Native-GT evaluation and strict aggregation for the Unity G-series."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .camera_guided_training import CameraGuidedRun
from .dataset import group_evaluation_sequences
from .evaluation import (
    build_reference_sequence,
    evaluate_method_sequence,
    to_evaluation_sequence,
    trunk_rotation_deg,
)
from .supervised_evaluation import (
    _angle_offset,
    _load_method_sequence,
    _visibility,
)
from .schema import UnityBenchmark


_METRICS = (
    "mpjpe_mm",
    "median_mm",
    "p95_mm",
    "angle_mae_deg",
    "angle_rmse_deg",
    "rom_error_deg",
    "peak_timing_error_frames",
)


def _complete_by_fold(
    chunks: Sequence[Mapping[str, object]], ablation: str
) -> dict[str, list[Mapping[str, object]]]:
    by_fold: dict[str, list[Mapping[str, object]]] = {}
    for row in chunks:
        by_fold.setdefault(str(row["fold"]), []).append(row)
    complete = (
        set(by_fold) == {"left_to_right", "right_to_left"}
        and len(chunks) == 6
        and all(
            len(rows) == 3
            and {int(row["seed"]) for row in rows} == {0, 1, 2}
            for rows in by_fold.values()
        )
    )
    if not complete:
        raise ValueError(
            f"incomplete 2x3 matrix for camera-guided {ablation}"
        )
    return by_fold


def aggregate_camera_guided_results(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Macro-average direction folds after averaging three seeds."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("split_kind") == "heldout_continuous":
            grouped.setdefault(str(row["ablation"]), []).append(row)
    output: list[Mapping[str, object]] = []
    for ablation, chunks in sorted(grouped.items()):
        by_fold = _complete_by_fold(chunks, ablation)
        summary: dict[str, object] = {
            "ablation": ablation,
            "folds": 2,
            "seeds": 3,
            "runs": 6,
            "ranking_group": "camera_feature_self_supervised",
            "training_supervision": "A6 self-supervision; fitted camera input",
        }
        for metric in _METRICS:
            if not all(metric in row for row in chunks):
                continue
            values_by_fold = {
                fold: np.asarray(
                    [float(row[metric]) for row in fold_rows],
                    dtype=np.float64,
                )
                for fold, fold_rows in by_fold.items()
            }
            if not all(np.isfinite(values).all() for values in values_by_fold.values()):
                continue
            fold_means = np.asarray(
                [values.mean() for values in values_by_fold.values()]
            )
            summary[f"macro_{metric}"] = float(fold_means.mean())
            summary[f"seed_std_{metric}"] = float(
                np.mean(
                    [
                        values.std(ddof=1)
                        for values in values_by_fold.values()
                    ]
                )
            )
        output.append(summary)
    return tuple(output)


def paired_comparisons_vs_g0(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_seed: int = 0,
) -> tuple[Mapping[str, object], ...]:
    """Compare every camera cell with the matched G0 fold/seed cell."""
    continuous = [
        row
        for row in rows
        if row.get("split_kind") == "heldout_continuous"
    ]
    by_method: dict[str, dict[tuple[str, int], Mapping[str, object]]] = {}
    for row in continuous:
        by_method.setdefault(str(row["ablation"]), {})[
            (str(row["fold"]), int(row["seed"]))
        ] = row
    baseline = by_method.get("G0")
    expected = {
        (fold, seed)
        for fold in ("left_to_right", "right_to_left")
        for seed in (0, 1, 2)
    }
    if baseline is None or set(baseline) != expected:
        raise ValueError("paired camera comparison requires complete G0")
    rng = np.random.default_rng(bootstrap_seed)
    output: list[Mapping[str, object]] = []
    for ablation, cells in sorted(by_method.items()):
        if ablation == "G0":
            continue
        if set(cells) != expected:
            raise ValueError(
                f"paired camera comparison requires complete {ablation}"
            )
        keys = sorted(expected)
        deltas = np.asarray(
            [
                float(cells[key]["mpjpe_mm"])
                - float(baseline[key]["mpjpe_mm"])
                for key in keys
            ],
            dtype=np.float64,
        )
        samples = deltas[
            rng.integers(0, len(deltas), size=(10_000, len(deltas)))
        ].mean(axis=1)
        output.append(
            {
                "ablation": ablation,
                "paired_cells": 6,
                "mean_delta_mpjpe_mm": float(deltas.mean()),
                "ci95_low_delta_mpjpe_mm": float(
                    np.percentile(samples, 2.5)
                ),
                "ci95_high_delta_mpjpe_mm": float(
                    np.percentile(samples, 97.5)
                ),
                "improved_cells": int((deltas < 0).sum()),
                "worsened_cells": int((deltas > 0).sum()),
                "comparison": "negative control"
                if ablation == "G5"
                else "camera feature ablation",
            }
        )
    return tuple(output)


def _motion_metrics(
    aligned_points: np.ndarray,
    valid: np.ndarray,
    actual_angles: np.ndarray,
    *,
    angle_offset: float,
) -> tuple[float, float]:
    predicted, predicted_valid = trunk_rotation_deg(aligned_points, valid)
    actual = np.asarray(actual_angles, dtype=np.float64)
    usable = predicted_valid & np.isfinite(predicted) & np.isfinite(actual)
    if usable.sum() < 3:
        return float("nan"), float("nan")
    indices = np.flatnonzero(usable)
    predicted_unwrapped = np.rad2deg(
        np.unwrap(np.deg2rad(predicted[usable] - angle_offset))
    )
    actual_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(actual[usable])))
    rom_error = abs(
        float(np.ptp(predicted_unwrapped))
        - float(np.ptp(actual_unwrapped))
    )
    predicted_centered = predicted_unwrapped - predicted_unwrapped[0]
    actual_centered = actual_unwrapped - actual_unwrapped[0]
    predicted_peak = int(indices[np.argmax(np.abs(predicted_centered))])
    actual_peak = int(indices[np.argmax(np.abs(actual_centered))])
    return rom_error, float(abs(predicted_peak - actual_peak))


def evaluate_camera_guided_runs(
    benchmark: UnityBenchmark,
    runs: Sequence[CameraGuidedRun],
) -> tuple[Mapping[str, object], ...]:
    """Load Unity-native 3D only here, after every supplied run is complete."""
    groups = group_evaluation_sequences(benchmark)
    references = {
        sequence_id: build_reference_sequence(sequence_id, frames)
        for sequence_id, frames in groups.items()
    }
    angle_offset = _angle_offset(references, groups)
    rows: list[Mapping[str, object]] = []
    for run in runs:
        provenance = json.loads(
            run.provenance_path.read_text(encoding="utf-8")
        )
        if provenance.get("unity_native_3d_available_to_training") is not False:
            raise ValueError("camera-guided provenance does not prove GT isolation")
        for sequence_id, split_kind in (
            (run.test_sequence, "heldout_continuous"),
            ("static_sweep", "static_ood"),
        ):
            if sequence_id == run.train_sequence:
                raise ValueError("camera-guided evaluation includes training data")
            candidate = to_evaluation_sequence(
                _load_method_sequence(
                    run.run_root / "inference" / f"{sequence_id}.npz"
                )
            )
            frames = groups[sequence_id]
            actual_angles = np.asarray(
                [frame.actual_angle_deg for frame in frames],
                dtype=np.float32,
            )
            evaluation = evaluate_method_sequence(
                candidate,
                references[sequence_id],
                visibility=_visibility(frames),
                actual_angles_deg=actual_angles,
                angle_offset_deg=angle_offset,
            )
            rom_error, peak_error = _motion_metrics(
                evaluation.aligned_points_m,
                evaluation.valid,
                actual_angles,
                angle_offset=angle_offset,
            )
            fitted = provenance.get("fitted_camera")
            rows.append(
                {
                    "ablation": run.ablation,
                    "fold": run.fold,
                    "seed": run.seed,
                    "sequence_id": sequence_id,
                    "split_kind": split_kind,
                    **dict(evaluation.summary),
                    "rom_error_deg": rom_error,
                    "peak_timing_error_frames": peak_error,
                    "camera_inlier_ratio": (
                        float(fitted["inlier_ratio"])
                        if isinstance(fitted, Mapping)
                        else float("nan")
                    ),
                    "camera_holdout_reprojection_px": (
                        float(fitted["holdout_reprojection_px"])
                        if isinstance(fitted, Mapping)
                        else float("nan")
                    ),
                    "unity_gt_used_for_training": False,
                }
            )
    return tuple(rows)


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_camera_guided_report(
    output_root: Path,
    *,
    run_rows: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
) -> Mapping[str, Path]:
    """Write complete machine-readable tables and a concise ranked report."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    by_method = tuple(
        sorted(
            aggregate_camera_guided_results(run_rows),
            key=lambda row: float(row["macro_mpjpe_mm"]),
        )
    )
    comparisons = paired_comparisons_vs_g0(run_rows)
    run_path = output_root / "metrics_by_sequence.csv"
    method_path = output_root / "by_method.csv"
    comparison_path = output_root / "comparisons_vs_g0.csv"
    _atomic_csv(run_path, run_rows)
    _atomic_csv(method_path, by_method)
    _atomic_csv(comparison_path, comparisons)
    report_path = output_root / "camera_feature_report.md"
    lines = [
        "# Fitted-camera feature fusion on Unity",
        "",
        "Unity-native 3D is loaded only by this evaluation stage.",
        "",
        "## Held-out continuous ranking",
        "",
        "| Rank | Method | MPJPE (mm) | Angle MAE (deg) |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(by_method, start=1):
        lines.append(
            f"| {rank} | {row['ablation']} | "
            f"{float(row['macro_mpjpe_mm']):.3f} | "
            f"{float(row.get('macro_angle_mae_deg', float('nan'))):.3f} |"
        )
    lines.extend(
        (
            "",
            "## Paired comparisons versus G0",
            "",
            "| Method | Delta MPJPE (mm) | 95% descriptive CI | Improved cells | Type |",
            "|---|---:|---:|---:|---|",
        )
    )
    for row in comparisons:
        lines.append(
            f"| {row['ablation']} | "
            f"{float(row['mean_delta_mpjpe_mm']):.3f} | "
            f"[{float(row['ci95_low_delta_mpjpe_mm']):.3f}, "
            f"{float(row['ci95_high_delta_mpjpe_mm']):.3f}] | "
            f"{row['improved_cells']}/6 | {row['comparison']} |"
        )
    lines.extend(
        (
            "",
            "The G5 row is the preregistered wrong-camera negative control. "
            "Intervals resample six fold/seed cells and are descriptive because "
            "Unity contains one avatar and one physical rig.",
            "",
        )
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "metrics_by_sequence": run_path,
        "by_method": method_path,
        "comparisons_vs_g0": comparison_path,
        "report": report_path,
        "provenance": provenance_path,
    }
