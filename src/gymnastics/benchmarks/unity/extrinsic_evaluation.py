"""Evaluation and reporting for calibrated learned Unity baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .dataset import group_evaluation_sequences
from .evaluation import (
    angular_residual_deg,
    build_reference_sequence,
    evaluate_method_sequence,
    to_evaluation_sequence,
    trunk_rotation_deg,
)
from .extrinsic_training import ExtrinsicRun, FUSION_3D_METHODS
from .mapping import EVALUATION_JOINT_NAMES, UNITY_JOINT_INDICES
from .schema import MethodSequence, UnityBenchmark


METRICS = (
    "mpjpe_mm",
    "median_mm",
    "p95_mm",
    "angle_mae_deg",
    "angle_rmse_deg",
)


def aggregate_extrinsic_results(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Macro-average directions after seed averaging for each method."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    summaries: list[Mapping[str, object]] = []
    expected_folds = {"left_to_right", "right_to_left"}
    expected_seeds = {0, 1, 2}
    for method, chunks in grouped.items():
        by_fold: dict[str, list[Mapping[str, object]]] = {}
        for row in chunks:
            by_fold.setdefault(str(row["fold"]), []).append(row)
        complete = (
            len(chunks) == 6
            and set(by_fold) == expected_folds
            and all(
                len(fold_rows) == 3
                and {int(row["seed"]) for row in fold_rows} == expected_seeds
                for fold_rows in by_fold.values()
            )
        )
        if not complete:
            raise ValueError(f"incomplete 2x3 extrinsic matrix for {method}")
        regimes = {str(row["input_regime"]) for row in chunks}
        if len(regimes) != 1:
            raise ValueError(f"mixed input regimes for {method}")
        summary: dict[str, object] = {
            "method": method,
            "input_regime": regimes.pop(),
            "folds": 2,
            "seeds": 3,
            "runs": 6,
            "training_supervision": "Unity native 3D",
            "camera_geometry": "Unity exact extrinsics",
        }
        for metric in METRICS:
            if not all(metric in row for row in chunks):
                continue
            fold_values = {
                fold: np.asarray(
                    [float(row[metric]) for row in fold_rows],
                    dtype=np.float64,
                )
                for fold, fold_rows in by_fold.items()
            }
            fold_means = np.asarray(
                [np.mean(values) for values in fold_values.values()]
            )
            fold_stds = np.asarray(
                [np.std(values, ddof=1) for values in fold_values.values()]
            )
            all_values = np.concatenate(tuple(fold_values.values()))
            summary[f"macro_{metric}"] = float(np.mean(fold_means))
            summary[f"seed_std_{metric}"] = float(np.mean(fold_stds))
            summary[f"min_{metric}"] = float(np.min(all_values))
            summary[f"max_{metric}"] = float(np.max(all_values))
        summaries.append(MappingProxyType(summary))
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                str(row["input_regime"]),
                float(row["macro_mpjpe_mm"]),
                str(row["method"]),
            ),
        )
    )


def _load_method_sequence(path: Path) -> MethodSequence:
    with np.load(path, allow_pickle=False) as payload:
        return MethodSequence(
            method=str(payload["method"].item()),
            sequence_id=str(payload["sequence_id"].item()),
            sample_ids=np.asarray(payload["sample_ids"], dtype=np.int64),
            points=np.asarray(payload["points"], dtype=np.float32),
            valid=np.asarray(payload["valid"], dtype=bool),
            joint_names=tuple(
                str(value) for value in payload["joint_names"].tolist()
            ),
            metadata=json.loads(str(payload["metadata"].item())),
        )


def _visibility(frames) -> dict[str, np.ndarray]:
    indices = [UNITY_JOINT_INDICES[name] for name in EVALUATION_JOINT_NAMES]
    return {
        camera_id: np.stack(
            [frame.visible[camera_id][indices] for frame in frames]
        )
        for camera_id in ("cam0", "cam1")
    }


def _angle_offset(
    references: Mapping[str, MethodSequence],
    groups,
) -> float:
    residuals: list[np.ndarray] = []
    for sequence_id, reference in references.items():
        angles, valid = trunk_rotation_deg(reference.points, reference.valid)
        actual = np.asarray(
            [frame.actual_angle_deg for frame in groups[sequence_id]],
            dtype=np.float32,
        )
        neutral = valid & (np.abs(actual) <= 1.0)
        if neutral.any():
            residuals.append(
                angular_residual_deg(angles[neutral], actual[neutral])
            )
    if not residuals:
        return 0.0
    radians = np.deg2rad(np.concatenate(residuals))
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * radians)))))


def evaluate_extrinsic_runs(
    benchmark: UnityBenchmark,
    runs: Sequence[ExtrinsicRun],
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Evaluate held-out directions and static OOD artifacts."""
    groups = group_evaluation_sequences(benchmark)
    references = {
        sequence_id: build_reference_sequence(sequence_id, frames)
        for sequence_id, frames in groups.items()
    }
    angle_offset = _angle_offset(references, groups)
    heldout: list[Mapping[str, object]] = []
    static: list[Mapping[str, object]] = []
    for run in runs:
        for sequence_id, split_kind in (
            (run.test_sequence, "heldout_continuous"),
            ("static_sweep", "static_ood"),
        ):
            if sequence_id == run.train_sequence:
                raise ValueError("training direction cannot be evaluated as held-out")
            path = run.run_root / "inference" / f"{sequence_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing extrinsic inference: {path}")
            candidate = to_evaluation_sequence(_load_method_sequence(path))
            frames = groups[sequence_id]
            result = evaluate_method_sequence(
                candidate,
                references[sequence_id],
                visibility=_visibility(frames),
                actual_angles_deg=np.asarray(
                    [frame.actual_angle_deg for frame in frames],
                    dtype=np.float32,
                ),
                angle_offset_deg=angle_offset,
            )
            row = MappingProxyType(
                {
                    **dict(result.summary),
                    "method": run.method,
                    "fold": run.fold,
                    "seed": run.seed,
                    "train_sequence": run.train_sequence,
                    "test_sequence": run.test_sequence,
                    "split_kind": split_kind,
                    "input_regime": (
                        "calibrated_3d_to_3d"
                        if run.method in FUSION_3D_METHODS
                        else "calibrated_2d_to_3d"
                    ),
                }
            )
            (heldout if split_kind == "heldout_continuous" else static).append(
                row
            )
    return tuple(heldout), tuple(static)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def write_extrinsic_report(
    heldout_rows: Sequence[Mapping[str, object]],
    *,
    static_rows: Sequence[Mapping[str, object]],
    output_root: Path,
    provenance: Mapping[str, object],
) -> Path:
    """Write strict aggregate artifacts and a regime-grouped Markdown report."""
    summaries = aggregate_extrinsic_results(heldout_rows)
    by_method = Path(output_root) / "evaluation/by_method.csv"
    _write_csv(by_method, summaries)
    _write_csv(Path(output_root) / "evaluation/run_results.csv", heldout_rows)
    _write_csv(Path(output_root) / "evaluation/static_diagnostics.csv", static_rows)
    payload = {
        "provenance": dict(provenance),
        "by_method": [dict(row) for row in summaries],
        "heldout_runs": [dict(row) for row in heldout_rows],
        "static_diagnostics": [dict(row) for row in static_rows],
    }
    result_path = Path(output_root) / "report/results.json"
    _atomic_json(result_path, payload)

    names = {
        "extrinsic_gate": "Extrinsic learned gate (`extrinsic_gate`)",
        "extrinsic_residual_tcn": (
            "Extrinsic residual TCN (`extrinsic_residual_tcn`)"
        ),
        "learnable_triangulation": (
            "Learnable algebraic triangulation (`learnable_triangulation`)"
        ),
    }
    lines = [
        "# Unity Calibrated Learned Baselines",
        "",
        "Unity native 3D is used for training. All headline rows are evaluated "
        "on the held-out motion direction with two folds and three seeds.",
        "",
    ]
    for regime, title in (
        ("calibrated_3d_to_3d", "Calibrated 3D-to-3D fusion"),
        ("calibrated_2d_to_3d", "Calibrated 2D-to-3D"),
    ):
        selected = sorted(
            (row for row in summaries if row["input_regime"] == regime),
            key=lambda row: float(row["macro_mpjpe_mm"]),
        )
        lines.extend(
            (
                f"## {title}",
                "",
                "| method | MPJPE (mm) | seed SD (mm) | angle MAE (deg) |",
                "|---|---:|---:|---:|",
            )
        )
        for row in selected:
            lines.append(
                f"| {names.get(str(row['method']), str(row['method']))} "
                f"| {float(row['macro_mpjpe_mm']):.3f} "
                f"| {float(row['seed_std_mpjpe_mm']):.3f} "
                f"| {float(row['macro_angle_mae_deg']):.3f} |"
            )
        lines.append("")
    lines.extend(
        (
            "## Interpretation boundary",
            "",
            "The two input regimes are not pooled into one ranking. The benchmark "
            "contains one avatar and one fixed camera pair; direction transfer and "
            "seed dispersion do not establish population-level generalization.",
            "",
        )
    )
    report_path = Path(output_root) / "report/unity_extrinsic_learning_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    return report_path
