"""Common Unity evaluation and strict fold/seed aggregation for fine-tuning."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .dataset import group_evaluation_sequences
from .evaluation import (
    EvaluationResult,
    angular_residual_deg,
    build_reference_sequence,
    evaluate_method_sequence,
    to_evaluation_sequence,
    trunk_rotation_deg,
)
from .mapping import EVALUATION_JOINT_NAMES, UNITY_JOINT_INDICES
from .schema import MethodSequence, UnityBenchmark
from .supervised import UnityFineTuneRun
from .supervised_data import UnitySupervisedSequence


@dataclass(frozen=True)
class FineTunedRunEvaluation:
    ablation: str
    fold: str
    seed: int
    split_kind: str
    evaluation: EvaluationResult


@dataclass(frozen=True)
class FineTunedEvaluationBundle:
    run_results: tuple[FineTunedRunEvaluation, ...]
    failures: tuple[Mapping[str, object], ...]
    tables: Mapping[str, tuple[Mapping[str, object], ...]]
    supervised_ranking: tuple[Mapping[str, object], ...]
    static_diagnostics: tuple[Mapping[str, object], ...]
    provenance: Mapping[str, object]


def evaluate_finetuned_sequence(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    visibility: Mapping[str, np.ndarray],
    actual_angles_deg: np.ndarray,
    fold: str,
    ablation: str,
    seed: int,
) -> EvaluationResult:
    """Evaluate a tensor pair through the shared one-Sim3 sequence evaluator."""
    prediction = np.asarray(candidate, dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("candidate and reference must have shape [T,16,3]")
    if prediction.shape[1:] != (16, 3):
        raise ValueError("fine-tuned evaluation requires Unity16 joints")
    sample_ids = np.arange(len(prediction), dtype=np.int64)
    prediction_valid = np.isfinite(prediction).all(axis=-1)
    target_valid = np.isfinite(target).all(axis=-1)
    candidate_sequence = MethodSequence(
        method=ablation,
        sequence_id="synthetic",
        sample_ids=sample_ids,
        points=prediction,
        valid=prediction_valid,
        joint_names=EVALUATION_JOINT_NAMES,
        metadata={
            "ranking_group": "unity_supervised",
            "fold": fold,
            "seed": seed,
            "ablation": ablation,
        },
    )
    reference_sequence = MethodSequence(
        method="unity_gt",
        sequence_id="synthetic",
        sample_ids=sample_ids,
        points=target,
        valid=target_valid,
        joint_names=EVALUATION_JOINT_NAMES,
        metadata={"ranking_group": "reference"},
    )
    return evaluate_method_sequence(
        candidate_sequence,
        reference_sequence,
        visibility=visibility,
        actual_angles_deg=actual_angles_deg,
    )


def _metric_names(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    preferred = (
        "mpjpe_mm",
        "median_mm",
        "p95_mm",
        "angle_mae_deg",
        "angle_rmse_deg",
    )
    return tuple(
        name
        for name in preferred
        if all(name in row for row in rows)
    )


def aggregate_finetuned_results(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Macro-average directions after seed averaging for each ablation."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ablation"]), []).append(row)
    summaries: list[Mapping[str, object]] = []
    expected_folds = {"left_to_right", "right_to_left"}
    expected_seeds = {0, 1, 2}
    for ablation, chunks in sorted(grouped.items()):
        by_fold: dict[str, list[Mapping[str, object]]] = {}
        for row in chunks:
            by_fold.setdefault(str(row["fold"]), []).append(row)
        complete = set(by_fold) == expected_folds and all(
            {int(row["seed"]) for row in fold_rows} == expected_seeds
            and len(fold_rows) == 3
            for fold_rows in by_fold.values()
        )
        if not complete or len(chunks) != 6:
            raise ValueError(
                f"incomplete 2x3 matrix for Unity supervised {ablation}"
            )
        summary: dict[str, object] = {
            "ablation": ablation,
            "folds": 2,
            "seeds": 3,
            "runs": 6,
            "ranking_group": "unity_supervised",
            "training_supervision": "Unity GT used for training",
        }
        for metric in _metric_names(chunks):
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
            fold_seed_stds = np.asarray(
                [np.std(values, ddof=1) for values in fold_values.values()]
            )
            all_values = np.concatenate(tuple(fold_values.values()))
            summary[f"macro_{metric}"] = float(np.mean(fold_means))
            summary[f"seed_std_{metric}"] = float(np.mean(fold_seed_stds))
            summary[f"min_{metric}"] = float(np.min(all_values))
            summary[f"max_{metric}"] = float(np.max(all_values))
        summaries.append(MappingProxyType(summary))
    return tuple(summaries)


def _load_method_sequence(path: Path) -> MethodSequence:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        return MethodSequence(
            method=str(payload["method"].item()),
            sequence_id=str(payload["sequence_id"].item()),
            sample_ids=np.asarray(payload["sample_ids"], dtype=np.int64),
            points=np.asarray(payload["points"], dtype=np.float32),
            valid=np.asarray(payload["valid"], dtype=bool),
            joint_names=tuple(
                str(value) for value in payload["joint_names"].tolist()
            ),
            metadata=metadata,
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


def evaluate_finetuned_runs(
    benchmark: UnityBenchmark,
    runs: Sequence[UnityFineTuneRun],
    sequences: Mapping[str, UnitySupervisedSequence],
) -> tuple[FineTunedRunEvaluation, ...]:
    """Evaluate each run on held-out continuous and static OOD only."""
    groups = group_evaluation_sequences(benchmark)
    references = {
        sequence_id: build_reference_sequence(sequence_id, frames)
        for sequence_id, frames in groups.items()
    }
    angle_offset = _angle_offset(references, groups)
    results: list[FineTunedRunEvaluation] = []
    for run in runs:
        for sequence_id, split_kind in (
            (run.test_sequence, "heldout_continuous"),
            ("static_sweep", "static_ood"),
        ):
            if sequence_id == run.train_sequence:
                raise ValueError("training sequence cannot be evaluated as held-out")
            path = run.run_root / "inference" / f"{sequence_id}.npz"
            candidate = to_evaluation_sequence(_load_method_sequence(path))
            if not np.array_equal(
                candidate.sample_ids, sequences[sequence_id].sample_ids
            ):
                raise ValueError("fine-tuned inference sample identity mismatch")
            frames = groups[sequence_id]
            evaluation = evaluate_method_sequence(
                candidate,
                references[sequence_id],
                visibility=_visibility(frames),
                actual_angles_deg=np.asarray(
                    [frame.actual_angle_deg for frame in frames],
                    dtype=np.float32,
                ),
                angle_offset_deg=angle_offset,
            )
            results.append(
                FineTunedRunEvaluation(
                    ablation=run.ablation,
                    fold=run.fold,
                    seed=run.seed,
                    split_kind=split_kind,
                    evaluation=evaluation,
                )
            )
    return tuple(results)


def _run_row(result: FineTunedRunEvaluation) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "ablation": result.ablation,
            "fold": result.fold,
            "seed": result.seed,
            "split_kind": result.split_kind,
            **dict(result.evaluation.summary),
            "source_checkpoint_sha256": result.evaluation.metadata.get(
                "source_checkpoint_sha256", ""
            ),
            "final_checkpoint_sha256": result.evaluation.metadata.get(
                "final_checkpoint_sha256", ""
            ),
        }
    )


def _by_fold_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["ablation"]), str(row["fold"])), []
        ).append(row)
    output: list[Mapping[str, object]] = []
    for (ablation, fold), chunks in sorted(grouped.items()):
        row: dict[str, object] = {
            "ablation": ablation,
            "fold": fold,
            "seeds": len(chunks),
        }
        for metric in _metric_names(chunks):
            values = np.asarray([float(item[metric]) for item in chunks])
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1))
        output.append(MappingProxyType(row))
    return tuple(output)


def _static_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ablation"]), []).append(row)
    output: list[Mapping[str, object]] = []
    for ablation, chunks in sorted(grouped.items()):
        row: dict[str, object] = {
            "ablation": ablation,
            "runs": len(chunks),
            "ranking_group": "static_ood_diagnostic",
        }
        for metric in _metric_names(chunks):
            values = np.asarray([float(item[metric]) for item in chunks])
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1))
            row[f"min_{metric}"] = float(np.min(values))
            row[f"max_{metric}"] = float(np.max(values))
        output.append(MappingProxyType(row))
    return tuple(output)


def build_finetuned_bundle(
    results: Sequence[FineTunedRunEvaluation],
    *,
    failures: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
) -> FineTunedEvaluationBundle:
    run_rows = tuple(_run_row(result) for result in results)
    continuous_rows = tuple(
        row
        for row in run_rows
        if row["split_kind"] == "heldout_continuous"
    )
    static_rows = tuple(
        row for row in run_rows if row["split_kind"] == "static_ood"
    )
    ranking = tuple(
        sorted(
            aggregate_finetuned_results(continuous_rows),
            key=lambda row: float(row["macro_mpjpe_mm"]),
        )
    )
    by_joint = tuple(
        MappingProxyType(
            {
                "ablation": result.ablation,
                "fold": result.fold,
                "seed": result.seed,
                "split_kind": result.split_kind,
                **dict(row),
            }
        )
        for result in results
        for row in result.evaluation.joint_rows
    )
    by_visibility = tuple(
        MappingProxyType(
            {
                "ablation": result.ablation,
                "fold": result.fold,
                "seed": result.seed,
                "split_kind": result.split_kind,
                **dict(row),
            }
        )
        for result in results
        for row in result.evaluation.visibility_rows
    )
    static_diagnostics = _static_rows(static_rows)
    tables = MappingProxyType(
        {
            "run_results": run_rows,
            "by_fold": _by_fold_rows(continuous_rows),
            "by_ablation": ranking,
            "by_sequence": run_rows,
            "by_joint": by_joint,
            "by_visibility": by_visibility,
            "static_diagnostics": static_diagnostics,
        }
    )
    return FineTunedEvaluationBundle(
        run_results=tuple(results),
        failures=tuple(
            MappingProxyType(dict(failure)) for failure in failures
        ),
        tables=tables,
        supervised_ranking=ranking,
        static_diagnostics=static_diagnostics,
        provenance=MappingProxyType(dict(provenance)),
    )


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(_plain(rows))
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _markdown_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> list[str]:
    if not rows:
        return ["_No results._"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _baseline_rows(
    baseline_results: Mapping[str, object],
    name: str,
) -> tuple[Mapping[str, object], ...]:
    rows = baseline_results.get(name, ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"baseline {name} must be a sequence")
    return tuple(
        row for row in rows if isinstance(row, Mapping)
    )


def _protocol_matched_zero_shot(
    rows: Sequence[Mapping[str, object]],
    baseline_results: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    tables = baseline_results.get("tables", {})
    by_sequence = (
        tables.get("by_sequence", ())
        if isinstance(tables, Mapping)
        else ()
    )
    sequence_rows = (
        tuple(row for row in by_sequence if isinstance(row, Mapping))
        if isinstance(by_sequence, Sequence)
        and not isinstance(by_sequence, (str, bytes))
        else ()
    )
    continuous_ids = {
        "continuous_left_060_r00",
        "continuous_right_060_r00",
    }
    output: list[Mapping[str, object]] = []
    for row in rows:
        method = str(row["method"])
        matching = [
            item
            for item in sequence_rows
            if str(item.get("method", "")) == method
            and str(item.get("sequence_id", "")) in continuous_ids
        ]
        normalized = dict(row)
        if {
            str(item["sequence_id"]) for item in matching
        } == continuous_ids and len(matching) == 2:
            for metric in (
                "mpjpe_mm",
                "median_mm",
                "p95_mm",
                "angle_mae_deg",
                "angle_rmse_deg",
            ):
                if all(metric in item for item in matching):
                    normalized[metric] = float(
                        np.mean([float(item[metric]) for item in matching])
                    )
            normalized["comparison_scope"] = "continuous_direction_macro"
        else:
            normalized["comparison_scope"] = "benchmark_overall_fallback"
        output.append(MappingProxyType(normalized))
    return tuple(
        sorted(output, key=lambda item: float(item["mpjpe_mm"]))
    )


def write_finetuned_report(
    bundle: FineTunedEvaluationBundle,
    output_root: Path,
    *,
    baseline_results: Mapping[str, object],
) -> Path:
    """Publish strict supervised and zero-shot rankings without leakage mixing."""
    output_root = Path(output_root)
    evaluation_root = output_root / "evaluation"
    report_root = output_root / "report"
    figures_root = report_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_results",
        "by_fold",
        "by_ablation",
        "by_joint",
        "by_visibility",
    ):
        _atomic_csv(
            evaluation_root / f"{name}.csv",
            bundle.tables[name],
        )
    _atomic_csv(
        evaluation_root / "static_diagnostics.csv",
        bundle.tables["static_diagnostics"],
    )
    per_frame = {
        (
            f"{result.ablation}__{result.fold}__seed{result.seed}"
            f"__{result.split_kind}"
        ): result.evaluation.errors_m
        for result in bundle.run_results
    }
    np.savez_compressed(
        evaluation_root / "per_frame_errors.npz", **per_frame
    )

    forbidden = {
        "triangulation_oracle2d",
        "sim3_face_stable_joint_weight",
    }
    baseline_valid = tuple(
        row
        for row in _baseline_rows(baseline_results, "valid_ranking")
        if str(row.get("method", "")) not in forbidden
        and str(row.get("ranking_group", "valid")) == "valid"
    )
    zero_shot = _protocol_matched_zero_shot(
        tuple(
            row
            for row in baseline_valid
            if str(row.get("method", "")).startswith("A")
        ),
        baseline_results,
    )
    valid_nonlearned = tuple(
        row
        for row in baseline_valid
        if not str(row.get("method", "")).startswith("A")
    )
    diagnostics = _baseline_rows(baseline_results, "diagnostics")
    machine = {
        "supervised_ranking": bundle.supervised_ranking,
        "zero_shot_ranking": zero_shot,
        "valid_nonlearned_ranking": valid_nonlearned,
        "diagnostics": diagnostics,
        "failures": bundle.failures,
        "tables": bundle.tables,
        "provenance": bundle.provenance,
    }
    _atomic_json(report_root / "results.json", machine)

    zero_by_method = {
        str(row["method"]): row for row in zero_shot
    }
    valid_by_method = {
        str(row["method"]): row for row in valid_nonlearned
    }
    conclusions: list[str] = []
    if bundle.supervised_ranking:
        best = bundle.supervised_ranking[0]
        ablation = str(best["ablation"])
        fine = float(best["macro_mpjpe_mm"])
        conclusions.append(
            f"- Best direction-held-out fine-tuned model: `{ablation}` at "
            f"{fine:.3f} mm macro MPJPE."
        )
        matching = zero_by_method.get(ablation)
        if matching is not None:
            zero = float(matching["mpjpe_mm"])
            change = fine - zero
            percentage = change / zero * 100.0
            direction = "improvement" if change < 0 else "degradation"
            conclusions.append(
                f"- Versus matching zero-shot `{ablation}` ({zero:.3f} mm): "
                f"{abs(change):.3f} mm / {abs(percentage):.2f}% {direction}."
            )
        direct = valid_by_method.get("avg_world_face_ref")
        if direct is not None:
            direct_value = float(direct["mpjpe_mm"])
            relation = "beats" if fine < direct_value else "does not beat"
            conclusions.append(
                f"- It {relation} the best direct-fusion reference "
                f"({direct_value:.3f} mm)."
            )
        triangulation = valid_by_method.get("triangulation_sam3d2d")
        if triangulation is not None:
            triangulation_value = float(triangulation["mpjpe_mm"])
            relation = "beats" if fine < triangulation_value else "does not approach"
            conclusions.append(
                f"- It {relation} SAM3D-2D triangulation "
                f"({triangulation_value:.3f} mm)."
            )
        fold_rows = [
            row
            for row in bundle.tables["by_fold"]
            if row["ablation"] == ablation
        ]
        if len(fold_rows) == 2:
            conclusions.append(
                "- Direction means: "
                + ", ".join(
                    f"{row['fold']}={float(row['mean_mpjpe_mm']):.3f} mm"
                    for row in fold_rows
                )
                + "."
            )
        static = next(
            (
                row
                for row in bundle.static_diagnostics
                if row["ablation"] == ablation
            ),
            None,
        )
        if static is not None:
            conclusions.append(
                "- Static OOD: "
                f"{float(static['mean_mpjpe_mm']):.3f} ± "
                f"{float(static['std_mpjpe_mm']):.3f} mm across runs."
            )

    try:
        import matplotlib.pyplot as plt

        methods = [
            str(row["ablation"]) for row in bundle.supervised_ranking
        ]
        if methods:
            supervised_values = [
                float(row["macro_mpjpe_mm"])
                for row in bundle.supervised_ranking
            ]
            zero_values = [
                float(zero_by_method[method]["mpjpe_mm"])
                if method in zero_by_method
                else np.nan
                for method in methods
            ]
            positions = np.arange(len(methods))
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.bar(
                positions - 0.2,
                zero_values,
                width=0.4,
                label="Zero-shot",
            )
            axis.bar(
                positions + 0.2,
                supervised_values,
                width=0.4,
                label="Unity-supervised",
            )
            axis.set_xticks(positions, methods)
            axis.set_ylabel("MPJPE (mm)")
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                figures_root / "zero_shot_vs_supervised_mpjpe.png",
                dpi=160,
            )
            plt.close(figure)
    except ImportError:
        pass

    lines = [
        "# Unity-Supervised Training",
        "",
        "Unity GT used for training; ranking evidence comes only from the "
        "held-out motion direction.",
        "",
        "## Direction-Held-Out Results",
        "",
        *conclusions,
        "",
        *_markdown_table(
            bundle.supervised_ranking,
            (
                "ablation",
                "macro_mpjpe_mm",
                "seed_std_mpjpe_mm",
                "macro_angle_mae_deg",
            ),
        ),
        "",
        "## Zero-Shot vs Fine-Tuned",
        "",
        *_markdown_table(
            zero_shot,
            ("method", "mpjpe_mm", "angle_mae_deg"),
        ),
        "",
        "## Static OOD Diagnostic",
        "",
        *_markdown_table(
            bundle.static_diagnostics,
            ("ablation", "mean_mpjpe_mm", "std_mpjpe_mm"),
        ),
        "",
        "## Triangulation Comparison",
        "",
        "Triangulation remains an independent non-learned comparison and is "
        "never mixed into the supervised ranking.",
        "",
        "## Interpretation Boundary",
        "",
        "This Unity benchmark contains one avatar in one rendered environment. "
        "It does not establish population-level generalization or statistical "
        "significance.",
        "",
    ]
    report_path = report_root / "unity_supervised_finetune_report.md"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    return report_path
