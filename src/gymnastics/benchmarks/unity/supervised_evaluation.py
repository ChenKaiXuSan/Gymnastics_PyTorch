"""Common Unity evaluation and strict fold/seed aggregation for fine-tuning."""

from __future__ import annotations

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
