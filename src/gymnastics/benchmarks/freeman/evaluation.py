"""Common FreeMan alignment, metrics, aggregation, and statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gymnastics.fusion.deterministic.experiment_matrix import (
    apply_sim3,
    fit_similarity,
)

from .mapping import FREEMAN_COCO17_NAMES, map_mhr70_to_freeman
from .schema import MethodPrediction, ReferenceSequence


_LEFT_HIP = FREEMAN_COCO17_NAMES.index("left-hip")
_RIGHT_HIP = FREEMAN_COCO17_NAMES.index("right-hip")
_METRIC_COLUMNS = (
    "sim3_mpjpe_mm",
    "median_mpjpe_mm",
    "p95_mpjpe_mm",
    "max_mpjpe_mm",
    "root_mpjpe_mm",
    "pa_mpjpe_mm",
    "velocity_error_mm_s",
    "acceleration_error_mm_s2",
    "auc",
    "coverage",
)


@dataclass(frozen=True)
class SessionMetrics:
    subject_id: int
    session_id: str
    fps: float
    split: str
    scenario: str | None
    action: str | None
    method: str
    classification: str
    frames_total: int
    frames_valid: int
    valid_points: int
    sim3_mpjpe_mm: float
    median_mpjpe_mm: float
    p95_mpjpe_mm: float
    max_mpjpe_mm: float
    root_mpjpe_mm: float
    pa_mpjpe_mm: float
    velocity_error_mm_s: float
    acceleration_error_mm_s2: float
    pck: Mapping[int, float]
    auc: float
    coverage: float
    per_joint_mpjpe_mm: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationTables:
    by_session: pd.DataFrame
    by_subject: pd.DataFrame
    by_method: pd.DataFrame
    by_joint: pd.DataFrame
    by_split: pd.DataFrame
    by_scenario: pd.DataFrame
    paired_statistics: pd.DataFrame
    failures: pd.DataFrame


def _matched_reference(
    prediction: MethodPrediction,
    reference: ReferenceSequence,
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.session_id != reference.session_id:
        raise ValueError("prediction and reference session IDs differ")
    lookup = {
        int(frame_id): index for index, frame_id in enumerate(reference.frame_ids)
    }
    missing = [
        int(frame_id)
        for frame_id in prediction.frame_ids
        if int(frame_id) not in lookup
    ]
    if missing:
        raise ValueError(f"prediction frame IDs are absent from reference frame IDs: {missing}")
    indices = np.asarray(
        [lookup[int(frame_id)] for frame_id in prediction.frame_ids],
        dtype=np.int64,
    )
    return reference.points_m[indices], reference.valid[indices]


def _mean_masked(values: np.ndarray, valid: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(valid, dtype=bool)]
    return float(np.mean(selected)) if selected.size else float("nan")


def _root_relative(points: np.ndarray) -> np.ndarray:
    root = 0.5 * (
        points[:, _LEFT_HIP : _LEFT_HIP + 1]
        + points[:, _RIGHT_HIP : _RIGHT_HIP + 1]
    )
    return points - root


def _pa_errors(
    candidate: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    errors = np.full(valid.shape, np.nan, dtype=np.float64)
    for frame in range(len(candidate)):
        frame_valid = valid[frame]
        if frame_valid.sum() < 3:
            continue
        transform = fit_similarity(
            candidate[frame, frame_valid],
            reference[frame, frame_valid],
        )
        aligned = apply_sim3(candidate[frame], transform)
        errors[frame, frame_valid] = np.linalg.norm(
            aligned[frame_valid] - reference[frame, frame_valid],
            axis=-1,
        )
    return errors


def _temporal_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    order: int,
) -> float:
    if len(candidate) <= order:
        return float("nan")
    candidate_delta = np.diff(candidate, n=order, axis=0) * (fps**order)
    reference_delta = np.diff(reference, n=order, axis=0) * (fps**order)
    temporal_valid = np.ones(candidate_delta.shape[:2], dtype=bool)
    for offset in range(order + 1):
        temporal_valid &= valid[offset : offset + len(candidate_delta)]
    errors = np.linalg.norm(candidate_delta - reference_delta, axis=-1)
    return _mean_masked(errors, temporal_valid) * 1000.0


def _normalized_auc(errors_mm: np.ndarray, maximum_mm: float) -> float:
    if errors_mm.size == 0 or maximum_mm <= 0:
        return float("nan")
    thresholds = np.linspace(0.0, maximum_mm, 1001)
    curve = np.asarray(
        [np.mean(errors_mm <= threshold) for threshold in thresholds],
        dtype=np.float64,
    )
    return float(np.trapezoid(curve, thresholds) / maximum_mm)


def evaluate_session(
    prediction: MethodPrediction,
    reference: ReferenceSequence,
    thresholds_mm: Sequence[float],
) -> SessionMetrics:
    """Evaluate one prediction with one Sim3 fitted across the full session."""
    mapped = map_mhr70_to_freeman(prediction.points, prediction.valid)
    reference_points, reference_valid = _matched_reference(prediction, reference)
    valid = mapped.valid & reference_valid
    if int(valid.sum()) < 3:
        raise ValueError("session evaluation requires at least three valid joints")
    candidate_fit = mapped.points[valid]
    reference_fit = reference_points[valid]
    if (
        np.linalg.matrix_rank(candidate_fit - candidate_fit.mean(axis=0)) < 2
        or np.linalg.matrix_rank(reference_fit - reference_fit.mean(axis=0)) < 2
    ):
        raise ValueError("session evaluation requires non-collinear valid joints")
    transform = fit_similarity(candidate_fit, reference_fit)
    aligned = apply_sim3(mapped.points, transform).astype(np.float64)
    errors_m = np.linalg.norm(aligned - reference_points, axis=-1)
    errors_mm = errors_m * 1000.0
    selected_errors = errors_mm[valid]
    root_candidate = _root_relative(aligned)
    root_reference = _root_relative(reference_points)
    root_valid = valid & (
        valid[:, _LEFT_HIP] & valid[:, _RIGHT_HIP]
    )[:, None]
    root_errors_mm = (
        np.linalg.norm(root_candidate - root_reference, axis=-1) * 1000.0
    )
    pa_errors_mm = _pa_errors(
        mapped.points,
        reference_points,
        valid,
    ) * 1000.0
    thresholds = tuple(float(value) for value in thresholds_mm)
    if not thresholds or any(
        not np.isfinite(value) or value <= 0 for value in thresholds
    ):
        raise ValueError("PCK thresholds must be positive finite millimetres")
    pck = {
        int(value) if float(value).is_integer() else float(value): float(
            np.mean(selected_errors <= value)
        )
        for value in thresholds
    }
    per_joint = tuple(
        _mean_masked(errors_mm[:, joint], valid[:, joint])
        for joint in range(errors_mm.shape[1])
    )
    frame_valid = valid.any(axis=1)
    classification = str(prediction.metadata.get("classification", "VALID"))
    return SessionMetrics(
        subject_id=reference.subject_id,
        session_id=reference.session_id,
        fps=reference.fps,
        split=reference.split,
        scenario=reference.scenario,
        action=reference.action,
        method=prediction.method,
        classification=classification,
        frames_total=len(prediction.frame_ids),
        frames_valid=int(frame_valid.sum()),
        valid_points=int(valid.sum()),
        sim3_mpjpe_mm=float(np.mean(selected_errors)),
        median_mpjpe_mm=float(np.median(selected_errors)),
        p95_mpjpe_mm=float(np.quantile(selected_errors, 0.95)),
        max_mpjpe_mm=float(np.max(selected_errors)),
        root_mpjpe_mm=_mean_masked(root_errors_mm, root_valid),
        pa_mpjpe_mm=_mean_masked(pa_errors_mm, valid),
        velocity_error_mm_s=_temporal_error(
            aligned,
            reference_points,
            valid,
            fps=prediction.fps,
            order=1,
        ),
        acceleration_error_mm_s2=_temporal_error(
            aligned,
            reference_points,
            valid,
            fps=prediction.fps,
            order=2,
        ),
        pck=pck,
        auc=_normalized_auc(selected_errors, max(thresholds)),
        coverage=float(valid.sum() / valid.size),
        per_joint_mpjpe_mm=per_joint,
    )


def _session_record(metric: SessionMetrics) -> dict[str, object]:
    record = asdict(metric)
    pck = record.pop("pck")
    record.pop("per_joint_mpjpe_mm")
    for threshold, value in pck.items():
        record[f"pck_{threshold}_mm"] = value
    return record


def _mean_table(
    frame: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    numeric = [
        column
        for column in (
            "frames_total",
            "frames_valid",
            "valid_points",
            *_METRIC_COLUMNS,
            *(column for column in frame.columns if column.startswith("pck_")),
        )
        if column in frame.columns
    ]
    return (
        frame.groupby(groups, dropna=False, as_index=False)[numeric]
        .mean(numeric_only=True)
        .sort_values(groups)
        .reset_index(drop=True)
    )


def aggregate_metrics(rows: Sequence[SessionMetrics]) -> EvaluationTables:
    """Build subject-balanced result tables from session metric rows."""
    if not rows:
        raise ValueError("aggregate_metrics requires at least one session row")
    by_session = pd.DataFrame([_session_record(row) for row in rows]).sort_values(
        ["subject_id", "session_id", "method"]
    ).reset_index(drop=True)
    by_subject = _mean_table(
        by_session,
        ["subject_id", "method", "classification"],
    )
    by_method = _mean_table(
        by_subject,
        ["method", "classification"],
    )
    by_split = _mean_table(
        by_session,
        ["split", "method", "classification"],
    )
    scenario_rows = by_session[by_session["scenario"].notna()]
    by_scenario = (
        _mean_table(
            scenario_rows,
            ["scenario", "method", "classification"],
        )
        if not scenario_rows.empty
        else pd.DataFrame()
    )
    joint_records = [
        {
            "subject_id": row.subject_id,
            "session_id": row.session_id,
            "fps": row.fps,
            "split": row.split,
            "scenario": row.scenario,
            "action": row.action,
            "method": row.method,
            "classification": row.classification,
            "joint": joint,
            "joint_name": FREEMAN_COCO17_NAMES[joint],
            "mpjpe_mm": value,
        }
        for row in rows
        for joint, value in enumerate(row.per_joint_mpjpe_mm)
    ]
    by_joint = pd.DataFrame(joint_records).sort_values(
        ["subject_id", "session_id", "method", "joint"]
    ).reset_index(drop=True)
    paired = (
        paired_method_tests(
            by_subject,
            seed=20260726,
            bootstrap_samples=10_000,
        )
        if {"view_a", "view_b"} & set(by_subject["method"])
        else pd.DataFrame()
    )
    return EvaluationTables(
        by_session=by_session,
        by_subject=by_subject,
        by_method=by_method,
        by_joint=by_joint,
        by_split=by_split,
        by_scenario=by_scenario,
        paired_statistics=paired,
        failures=pd.DataFrame(
            columns=["subject_id", "session_id", "stage", "reason"]
        ),
    )


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_method_tests(
    subject_table: pd.DataFrame,
    *,
    seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    """Compare valid methods with each fixed single-view baseline by subject."""
    required = {
        "subject_id",
        "method",
        "classification",
        "sim3_mpjpe_mm",
    }
    if not required.issubset(subject_table.columns):
        raise ValueError(f"subject table missing columns: {sorted(required - set(subject_table))}")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    valid = subject_table[subject_table["classification"] == "VALID"]
    baselines = [
        name for name in ("view_a", "view_b") if name in set(valid["method"])
    ]
    methods = sorted(set(valid["method"]) - {"view_a", "view_b"})
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    measured_indices: list[int] = []
    p_values: list[float] = []
    for baseline in baselines:
        baseline_rows = valid[valid["method"] == baseline][
            ["subject_id", "sim3_mpjpe_mm"]
        ].rename(columns={"sim3_mpjpe_mm": "baseline_value"})
        for method in methods:
            candidate_rows = valid[valid["method"] == method][
                ["subject_id", "sim3_mpjpe_mm"]
            ].rename(columns={"sim3_mpjpe_mm": "candidate_value"})
            matched = baseline_rows.merge(candidate_rows, on="subject_id", how="inner")
            difference = (
                matched["candidate_value"].to_numpy(dtype=np.float64)
                - matched["baseline_value"].to_numpy(dtype=np.float64)
            )
            record: dict[str, object] = {
                "method": method,
                "baseline": baseline,
                "matched_subjects": len(difference),
                "mean_difference_mm": float(np.mean(difference))
                if len(difference)
                else float("nan"),
                "median_difference_mm": float(np.median(difference))
                if len(difference)
                else float("nan"),
                "ci95_low_mm": float("nan"),
                "ci95_high_mm": float("nan"),
                "p_value": float("nan"),
                "holm_p_value": float("nan"),
                "status": "insufficient_subject_coverage",
            }
            if len(difference) >= 10:
                sample_indices = rng.integers(
                    0,
                    len(difference),
                    size=(bootstrap_samples, len(difference)),
                )
                bootstrap_means = difference[sample_indices].mean(axis=1)
                record["ci95_low_mm"] = float(np.quantile(bootstrap_means, 0.025))
                record["ci95_high_mm"] = float(np.quantile(bootstrap_means, 0.975))
                if np.allclose(difference, 0):
                    p_value = 1.0
                else:
                    p_value = float(wilcoxon(difference).pvalue)
                record["p_value"] = p_value
                record["status"] = "measured"
                measured_indices.append(len(records))
                p_values.append(p_value)
            records.append(record)
    adjusted = _holm_adjust(p_values)
    for index, value in zip(measured_indices, adjusted):
        records[index]["holm_p_value"] = value
    columns = [
        "method",
        "baseline",
        "matched_subjects",
        "mean_difference_mm",
        "median_difference_mm",
        "ci95_low_mm",
        "ci95_high_mm",
        "p_value",
        "holm_p_value",
        "status",
    ]
    return pd.DataFrame(records, columns=columns)
