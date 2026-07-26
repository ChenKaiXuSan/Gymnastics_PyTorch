"""Common sequence-level evaluation for all Unity benchmark methods."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from gymnastics.fusion.deterministic.experiment_matrix import (
    Sim3Transform,
    apply_sim3,
    fit_similarity,
)

from .mapping import EVALUATION_JOINT_NAMES
from .schema import MethodSequence


@dataclass(frozen=True)
class SequenceErrors:
    errors_m: np.ndarray
    valid: np.ndarray
    aligned_points_m: np.ndarray
    transform: Sim3Transform


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    sequence_id: str
    sample_ids: np.ndarray
    joint_names: tuple[str, ...]
    errors_m: np.ndarray
    valid: np.ndarray
    aligned_points_m: np.ndarray
    summary: Mapping[str, float | int | str]
    joint_rows: tuple[Mapping[str, object], ...]
    visibility_rows: tuple[Mapping[str, object], ...]
    angle_errors_deg: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class EvaluationBundle:
    results: tuple[EvaluationResult, ...]
    failures: tuple[Mapping[str, object], ...]
    valid_ranking: tuple[Mapping[str, object], ...]
    diagnostics: tuple[Mapping[str, object], ...]
    tables: Mapping[str, tuple[Mapping[str, object], ...]]
    provenance: Mapping[str, object]


def sequence_joint_errors(
    candidate: np.ndarray,
    candidate_valid: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
) -> SequenceErrors:
    prediction = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.shape[-1] != 3:
        raise ValueError("candidate and target must have equal shape [T,J,3]")
    valid = (
        np.asarray(candidate_valid, dtype=bool)
        & np.asarray(target_valid, dtype=bool)
        & np.isfinite(prediction).all(axis=-1)
        & np.isfinite(reference).all(axis=-1)
    )
    if valid.shape != prediction.shape[:-1]:
        raise ValueError("valid masks must have shape [T,J]")
    if valid.sum() < 3:
        transform = Sim3Transform(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )
    else:
        transform = fit_similarity(prediction[valid], reference[valid])
    aligned = apply_sim3(prediction, transform).astype(np.float32)
    errors = np.linalg.norm(aligned - reference, axis=-1).astype(np.float32)
    errors[~valid] = np.nan
    return SequenceErrors(errors, valid, aligned, transform)


def angular_residual_deg(
    prediction_deg: np.ndarray, target_deg: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(prediction_deg, dtype=np.float64)
        - np.asarray(target_deg, dtype=np.float64)
        + 180.0
    ) % 360.0 - 180.0


def trunk_rotation_deg(
    points: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(points, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    index = {name: i for i, name in enumerate(EVALUATION_JOINT_NAMES)}
    required = (
        index["Hips"],
        index["Neck"],
        index["LeftUpperArm"],
        index["RightUpperArm"],
        index["LeftUpperLeg"],
        index["RightUpperLeg"],
    )
    frame_valid = np.all(mask[:, required], axis=1)
    vertical = pose[:, index["Neck"]] - pose[:, index["Hips"]]
    pelvis = (
        pose[:, index["RightUpperLeg"]] - pose[:, index["LeftUpperLeg"]]
    )
    thorax = (
        pose[:, index["RightUpperArm"]] - pose[:, index["LeftUpperArm"]]
    )

    def normalize(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        return vector / np.maximum(norm, 1e-12), norm[:, 0] > 1e-8

    vertical, good_vertical = normalize(vertical)
    pelvis -= np.sum(pelvis * vertical, axis=-1, keepdims=True) * vertical
    thorax -= np.sum(thorax * vertical, axis=-1, keepdims=True) * vertical
    pelvis, good_pelvis = normalize(pelvis)
    thorax, good_thorax = normalize(thorax)
    frame_valid &= good_vertical & good_pelvis & good_thorax
    sine = np.sum(np.cross(pelvis, thorax) * vertical, axis=-1)
    cosine = np.sum(pelvis * thorax, axis=-1)
    angle = np.rad2deg(np.arctan2(sine, cosine)).astype(np.float32)
    angle[~frame_valid] = np.nan
    return angle, frame_valid


def _stats(values_m: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values_m, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "valid_points": 0,
            "mpjpe_mm": float("nan"),
            "median_mm": float("nan"),
            "p95_mm": float("nan"),
        }
    return {
        "valid_points": int(len(values)),
        "mpjpe_mm": float(np.mean(values) * 1000.0),
        "median_mm": float(np.median(values) * 1000.0),
        "p95_mm": float(np.percentile(values, 95) * 1000.0),
    }


def _visibility_rows(
    method: str,
    sequence_id: str,
    errors: np.ndarray,
    valid: np.ndarray,
    visibility: Mapping[str, np.ndarray],
) -> tuple[Mapping[str, object], ...]:
    cam0 = np.asarray(visibility["cam0"], dtype=bool)
    cam1 = np.asarray(visibility["cam1"], dtype=bool)
    partitions = {
        "cam0_visible": cam0,
        "cam0_occluded": ~cam0,
        "cam1_visible": cam1,
        "cam1_occluded": ~cam1,
        "both_visible": cam0 & cam1,
        "one_visible": cam0 ^ cam1,
        "neither_visible": ~cam0 & ~cam1,
    }
    rows = []
    for name, partition in partitions.items():
        row = {
            "method": method,
            "sequence_id": sequence_id,
            "visibility": name,
            **_stats(errors[valid & partition]),
        }
        rows.append(MappingProxyType(row))
    return tuple(rows)


def evaluate_method_sequence(
    candidate: MethodSequence,
    reference: MethodSequence,
    *,
    visibility: Mapping[str, np.ndarray],
    actual_angles_deg: np.ndarray,
    angle_offset_deg: float = 0.0,
) -> EvaluationResult:
    if candidate.sequence_id != reference.sequence_id:
        raise ValueError("candidate and reference sequence IDs differ")
    if candidate.joint_names != EVALUATION_JOINT_NAMES:
        raise ValueError("candidate must use the Unity16 evaluation joint order")
    if reference.joint_names != EVALUATION_JOINT_NAMES:
        raise ValueError("reference must use the Unity16 evaluation joint order")
    if not np.array_equal(candidate.sample_ids, reference.sample_ids):
        raise ValueError("candidate and reference sample IDs differ")
    evaluated = sequence_joint_errors(
        candidate.points,
        candidate.valid,
        reference.points,
        reference.valid,
    )
    summary: dict[str, float | int | str] = {
        "method": candidate.method,
        "sequence_id": candidate.sequence_id,
        "eval_frames": int(len(candidate.sample_ids)),
        "ranking_group": str(candidate.metadata.get("ranking_group", "valid")),
        **_stats(evaluated.errors_m[evaluated.valid]),
    }
    joint_rows = tuple(
        MappingProxyType(
            {
                "method": candidate.method,
                "sequence_id": candidate.sequence_id,
                "joint": name,
                **_stats(
                    evaluated.errors_m[:, joint_index][
                        evaluated.valid[:, joint_index]
                    ]
                ),
            }
        )
        for joint_index, name in enumerate(candidate.joint_names)
    )
    predicted_angles, angle_valid = trunk_rotation_deg(
        evaluated.aligned_points_m, evaluated.valid
    )
    actual = np.asarray(actual_angles_deg, dtype=np.float32)
    if actual.shape != (len(candidate.sample_ids),):
        raise ValueError("actual_angles_deg must have shape [T]")
    angle_errors = np.full(actual.shape, np.nan, dtype=np.float32)
    usable_angles = angle_valid & np.isfinite(actual)
    angle_errors[usable_angles] = angular_residual_deg(
        predicted_angles[usable_angles] - angle_offset_deg,
        actual[usable_angles],
    )
    if usable_angles.any():
        summary["angle_mae_deg"] = float(np.nanmean(np.abs(angle_errors)))
        summary["angle_rmse_deg"] = float(
            np.sqrt(np.nanmean(angle_errors.astype(np.float64) ** 2))
        )
    else:
        summary["angle_mae_deg"] = float("nan")
        summary["angle_rmse_deg"] = float("nan")
    return EvaluationResult(
        method=candidate.method,
        sequence_id=candidate.sequence_id,
        sample_ids=np.asarray(candidate.sample_ids),
        joint_names=candidate.joint_names,
        errors_m=evaluated.errors_m,
        valid=evaluated.valid,
        aligned_points_m=evaluated.aligned_points_m,
        summary=MappingProxyType(summary),
        joint_rows=joint_rows,
        visibility_rows=_visibility_rows(
            candidate.method,
            candidate.sequence_id,
            evaluated.errors_m,
            evaluated.valid,
            visibility,
        ),
        angle_errors_deg=angle_errors,
        metadata=MappingProxyType(dict(candidate.metadata)),
    )


def summarize_results(
    results: Sequence[EvaluationResult],
    *,
    failures: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object] | None = None,
) -> EvaluationBundle:
    grouped: dict[str, list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.method, []).append(result)
    summary_rows: list[Mapping[str, object]] = []
    for method, chunks in grouped.items():
        errors = np.concatenate(
            [chunk.errors_m[chunk.valid] for chunk in chunks]
        )
        angle_errors = np.concatenate(
            [chunk.angle_errors_deg[np.isfinite(chunk.angle_errors_deg)] for chunk in chunks]
        )
        row: dict[str, object] = {
            "method": method,
            "sequences": len(chunks),
            "ranking_group": chunks[0].summary["ranking_group"],
            **_stats(errors),
            "angle_mae_deg": float(np.mean(np.abs(angle_errors)))
            if len(angle_errors)
            else float("nan"),
            "angle_rmse_deg": float(np.sqrt(np.mean(angle_errors**2)))
            if len(angle_errors)
            else float("nan"),
        }
        summary_rows.append(MappingProxyType(row))
    valid_ranking = tuple(
        sorted(
            (row for row in summary_rows if row["ranking_group"] == "valid"),
            key=lambda row: float(row["mpjpe_mm"]),
        )
    )
    diagnostics = tuple(
        sorted(
            (row for row in summary_rows if row["ranking_group"] == "diagnostic"),
            key=lambda row: float(row["mpjpe_mm"]),
        )
    )
    tables = {
        "summary": tuple(summary_rows),
        "by_sequence": tuple(result.summary for result in results),
        "by_joint": tuple(
            row for result in results for row in result.joint_rows
        ),
        "by_visibility": tuple(
            row for result in results for row in result.visibility_rows
        ),
    }
    return EvaluationBundle(
        results=tuple(results),
        failures=tuple(MappingProxyType(dict(item)) for item in failures),
        valid_ranking=valid_ranking,
        diagnostics=diagnostics,
        tables=MappingProxyType(tables),
        provenance=MappingProxyType(dict(provenance or {})),
    )
