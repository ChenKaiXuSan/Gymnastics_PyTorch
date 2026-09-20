"""Evaluation of fused sequences against the triangulated pseudo-reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from fusion.baselines.data import build_pair_index, load_triangulated_sequence
from fusion.baselines.methods import (
    PELVIS_INDICES,
    apply_sim3,
    estimate_joint_weights,
    fit_similarity,
    root_normalize,
)


@dataclass(frozen=True)
class PersonMetric:
    person_id: str
    method: str
    eval_frames: int
    valid_points: int
    mpjpe: float
    median: float
    p95: float
    max_error: float


@dataclass(frozen=True)
class JointMetric:
    person_id: str
    method: str
    joint: int
    valid_points: int
    mpjpe: float
    median: float
    p95: float
    max_error: float


ALIGNMENT_MODES = ("root", "similarity", "procrustes")


DEFAULT_ALIGNMENT = "similarity"


def align_candidate(
    candidate: np.ndarray,
    triangulated: np.ndarray,
    valid: np.ndarray,
    alignment: str,
) -> np.ndarray:
    """Bring ``candidate`` into the triangulated frame before measuring error.

    ``root`` only removes the pelvis translation. It leaves any difference in
    world orientation or scale between the SAM3D world frame and the calibrated
    triangulation frame inside the reported error, which makes the metric
    insensitive to the fusion itself.

    ``similarity`` fits one Sim3 transform per sequence, which removes exactly
    that static frame and scale mismatch while keeping per-frame pose and
    rotation differences measurable. ``procrustes`` fits one transform per frame
    and therefore also removes per-frame orientation error; it is a diagnostic
    lower bound on shape error, not a pose metric.
    """
    if alignment not in ALIGNMENT_MODES:
        raise ValueError(f"alignment must be one of {ALIGNMENT_MODES}: {alignment}")
    if alignment == "root":
        return root_normalize(candidate)
    if alignment == "similarity":
        if valid.sum() < 3:
            return root_normalize(candidate)
        transform = fit_similarity(candidate[valid], triangulated[valid])
        return apply_sim3(candidate, transform).astype(np.float64)
    aligned = np.array(candidate, dtype=np.float64, copy=True)
    for frame in range(candidate.shape[0]):
        frame_valid = valid[frame]
        if frame_valid.sum() < 3:
            aligned[frame] = candidate[frame] - np.nanmean(
                candidate[frame][list(PELVIS_INDICES)], axis=0
            )
            continue
        transform = fit_similarity(
            candidate[frame][frame_valid], triangulated[frame][frame_valid]
        )
        aligned[frame] = apply_sim3(candidate[frame], transform)
    return aligned


def joint_errors(
    candidate: np.ndarray,
    triangulated: np.ndarray,
    alignment: str = DEFAULT_ALIGNMENT,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(candidate).all(axis=-1) & np.isfinite(triangulated).all(axis=-1)
    if alignment == "root":
        candidate_aligned = root_normalize(candidate)
        reference = root_normalize(triangulated)
    else:
        candidate_aligned = align_candidate(candidate, triangulated, valid, alignment)
        reference = triangulated
    errors = np.linalg.norm(candidate_aligned - reference, axis=-1)
    return errors, valid


def summarize_values(values: np.ndarray) -> Dict[str, float | int]:
    if values.size == 0:
        return {
            "valid_points": 0,
            "mpjpe": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max_error": float("nan"),
        }
    return {
        "valid_points": int(values.size),
        "mpjpe": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max_error": float(np.max(values)),
    }


def evaluate_sequence(
    person_id: str,
    method: str,
    fused_world: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
    triangulated_person_root: Path,
    alignment: str = DEFAULT_ALIGNMENT,
) -> Tuple[PersonMetric, List[JointMetric]]:
    pair_index = build_pair_index(face_map, side_map)
    all_values: List[np.ndarray] = []
    joint_values: List[List[np.ndarray]] = []
    eval_frames = 0
    n_joints = fused_world.shape[1]
    joint_values = [[] for _ in range(n_joints)]

    for cycle_root in sorted(triangulated_person_root.glob("cycle_*")):
        triangulated, pairs = load_triangulated_sequence(cycle_root)
        candidate_frames = []
        tri_frames = []
        for tri_frame, pair in zip(triangulated, pairs):
            fused_idx = pair_index.get(pair)
            if fused_idx is None:
                continue
            candidate_frames.append(fused_world[fused_idx])
            tri_frames.append(tri_frame)
        if not candidate_frames:
            continue
        eval_frames += len(candidate_frames)
        errors, valid = joint_errors(
            np.stack(candidate_frames), np.stack(tri_frames), alignment=alignment
        )
        all_values.append(errors[valid])
        for joint_idx in range(n_joints):
            joint_values[joint_idx].append(errors[:, joint_idx][valid[:, joint_idx]])

    values = np.concatenate(all_values) if all_values else np.asarray([], dtype=np.float32)
    stats = summarize_values(values)
    person_metric = PersonMetric(
        person_id=person_id,
        method=method,
        eval_frames=eval_frames,
        valid_points=int(stats["valid_points"]),
        mpjpe=float(stats["mpjpe"]),
        median=float(stats["median"]),
        p95=float(stats["p95"]),
        max_error=float(stats["max_error"]),
    )

    joint_metrics: List[JointMetric] = []
    for joint_idx, chunks in enumerate(joint_values):
        joint_error_values = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        joint_stats = summarize_values(joint_error_values)
        joint_metrics.append(
            JointMetric(
                person_id=person_id,
                method=method,
                joint=joint_idx,
                valid_points=int(joint_stats["valid_points"]),
                mpjpe=float(joint_stats["mpjpe"]),
                median=float(joint_stats["median"]),
                p95=float(joint_stats["p95"]),
                max_error=float(joint_stats["max_error"]),
            )
        )
    return person_metric, joint_metrics


def estimate_weights_from_triangulated(
    face: np.ndarray,
    side_aligned: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
    triangulated_person_root: Path,
    alignment: str = DEFAULT_ALIGNMENT,
) -> np.ndarray:
    pair_index = build_pair_index(face_map, side_map)
    face_errors_by_joint: List[List[np.ndarray]] = [[] for _ in range(face.shape[1])]
    side_errors_by_joint: List[List[np.ndarray]] = [[] for _ in range(face.shape[1])]
    for cycle_root in sorted(triangulated_person_root.glob("cycle_*")):
        triangulated, pairs = load_triangulated_sequence(cycle_root)
        face_frames = []
        side_frames = []
        tri_frames = []
        for tri_frame, pair in zip(triangulated, pairs):
            idx = pair_index.get(pair)
            if idx is None:
                continue
            face_frames.append(face[idx])
            side_frames.append(side_aligned[idx])
            tri_frames.append(tri_frame)
        if not face_frames:
            continue
        tri_seq = np.stack(tri_frames)
        face_errors, face_valid = joint_errors(
            np.stack(face_frames), tri_seq, alignment=alignment
        )
        side_errors, side_valid = joint_errors(
            np.stack(side_frames), tri_seq, alignment=alignment
        )
        for joint_idx in range(face.shape[1]):
            face_errors_by_joint[joint_idx].append(face_errors[:, joint_idx][face_valid[:, joint_idx]])
            side_errors_by_joint[joint_idx].append(side_errors[:, joint_idx][side_valid[:, joint_idx]])

    face_joint_errors = np.empty((face.shape[1],), dtype=np.float32)
    side_joint_errors = np.empty((face.shape[1],), dtype=np.float32)
    for joint_idx in range(face.shape[1]):
        face_values = np.concatenate(face_errors_by_joint[joint_idx]) if face_errors_by_joint[joint_idx] else np.asarray([1.0])
        side_values = np.concatenate(side_errors_by_joint[joint_idx]) if side_errors_by_joint[joint_idx] else np.asarray([1.0])
        face_joint_errors[joint_idx] = float(np.mean(face_values))
        side_joint_errors[joint_idx] = float(np.mean(side_values))
    return estimate_joint_weights(face_joint_errors, side_joint_errors)

