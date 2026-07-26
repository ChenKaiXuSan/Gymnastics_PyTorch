"""Adapters from synchronized Unity samples into existing fusion algorithms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
from typing import Mapping

import numpy as np

from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.deterministic.experiment_matrix import (
    ALL_METHODS,
    STABLE_SIM3_JOINTS,
    apply_sim3,
    bodypart_weights,
    current_body_average,
    estimate_joint_weights,
    fit_similarity,
    fuse_weighted,
    root_align_to_reference,
    sim3_align_to_reference,
    smooth_sequence,
)

from .dataset import group_evaluation_sequences
from .mapping import (
    MHR70_EVALUATION_SOURCES,
    map_mhr70_to_unity,
    select_unity_evaluation_joints,
)
from .sam3d import load_sam3d_camera_cache
from .schema import MethodSequence, UnityBenchmark


def fuse_deterministic_sequence(
    method: str,
    cam0: np.ndarray,
    cam1: np.ndarray,
    *,
    leaky_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Fuse one synchronized MHR70 sequence without dataset file assumptions."""
    face = np.asarray(cam0, dtype=np.float32)
    side = np.asarray(cam1, dtype=np.float32)
    if face.shape != side.shape or face.ndim != 3 or face.shape[1:] != (70, 3):
        raise ValueError("cam0 and cam1 must have equal shape [T,70,3]")
    if method not in ALL_METHODS:
        raise ValueError(f"unsupported deterministic method: {method}")
    metadata: dict[str, object] = {
        "method": method,
        "camera_reference": "cam0",
        "offset_cam1_to_cam0": 0,
        "ranking_group": "valid",
    }
    sim3_stable = None
    stable_scales = None
    if method == "avg_body_current":
        fused = current_body_average(face, side)
    elif method == "avg_world_face_ref":
        fused = 0.5 * (face + side)
    elif method == "root_face_stable":
        fused = 0.5 * (face + root_align_to_reference(side, face))
    elif method == "sim3_face_all":
        aligned, scales = sim3_align_to_reference(
            side, face, tuple(range(face.shape[1]))
        )
        fused = 0.5 * (face + aligned)
        metadata["scale_mean"] = float(np.mean(scales))
    else:
        sim3_stable, stable_scales = sim3_align_to_reference(
            side, face, STABLE_SIM3_JOINTS
        )
        metadata["sim3_joints"] = list(STABLE_SIM3_JOINTS)
        metadata["scale_mean"] = float(np.mean(stable_scales))
        if method == "sim3_face_stable":
            fused = 0.5 * (face + sim3_stable)
        elif method == "sim3_face_stable_joint_weight":
            if leaky_weights is None:
                raise ValueError(
                    "sim3_face_stable_joint_weight requires explicit leaky_weights"
                )
            fused = fuse_weighted(face, sim3_stable, leaky_weights)
            metadata["ranking_group"] = "diagnostic"
            metadata["leakage"] = "unity_gt_joint_weights"
            metadata["joint_weights"] = np.asarray(leaky_weights).tolist()
        elif method == "sim3_face_stable_bodypart_weight":
            weights = bodypart_weights(face.shape[1])
            fused = fuse_weighted(face, sim3_stable, weights)
            metadata["joint_weights"] = weights.tolist()
        elif method == "sim3_face_stable_smooth_transform":
            fused = 0.5 * (face + smooth_sequence(sim3_stable, win=5))
            metadata["smooth_target"] = "side_after_sim3"
            metadata["smooth_window"] = 5
        elif method == "sim3_face_stable_smooth_kpt":
            fused = smooth_sequence(0.5 * (face + sim3_stable), win=5)
            metadata["smooth_target"] = "fused_world"
            metadata["smooth_window"] = 5
        else:
            raise AssertionError(f"unhandled deterministic method: {method}")
    return np.asarray(fused, dtype=np.float32), metadata


def _aligned_joint_errors(
    candidate: np.ndarray,
    candidate_valid: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    common = candidate_valid & target_valid
    transform = fit_similarity(candidate[common], target[common])
    aligned = apply_sim3(candidate, transform)
    errors = np.linalg.norm(aligned - target, axis=-1)
    return errors, common


def estimate_leaky_joint_weights(
    face: np.ndarray,
    side: np.ndarray,
    gt_world: np.ndarray,
    gt_valid: np.ndarray,
) -> np.ndarray:
    """Fit MHR70 view weights from Unity GT for the labelled diagnostic only."""
    aligned_side, _ = sim3_align_to_reference(
        np.asarray(side, dtype=np.float32),
        np.asarray(face, dtype=np.float32),
        STABLE_SIM3_JOINTS,
    )
    face_mapped = map_mhr70_to_unity(face)
    side_mapped = map_mhr70_to_unity(aligned_side)
    gt_mapped = select_unity_evaluation_joints(gt_world, gt_valid)
    face_errors, face_common = _aligned_joint_errors(
        face_mapped.points,
        face_mapped.valid,
        gt_mapped.points,
        gt_mapped.valid,
    )
    side_errors, side_common = _aligned_joint_errors(
        side_mapped.points,
        side_mapped.valid,
        gt_mapped.points,
        gt_mapped.valid,
    )
    face_joint = np.asarray(
        [
            np.mean(face_errors[:, joint][face_common[:, joint]])
            for joint in range(face_errors.shape[1])
        ],
        dtype=np.float32,
    )
    side_joint = np.asarray(
        [
            np.mean(side_errors[:, joint][side_common[:, joint]])
            for joint in range(side_errors.shape[1])
        ],
        dtype=np.float32,
    )
    mapped_weights = estimate_joint_weights(face_joint, side_joint)
    weights = np.full((70, 2), 0.5, dtype=np.float32)
    for target_index, name in enumerate(face_mapped.joint_names):
        for source_index in MHR70_EVALUATION_SOURCES[name]:
            weights[source_index] = mapped_weights[target_index]
    return weights


def _save_sequence(path: Path, sequence: MethodSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            method=np.asarray(sequence.method),
            sequence_id=np.asarray(sequence.sequence_id),
            sample_ids=sequence.sample_ids,
            points=sequence.points,
            valid=sequence.valid,
            joint_names=np.asarray(sequence.joint_names),
            metadata=np.asarray(json.dumps(dict(sequence.metadata), sort_keys=True)),
        )
    temporary.replace(path)


def run_deterministic_fusion(
    benchmark: UnityBenchmark,
    cache_root: Path,
    output_root: Path,
    *,
    methods: Sequence[str] = ALL_METHODS,
) -> tuple[MethodSequence, ...]:
    outputs: list[MethodSequence] = []
    for sequence_id, frames in group_evaluation_sequences(benchmark).items():
        sample_ids = tuple(frame.sample_id for frame in frames)
        face = load_sam3d_camera_cache(cache_root, "cam0", sample_ids)
        side = load_sam3d_camera_cache(cache_root, "cam1", sample_ids)
        gt_world = np.stack([frame.gt_world_m for frame in frames])
        gt_valid = np.stack([frame.gt_available for frame in frames])
        fused_valid = face.valid_3d & side.valid_3d
        for method in methods:
            leaky_weights = (
                estimate_leaky_joint_weights(
                    face.points_3d, side.points_3d, gt_world, gt_valid
                )
                if method == "sim3_face_stable_joint_weight"
                else None
            )
            fused, metadata = fuse_deterministic_sequence(
                method,
                face.points_3d,
                side.points_3d,
                leaky_weights=leaky_weights,
            )
            fused = np.where(fused_valid[..., None], fused, 0)
            sequence = MethodSequence(
                method=method,
                sequence_id=sequence_id,
                sample_ids=np.asarray(sample_ids),
                points=fused,
                valid=fused_valid,
                joint_names=tuple(mhr_names),
                metadata=metadata,
            )
            _save_sequence(
                Path(output_root)
                / "deterministic"
                / method
                / f"{sequence_id}.npz",
                sequence,
            )
            outputs.append(sequence)
    return tuple(outputs)
