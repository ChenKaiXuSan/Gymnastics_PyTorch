"""Unity camera projection and calibrated triangulation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import group_evaluation_sequences
from .mapping import EVALUATION_JOINT_NAMES, select_unity_evaluation_joints
from .schema import MethodSequence, UnityBenchmark, UnityCamera


def pixel_projection(camera: UnityCamera) -> np.ndarray:
    """Return the 3x4 homogeneous world-to-pixel projection matrix."""
    clip_from_world = camera.clip_projection @ camera.world_to_camera
    sx = (camera.image_size[0] - 1) / 2.0
    sy = (camera.image_size[1] - 1) / 2.0
    return np.stack(
        (
            sx * (clip_from_world[0] + clip_from_world[3]),
            sy * (clip_from_world[3] - clip_from_world[1]),
            clip_from_world[3],
        )
    )


def project_world(
    points_m: np.ndarray, camera: UnityCamera
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_m, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("world points must end with xyz")
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate(
        (flat, np.ones((len(flat), 1), dtype=np.float64)), axis=1
    )
    projected = homogeneous @ pixel_projection(camera).T
    pixels = np.full((len(flat), 2), np.nan, dtype=np.float64)
    good = np.isfinite(projected).all(axis=1) & (np.abs(projected[:, 2]) > 1e-12)
    pixels[good] = projected[good, :2] / projected[good, 2:3]
    camera_local = homogeneous @ np.linalg.inv(camera.camera_to_world).T
    depth = -camera_local[:, 2]
    return (
        pixels.reshape(points.shape[:-1] + (2,)).astype(np.float32),
        depth.reshape(points.shape[:-1]).astype(np.float32),
    )


def triangulate_pixels(
    cam0_pixels: np.ndarray,
    cam1_pixels: np.ndarray,
    cam0: UnityCamera,
    cam1: UnityCamera,
) -> np.ndarray:
    left = np.asarray(cam0_pixels, dtype=np.float64)
    right = np.asarray(cam1_pixels, dtype=np.float64)
    if left.shape != right.shape or left.shape[-1] != 2:
        raise ValueError("camera pixels must have equal shape ending in xy")
    flat_left = left.reshape(-1, 2)
    flat_right = right.reshape(-1, 2)
    valid = np.isfinite(flat_left).all(axis=1) & np.isfinite(flat_right).all(axis=1)
    output = np.full((len(flat_left), 3), np.nan, dtype=np.float32)
    if valid.any():
        homogeneous = cv2.triangulatePoints(
            pixel_projection(cam0),
            pixel_projection(cam1),
            flat_left[valid].T,
            flat_right[valid].T,
        )
        denominators = homogeneous[3]
        finite = np.isfinite(homogeneous).all(axis=0) & (np.abs(denominators) > 1e-12)
        indices = np.flatnonzero(valid)
        output[indices[finite]] = (
            homogeneous[:3, finite] / denominators[finite]
        ).T.astype(np.float32)
    return output.reshape(left.shape[:-1] + (3,))


def reprojection_errors(
    points_m: np.ndarray, pixels: np.ndarray, camera: UnityCamera
) -> np.ndarray:
    projected, _ = project_world(points_m, camera)
    target = np.asarray(pixels, dtype=np.float32)
    valid = np.isfinite(projected).all(axis=-1) & np.isfinite(target).all(axis=-1)
    errors = np.full(valid.shape, np.nan, dtype=np.float32)
    errors[valid] = np.linalg.norm(projected[valid] - target[valid], axis=-1)
    return errors


def _save_method_sequence(path: Path, sequence: MethodSequence) -> None:
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


def run_oracle_triangulation(
    benchmark: UnityBenchmark, output_root: Path
) -> tuple[MethodSequence, ...]:
    groups = group_evaluation_sequences(benchmark)
    outputs: list[MethodSequence] = []
    for sequence_id, frames in groups.items():
        cam0_pixels = np.stack([frame.gt_pixels["cam0"] for frame in frames])
        cam1_pixels = np.stack([frame.gt_pixels["cam1"] for frame in frames])
        available = np.stack([frame.gt_available for frame in frames])
        reconstructed = triangulate_pixels(
            cam0_pixels,
            cam1_pixels,
            benchmark.cameras["cam0"],
            benchmark.cameras["cam1"],
        )
        mapped = select_unity_evaluation_joints(reconstructed, available)
        gt = select_unity_evaluation_joints(
            np.stack([frame.gt_world_m for frame in frames]), available
        )
        common = mapped.valid & gt.valid
        raw_errors = np.linalg.norm(mapped.points - gt.points, axis=-1)
        reprojection = {
            camera_id: reprojection_errors(
                reconstructed,
                cam0_pixels if camera_id == "cam0" else cam1_pixels,
                camera,
            )
            for camera_id, camera in benchmark.cameras.items()
        }
        metadata = {
            "ranking_group": "diagnostic",
            "oracle": True,
            "raw_mpjpe_m": float(np.mean(raw_errors[common])),
            "raw_max_error_m": float(np.max(raw_errors[common])),
            "cam0_reprojection_mean_px": float(np.nanmean(reprojection["cam0"])),
            "cam0_reprojection_max_px": float(np.nanmax(reprojection["cam0"])),
            "cam1_reprojection_mean_px": float(np.nanmean(reprojection["cam1"])),
            "cam1_reprojection_max_px": float(np.nanmax(reprojection["cam1"])),
        }
        sequence = MethodSequence(
            method="triangulation_oracle2d",
            sequence_id=sequence_id,
            sample_ids=np.asarray([frame.sample_id for frame in frames]),
            points=mapped.points,
            valid=mapped.valid,
            joint_names=EVALUATION_JOINT_NAMES,
            metadata=metadata,
        )
        _save_method_sequence(
            Path(output_root) / "oracle2d" / f"{sequence_id}.npz", sequence
        )
        outputs.append(sequence)
    return tuple(outputs)
