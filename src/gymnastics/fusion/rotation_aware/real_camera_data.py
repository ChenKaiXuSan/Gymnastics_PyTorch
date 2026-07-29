"""Camera-feature adapters for the collected two-view gymnastics dataset.

The camera fit is estimated from SAM3D inputs and is never populated from the
triangulated evaluation reference.  G0 remains camera-free; G1--G5 follow the
camera-guided ablations used by the Unity benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gymnastics.benchmarks.unity.camera_features import (
    CameraFeatureSequence,
    FittedRelativeCamera,
    build_camera_feature_sequence,
)
from gymnastics.fusion.rotation_aware.inference import CanonicalTrial, canonicalize_trial
from gymnastics.fusion.rotation_aware.schema import PosePairTrial
from gymnastics.triangulation.sam3d_from_split_cycle import (
    load_calibration,
    load_keypoints_2d,
)


CAMERA_ABLATIONS = frozenset({"G0", "G1", "G2", "G3", "G4", "G5"})


@dataclass(frozen=True)
class PersonCameraFit:
    """Input-level relative-camera estimate and its audit diagnostics."""

    person_id: str
    fitted: FittedRelativeCamera
    rig_cluster: int
    method: str
    num_frames: int
    bone_cv_pct: float


@dataclass(frozen=True)
class RealCameraTrial:
    """Canonical real-data trial with optional frame-aligned camera features."""

    canonical_trial: CanonicalTrial
    camera_fit: PersonCameraFit | None
    camera_features: CameraFeatureSequence | None
    ablation: str

    def __post_init__(self) -> None:
        if self.ablation not in CAMERA_ABLATIONS:
            raise ValueError(f"Unsupported camera ablation: {self.ablation}")
        if self.ablation == "G0":
            if self.camera_fit is not None or self.camera_features is not None:
                raise ValueError("G0 must not contain camera features")
            return
        if self.camera_fit is None or self.camera_features is None:
            raise ValueError(f"{self.ablation} requires a camera fit and features")
        expected = self.canonical_trial.trial.face.shape[0]
        if self.camera_features.joint_features.shape[0] != expected:
            raise ValueError("Camera feature length does not match canonical trial")


def _load_audit(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    persons = payload.get("persons")
    if not isinstance(persons, Mapping):
        raise ValueError(f"Camera audit has no 'persons' mapping: {path}")
    return persons


def _normalise_translation(value: Any) -> np.ndarray:
    translation = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(translation))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Camera translation direction must be finite and non-zero")
    return translation / norm


def _fitted_camera(person_id: str, record: Mapping[str, Any]) -> PersonCameraFit:
    sample_count = max(int(record.get("num_frames", 0)), 1)
    camera = FittedRelativeCamera(
        rotation_face_to_side=np.asarray(record["R"], dtype=np.float64).reshape(3, 3),
        translation_direction_face_to_side=_normalise_translation(record["t"]),
        inlier_ratio=float(record.get("inlier_ratio", float("nan"))),
        holdout_reprojection_px=float(
            record.get("holdout_reproj_px", float("nan"))
        ),
        fit_sample_ids=np.arange(sample_count, dtype=np.int64),
    )
    return PersonCameraFit(
        person_id=str(person_id),
        fitted=camera,
        rig_cluster=int(record.get("rig_cluster", -1)),
        method=str(record.get("method", "unknown")),
        num_frames=int(record.get("num_frames", 0)),
        bone_cv_pct=float(record.get("bone_cv_pct", float("nan"))),
    )


def _perturb_camera(camera: FittedRelativeCamera, degrees: float = 30.0) -> FittedRelativeCamera:
    angle = np.deg2rad(float(degrees))
    perturbation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    return FittedRelativeCamera(
        rotation_face_to_side=perturbation
        @ np.asarray(camera.rotation_face_to_side),
        translation_direction_face_to_side=perturbation
        @ np.asarray(camera.translation_direction_face_to_side),
        inlier_ratio=camera.inlier_ratio,
        holdout_reprojection_px=camera.holdout_reprojection_px,
        fit_sample_ids=camera.fit_sample_ids.copy(),
    )


def _feature_subset(
    features: CameraFeatureSequence,
    ablation: str,
) -> CameraFeatureSequence:
    global_features = features.global_features.copy()
    joint_features = features.joint_features.copy()
    if ablation == "G1":
        global_features[..., -2:] = 0.0
        joint_features[...] = 0.0
    elif ablation == "G2":
        joint_features[...] = 0.0
    return CameraFeatureSequence(
        global_features=global_features,
        joint_features=joint_features,
        valid=features.valid.copy(),
        global_schema=features.global_schema,
        joint_schema=features.joint_schema,
    )


def _frame_path(
    sam3d_person_root: str | Path,
    person_id: str,
    view: str,
    frame_index: int,
) -> Path:
    return (
        Path(sam3d_person_root)
        / str(person_id)
        / view
        / f"{int(frame_index):06d}_sam3d_body.npz"
    )


@lru_cache(maxsize=200_000)
def _cached_keypoints_2d(path: str) -> np.ndarray:
    points = np.asarray(load_keypoints_2d(Path(path)), dtype=np.float64)
    points.setflags(write=False)
    return points


def _load_view_pixels(
    sam3d_person_root: str | Path,
    person_id: str,
    view: str,
    frame_map: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for frame_index in frame_map:
        path = _frame_path(sam3d_person_root, person_id, view, int(frame_index))
        if not path.is_file():
            raise FileNotFoundError(f"Missing SAM3D frame: {path}")
        points = np.array(_cached_keypoints_2d(str(path)), copy=True)
        points = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64),
            P=np.asarray(camera_matrix, dtype=np.float64),
        ).reshape(-1, 2)
        frames.append(points)
    return np.stack(frames, axis=0)


def load_real_camera_trials(
    *,
    raw_trials: Sequence[PosePairTrial | CanonicalTrial],
    skeleton: Any,
    sam3d_person_root: str | Path,
    camera_audit_path: str | Path,
    face_calibration_path: str | Path,
    side_calibration_path: str | Path,
    ablation: str,
) -> list[RealCameraTrial]:
    """Attach per-person fitted-camera features to canonical real trials."""

    if ablation not in CAMERA_ABLATIONS:
        raise ValueError(f"Unsupported camera ablation: {ablation}")

    canonical_trials = [
        trial if isinstance(trial, CanonicalTrial) else canonicalize_trial(trial, skeleton)
        for trial in raw_trials
    ]
    if ablation == "G0":
        return [
            RealCameraTrial(
                canonical_trial=trial,
                camera_fit=None,
                camera_features=None,
                ablation=ablation,
            )
            for trial in canonical_trials
        ]

    audit = _load_audit(camera_audit_path)
    face_calibration = load_calibration(face_calibration_path)
    side_calibration = load_calibration(side_calibration_path)
    intrinsics = np.stack([face_calibration["K"], side_calibration["K"]], axis=0)
    image_sizes = np.stack(
        [
            np.asarray(face_calibration["image_size"], dtype=np.float64),
            np.asarray(side_calibration["image_size"], dtype=np.float64),
        ],
        axis=0,
    )

    output: list[RealCameraTrial] = []
    for canonical in canonical_trials:
        raw = canonical.trial
        person_id = str(raw.person_id)
        if person_id not in audit:
            raise KeyError(f"No fitted-camera audit for person {person_id}")
        person_fit = _fitted_camera(person_id, audit[person_id])
        feature_camera = (
            _perturb_camera(person_fit.fitted)
            if ablation == "G5"
            else person_fit.fitted
        )
        if ablation == "G5":
            person_fit = PersonCameraFit(
                person_id=person_fit.person_id,
                fitted=feature_camera,
                rig_cluster=person_fit.rig_cluster,
                method=person_fit.method,
                num_frames=person_fit.num_frames,
                bone_cv_pct=person_fit.bone_cv_pct,
            )

        face_pixels = _load_view_pixels(
            sam3d_person_root,
            person_id,
            "face",
            np.asarray(raw.face_map, dtype=np.int64),
            face_calibration["K"],
            face_calibration["dist"],
        )
        side_pixels = _load_view_pixels(
            sam3d_person_root,
            person_id,
            "side",
            np.asarray(raw.side_map, dtype=np.int64),
            side_calibration["K"],
            side_calibration["dist"],
        )
        pixels = np.stack([face_pixels, side_pixels], axis=1)
        valid = np.isfinite(pixels).all(axis=-1) & (np.linalg.norm(pixels, axis=-1) > 0)
        features = build_camera_feature_sequence(
            pixels=pixels,
            valid=valid,
            intrinsics=intrinsics,
            image_sizes=image_sizes,
            fitted=feature_camera,
        )
        features = _feature_subset(features, ablation)
        output.append(
            RealCameraTrial(
                canonical_trial=canonical,
                camera_fit=person_fit,
                camera_features=features,
                ablation=ablation,
            )
        )
    return output


class CameraWindowDataset(Dataset):
    """Augment a rotation-aware dataset with padded camera features."""

    def __init__(
        self,
        base_dataset: Dataset,
        camera_trials: Sequence[RealCameraTrial],
    ) -> None:
        self.base_dataset = base_dataset
        self._features = {
            (
                trial.canonical_trial.trial.person_id,
                trial.canonical_trial.trial.trial_id,
            ): trial.camera_features
            for trial in camera_trials
        }

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        person_id = str(sample["person_id"])
        trial_id = str(sample["trial_id"])
        features = self._features[(person_id, trial_id)]
        if features is None:
            return sample

        frame_indices = np.asarray(sample["global_frame_index"], dtype=np.int64)
        window_length = frame_indices.shape[0]
        global_dim = features.global_features.shape[-1]
        joint_count, joint_dim = features.joint_features.shape[1:]
        joint_features = np.zeros(
            (window_length, joint_count, joint_dim), dtype=np.float32
        )
        valid = np.zeros((window_length, joint_count), dtype=bool)
        in_range = (frame_indices >= 0) & (
            frame_indices < features.global_features.shape[0]
        )
        destination = np.flatnonzero(in_range)
        source = frame_indices[in_range]
        joint_features[destination] = features.joint_features[source]
        valid[destination] = features.valid[source]
        sample["camera_global_features"] = torch.from_numpy(
            np.array(features.global_features, copy=True)
        )
        sample["camera_joint_features"] = torch.from_numpy(joint_features)
        sample["camera_valid"] = torch.from_numpy(valid)
        return sample


class CameraCompleteCycleDataset(CameraWindowDataset):
    """Semantic alias for complete-cycle evaluation datasets."""
