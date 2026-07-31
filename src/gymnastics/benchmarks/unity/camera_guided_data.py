"""Unity sequences for fitted-camera self-supervised fusion without 3D targets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gymnastics.fusion.rotation_aware.camera import CameraConditioningConfig
from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.rotation_aware.dataset import (
    PosePairWindowDataset,
    SplitManifest,
    WindowConfig,
)
from gymnastics.fusion.rotation_aware.inference import (
    CanonicalTrial,
    canonicalize_trial,
)
from gymnastics.fusion.rotation_aware.schema import PosePairTrial

from .camera_features import (
    CameraFeatureSequence,
    FittedRelativeCamera,
    build_camera_feature_sequence,
    fit_relative_camera_from_training_2d,
)
from .dataset import group_evaluation_sequences
from .fusion import build_pose_pair_trial
from .geometry import pixel_projection
from .sam3d import load_sam3d_camera_cache
from .schema import UnityBenchmark
from .supervised_data import UnityFold


CAMERA_GUIDED_ABLATIONS = ("G0", "G1", "G2", "G3", "G4", "G5")


def camera_conditioning_config(
    ablation: str,
) -> CameraConditioningConfig | None:
    """Map the preregistered G-series cell to its model-side conditioner."""
    if ablation not in CAMERA_GUIDED_ABLATIONS:
        raise ValueError(f"unsupported camera-guided ablation: {ablation}")
    if ablation == "G0":
        return None
    return CameraConditioningConfig(
        global_channels=19,
        joint_channels=8,
        mode="film" if ablation in {"G4", "G5"} else "additive",
    )


@dataclass(frozen=True)
class UnityCameraGuidedSequence:
    """One synchronized sequence carrying inputs and camera features, never GT."""

    sequence_id: str
    sample_ids: np.ndarray
    raw_trial: PosePairTrial
    canonical_trial: CanonicalTrial
    pixels_2d: np.ndarray
    valid_2d: np.ndarray
    fitted_camera: FittedRelativeCamera | None
    camera_features: CameraFeatureSequence | None
    ablation: str

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids, dtype=np.int64)
        pixels = np.asarray(self.pixels_2d, dtype=np.float32)
        valid = np.asarray(self.valid_2d, dtype=bool)
        frames = len(sample_ids)
        if self.ablation not in CAMERA_GUIDED_ABLATIONS:
            raise ValueError("invalid camera-guided ablation")
        if sample_ids.shape != (frames,) or len(set(sample_ids.tolist())) != frames:
            raise ValueError("sample_ids must be unique with shape [T]")
        if pixels.shape != (frames, 2, 70, 2):
            raise ValueError("pixels_2d must have shape [T,2,70,2]")
        if valid.shape != (frames, 2, 70):
            raise ValueError("valid_2d must have shape [T,2,70]")
        if self.raw_trial.face.shape != (frames, 70, 3):
            raise ValueError("raw trial must have shape [T,70,3]")
        if self.canonical_trial.trial.face.shape != (frames, 70, 3):
            raise ValueError("canonical trial must have shape [T,70,3]")
        if self.ablation == "G0":
            if self.fitted_camera is not None or self.camera_features is not None:
                raise ValueError("G0 cannot carry camera features")
        elif self.fitted_camera is None or self.camera_features is None:
            raise ValueError("G1-G5 require fitted camera features")
        elif self.camera_features.joint_features.shape[:2] != (frames, 70):
            raise ValueError("camera features do not match the sequence")
        sample_ids = np.array(sample_ids, copy=True)
        pixels = np.array(pixels, copy=True)
        valid = np.array(valid, copy=True)
        sample_ids.setflags(write=False)
        pixels.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "pixels_2d", pixels)
        object.__setattr__(self, "valid_2d", valid)


@dataclass(frozen=True)
class _InputSequence:
    sequence_id: str
    sample_ids: np.ndarray
    raw_trial: PosePairTrial
    canonical_trial: CanonicalTrial
    pixels_2d: np.ndarray
    valid_2d: np.ndarray


def _camera_intrinsics(benchmark: UnityBenchmark) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for camera_id in ("cam0", "cam1"):
        projection = -pixel_projection(benchmark.cameras[camera_id]).astype(
            np.float64
        )
        intrinsics = cv2.decomposeProjectionMatrix(projection)[0]
        intrinsics /= intrinsics[2, 2]
        if (
            not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
        ):
            raise ValueError("Unity camera intrinsics cannot be decomposed")
        matrices.append(intrinsics)
    return np.stack(matrices)


def _load_input_sequences(
    benchmark: UnityBenchmark,
    sam3d_root: Path,
    *,
    skeleton_path: Path,
    fps: float,
) -> Mapping[str, _InputSequence]:
    groups = group_evaluation_sequences(benchmark)
    required = (
        "continuous_left_060_r00",
        "continuous_right_060_r00",
        "static_sweep",
    )
    missing = [name for name in required if name not in groups]
    if missing:
        raise ValueError(f"missing required Unity sequences: {missing}")
    skeleton = load_skeleton_spec(skeleton_path)
    output: dict[str, _InputSequence] = {}
    for sequence_id in required:
        frames = groups[sequence_id]
        sample_ids = np.asarray(
            [frame.sample_id for frame in frames], dtype=np.int64
        )
        cam0 = load_sam3d_camera_cache(sam3d_root, "cam0", sample_ids)
        cam1 = load_sam3d_camera_cache(sam3d_root, "cam1", sample_ids)
        raw = build_pose_pair_trial(
            sequence_id,
            sample_ids,
            cam0.points_3d,
            cam1.points_3d,
            cam0.valid_3d,
            cam1.valid_3d,
            fps=fps,
        )
        raw = replace(
            raw,
            source_metadata={
                **dict(raw.source_metadata),
                "camera_feature_training_input": True,
                "unity_native_3d_loaded": False,
            },
        )
        output[sequence_id] = _InputSequence(
            sequence_id=sequence_id,
            sample_ids=sample_ids,
            raw_trial=raw,
            canonical_trial=canonicalize_trial(raw, skeleton),
            pixels_2d=np.stack((cam0.points_2d, cam1.points_2d), axis=1),
            valid_2d=np.stack((cam0.valid_2d, cam1.valid_2d), axis=1),
        )
    return MappingProxyType(output)


def _perturb_camera(
    fitted: FittedRelativeCamera, *, degrees: float = 30.0
) -> FittedRelativeCamera:
    angle = np.deg2rad(float(degrees))
    perturbation = np.array(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    return FittedRelativeCamera(
        rotation_face_to_side=perturbation
        @ np.asarray(fitted.rotation_face_to_side),
        translation_direction_face_to_side=(
            perturbation
            @ np.asarray(fitted.translation_direction_face_to_side)
        ),
        inlier_ratio=fitted.inlier_ratio,
        holdout_reprojection_px=fitted.holdout_reprojection_px,
        fit_sample_ids=fitted.fit_sample_ids,
    )


def _feature_subset(
    features: CameraFeatureSequence, ablation: str
) -> CameraFeatureSequence:
    global_features = np.array(features.global_features, copy=True)
    joint_features = np.array(features.joint_features, copy=True)
    if ablation == "G1":
        global_features[-2:] = 0.0
        joint_features[:] = 0.0
    elif ablation == "G2":
        joint_features[:] = 0.0
    return CameraFeatureSequence(
        global_features=global_features,
        joint_features=joint_features,
        valid=features.valid,
        global_schema=features.global_schema,
        joint_schema=features.joint_schema,
    )


def build_camera_guided_sequences(
    benchmark: UnityBenchmark,
    sam3d_root: Path,
    *,
    skeleton_path: Path,
    fps: float,
    fold: UnityFold,
    ablation: str,
    threshold_px: float = 2.0,
) -> Mapping[str, UnityCameraGuidedSequence]:
    """Build one fold without copying Unity-native 3D into training contracts."""
    camera_conditioning_config(ablation)
    inputs = _load_input_sequences(
        benchmark,
        Path(sam3d_root),
        skeleton_path=Path(skeleton_path),
        fps=float(fps),
    )
    if fold.train_sequence not in inputs or fold.test_sequence not in inputs:
        raise ValueError("fold sequences are unavailable")
    if ablation == "G0":
        return MappingProxyType(
            {
                name: UnityCameraGuidedSequence(
                    sequence_id=value.sequence_id,
                    sample_ids=value.sample_ids,
                    raw_trial=value.raw_trial,
                    canonical_trial=value.canonical_trial,
                    pixels_2d=value.pixels_2d,
                    valid_2d=value.valid_2d,
                    fitted_camera=None,
                    camera_features=None,
                    ablation=ablation,
                )
                for name, value in inputs.items()
            }
        )
    intrinsics = _camera_intrinsics(benchmark)
    train = inputs[fold.train_sequence]
    fitted = fit_relative_camera_from_training_2d(
        train.pixels_2d,
        train.valid_2d,
        intrinsics,
        sample_ids=train.sample_ids,
        threshold_px=threshold_px,
    )
    if ablation == "G5":
        fitted = _perturb_camera(fitted)
    image_sizes = np.asarray(
        (
            benchmark.cameras["cam0"].image_size,
            benchmark.cameras["cam1"].image_size,
        ),
        dtype=np.float64,
    )
    output: dict[str, UnityCameraGuidedSequence] = {}
    for name, value in inputs.items():
        features = _feature_subset(
            build_camera_feature_sequence(
                fitted,
                value.pixels_2d,
                value.valid_2d,
                intrinsics,
                image_sizes=image_sizes,
            ),
            ablation,
        )
        output[name] = UnityCameraGuidedSequence(
            sequence_id=value.sequence_id,
            sample_ids=value.sample_ids,
            raw_trial=value.raw_trial,
            canonical_trial=value.canonical_trial,
            pixels_2d=value.pixels_2d,
            valid_2d=value.valid_2d,
            fitted_camera=fitted,
            camera_features=features,
            ablation=ablation,
        )
    return MappingProxyType(output)


class UnityCameraGuidedWindowDataset(Dataset[dict[str, object]]):
    """Attach fitted-camera tensors to existing canonical A6 windows."""

    def __init__(
        self,
        sequence: UnityCameraGuidedSequence,
        *,
        skeleton_path: Path,
        length: int,
        stride: int,
    ) -> None:
        skeleton = load_skeleton_spec(Path(skeleton_path))
        person_id = sequence.canonical_trial.trial.person_id
        self.sequence = sequence
        self._windows = PosePairWindowDataset(
            [sequence.canonical_trial.trial],
            skeleton=skeleton,
            manifest=SplitManifest(
                train=(person_id,),
                val=(),
                test=(),
            ),
            split="train",
            config=WindowConfig(
                length=length,
                train_stride=stride,
                eval_stride=stride,
            ),
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = dict(self._windows[index])
        features = self.sequence.camera_features
        if features is None:
            return sample
        frame_indices = sample["global_frame_index"]
        if not isinstance(frame_indices, torch.Tensor):
            raise TypeError("window global_frame_index must be a tensor")
        length = len(frame_indices)
        joint = torch.zeros((length, 70, 8), dtype=torch.float32)
        valid = torch.zeros((length, 70), dtype=torch.bool)
        present = frame_indices >= 0
        source = frame_indices[present].numpy()
        if len(source):
            joint[present] = torch.from_numpy(
                np.array(features.joint_features[source], copy=True)
            )
            valid[present] = torch.from_numpy(
                np.array(features.valid[source], copy=True)
            )
        sample["camera_global_features"] = torch.from_numpy(
            np.array(features.global_features, copy=True)
        )
        sample["camera_joint_features"] = joint
        sample["camera_valid"] = valid & present[:, None]
        return sample
