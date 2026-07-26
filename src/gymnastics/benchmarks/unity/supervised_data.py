"""Leakage-safe Unity-supervised sequence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.rotation_aware.inference import (
    CanonicalTrial,
    canonicalize_trial,
)
from gymnastics.fusion.rotation_aware.schema import PosePairTrial

from .dataset import group_evaluation_sequences
from .fusion import build_pose_pair_trial
from .mapping import select_unity_evaluation_joints
from .sam3d import load_sam3d_camera_cache
from .schema import UnityBenchmark


@dataclass(frozen=True)
class UnityFold:
    """One strict direction-transfer fold."""

    name: str
    train_sequence: str
    test_sequence: str


UNITY_SUPERVISED_FOLDS: Mapping[str, UnityFold] = MappingProxyType(
    {
        "left_to_right": UnityFold(
            "left_to_right",
            "continuous_left_060_r00",
            "continuous_right_060_r00",
        ),
        "right_to_left": UnityFold(
            "right_to_left",
            "continuous_right_060_r00",
            "continuous_left_060_r00",
        ),
    }
)


def _readonly(value: np.ndarray, *, dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class UnitySupervisedSequence:
    """One synchronized SAM3D pose pair joined to Unity16 ground truth."""

    sequence_id: str
    sample_ids: np.ndarray
    raw_trial: PosePairTrial
    canonical_trial: CanonicalTrial
    gt_unity16_m: np.ndarray
    gt_valid: np.ndarray

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids, dtype=np.int64)
        gt = np.asarray(self.gt_unity16_m, dtype=np.float32)
        valid = np.asarray(self.gt_valid, dtype=bool)
        frames = len(sample_ids)
        if not self.sequence_id:
            raise ValueError("sequence_id is required")
        if sample_ids.shape != (frames,):
            raise ValueError("sample_ids must have shape [T]")
        if len(set(sample_ids.tolist())) != frames:
            raise ValueError("sample_ids must be unique")
        if self.raw_trial.face.shape != (frames, 70, 3):
            raise ValueError("raw trial must have shape [T,70,3]")
        if self.canonical_trial.trial.face.shape != (frames, 70, 3):
            raise ValueError("canonical trial must have shape [T,70,3]")
        if gt.shape != (frames, 16, 3):
            raise ValueError("gt_unity16_m must have shape [T,16,3]")
        if valid.shape != (frames, 16):
            raise ValueError("gt_valid must have shape [T,16]")
        if self.raw_trial.trial_id != self.sequence_id:
            raise ValueError("raw trial identity does not match sequence")
        if self.canonical_trial.trial.trial_id != self.sequence_id:
            raise ValueError("canonical trial identity does not match sequence")
        object.__setattr__(
            self, "sample_ids", _readonly(sample_ids, dtype=np.int64)
        )
        object.__setattr__(
            self, "gt_unity16_m", _readonly(gt, dtype=np.float32)
        )
        object.__setattr__(self, "gt_valid", _readonly(valid, dtype=bool))


def build_supervised_sequence(
    benchmark: UnityBenchmark,
    sam3d_root: Path,
    sequence_id: str,
    *,
    skeleton_path: Path,
    fps: float,
) -> UnitySupervisedSequence:
    """Join one exact evaluation sequence to cached SAM3D inputs."""
    groups = group_evaluation_sequences(benchmark)
    if sequence_id not in groups:
        raise KeyError(f"Unity sequence is unavailable: {sequence_id}")
    frames = groups[sequence_id]
    sample_ids = np.asarray(
        [frame.sample_id for frame in frames], dtype=np.int64
    )
    cam0 = load_sam3d_camera_cache(sam3d_root, "cam0", sample_ids)
    cam1 = load_sam3d_camera_cache(sam3d_root, "cam1", sample_ids)
    if not np.array_equal(cam0.sample_ids, sample_ids):
        raise ValueError("cam0 cache sample order does not match Unity")
    if not np.array_equal(cam1.sample_ids, sample_ids):
        raise ValueError("cam1 cache sample order does not match Unity")
    raw_trial = build_pose_pair_trial(
        sequence_id,
        sample_ids,
        cam0.points_3d,
        cam1.points_3d,
        cam0.valid_3d,
        cam1.valid_3d,
        fps=fps,
    )
    canonical_trial = canonicalize_trial(
        raw_trial, load_skeleton_spec(Path(skeleton_path))
    )
    gt = select_unity_evaluation_joints(
        np.stack([frame.gt_world_m for frame in frames]),
        np.stack([frame.gt_available for frame in frames]),
    )
    return UnitySupervisedSequence(
        sequence_id=sequence_id,
        sample_ids=sample_ids,
        raw_trial=raw_trial,
        canonical_trial=canonical_trial,
        gt_unity16_m=gt.points,
        gt_valid=gt.valid,
    )


def build_supervised_sequences(
    benchmark: UnityBenchmark,
    sam3d_root: Path,
    *,
    skeleton_path: Path,
    fps: float,
) -> Mapping[str, UnitySupervisedSequence]:
    """Build the two directions and static diagnostic sequence."""
    groups = group_evaluation_sequences(benchmark)
    required = (
        "continuous_left_060_r00",
        "continuous_right_060_r00",
        "static_sweep",
    )
    missing = [sequence_id for sequence_id in required if sequence_id not in groups]
    if missing:
        raise ValueError(f"missing required Unity sequences: {missing}")
    return MappingProxyType(
        {
            sequence_id: build_supervised_sequence(
                benchmark,
                sam3d_root,
                sequence_id,
                skeleton_path=skeleton_path,
                fps=fps,
            )
            for sequence_id in required
        }
    )


def select_supervised_fold(
    sequences: Mapping[str, UnitySupervisedSequence],
    fold: UnityFold,
) -> tuple[
    UnitySupervisedSequence,
    UnitySupervisedSequence,
    UnitySupervisedSequence,
]:
    train = sequences[fold.train_sequence]
    test = sequences[fold.test_sequence]
    static = sequences["static_sweep"]
    audit_fold_isolation(fold, train, test, static)
    return train, test, static


def audit_fold_isolation(
    fold: UnityFold,
    train: UnitySupervisedSequence,
    test: UnitySupervisedSequence,
    static: UnitySupervisedSequence,
) -> None:
    """Reject sequence identity, sample overlap, or coverage leakage."""
    if train.sequence_id != fold.train_sequence:
        raise ValueError("training sequence does not match fold")
    if test.sequence_id != fold.test_sequence:
        raise ValueError("test sequence does not match fold")
    if static.sequence_id != "static_sweep":
        raise ValueError("static diagnostic sequence is missing")
    groups = (
        set(train.sample_ids.tolist()),
        set(test.sample_ids.tolist()),
        set(static.sample_ids.tolist()),
    )
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("sample leakage across Unity supervised fold")
    if tuple(map(len, groups)) != (97, 97, 5):
        raise ValueError("unexpected Unity supervised fold sizes")
