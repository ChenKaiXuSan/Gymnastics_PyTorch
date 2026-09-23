"""Build ``PosePairTrial`` sequences from the cached Fit3D predictions.

One sequence per subject and exercise. The four cameras are hardware
synchronised, so view A and view B share the frame grid; the grid is the
frames both caches and the reference cover, thinned by ``frame_stride``.
Movement cycles are the dataset's own repetition annotations, not a
detector's output (see :func:`fusion.benchmarks.fit3d.dataset.repetition_bounds`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.skeletons.mhr70 import mhr_names
from fusion.keypoints.schema import PosePairTrial

from .dataset import load_reference
from .sam3d import cached_frames, load_prediction
from .schema import NATIVE_FPS, Fit3DSequence, SelectedViews, ViewPrediction


def common_frames(
    derived_root: Path,
    sequence: Fit3DSequence,
    views: SelectedViews,
    *,
    frame_stride: int = 1,
    split: str = "train",
) -> np.ndarray:
    """Frame grid covered by the reference and both cached views."""
    limits = [int(sequence.frames)]
    for camera in (views.view_a, views.view_b):
        frames = cached_frames(derived_root, sequence, camera, split=split)
        if frames <= 0:
            raise FileNotFoundError(f"missing SAM3D cache for {sequence.subject}/{sequence.action}/{camera}")
        limits.append(frames)
    return np.arange(0, min(limits), max(1, int(frame_stride)), dtype=np.int64)


def build_trial(
    sequence: Fit3DSequence,
    views: SelectedViews,
    predictions: tuple[ViewPrediction, ViewPrediction],
    *,
    frame_ids: np.ndarray,
    frame_stride: int = 1,
) -> PosePairTrial:
    """One subject/exercise as a dual-view trial on the common frame grid."""
    view_a, view_b = predictions
    if not np.array_equal(view_a.frame_ids, view_b.frame_ids):
        raise ValueError(f"{sequence.sequence_key}: the two views cover different frames")
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    fps = NATIVE_FPS / max(1, int(frame_stride))
    return PosePairTrial(
        face=view_a.points_3d.astype(np.float32),
        side=view_b.points_3d.astype(np.float32),
        valid_face=view_a.valid_3d,
        valid_side=view_b.valid_3d,
        timestamps=frame_ids.astype(np.float64) / NATIVE_FPS,
        face_map=frame_ids,
        side_map=frame_ids,
        joint_names=tuple(mhr_names),
        person_id=sequence.subject_key,
        trial_id=sequence.sequence_key,
        fps=float(fps),
        source_metadata={
            "dataset": "fit3d",
            "action": sequence.action,
            "camera_reference": views.view_a,
            "view_a": views.view_a,
            "view_b": views.view_b,
            "separation_deg": float(views.separation_deg),
            "offset_side_to_face": 0,
            "frame_stride": int(frame_stride),
        },
    )


def load_sequence(
    sequence: Fit3DSequence,
    views: SelectedViews,
    derived_root: Path,
    *,
    frame_stride: int = 1,
    split: str = "train",
    reference: bool = False,
) -> tuple[PosePairTrial, np.ndarray | None]:
    """Trial of one sequence and, optionally, its reference on the same frames."""
    frame_ids = common_frames(derived_root, sequence, views, frame_stride=frame_stride, split=split)
    pair = tuple(load_prediction(derived_root, sequence, camera, frame_ids, split=split) for camera in (views.view_a, views.view_b))
    trial = build_trial(sequence, views, pair, frame_ids=frame_ids, frame_stride=frame_stride)
    reference_points = load_reference(sequence.reference_path)[frame_ids] if reference else None
    return trial, reference_points


def scale_bounds(bounds: tuple[tuple[int, int], ...], frame_ids: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Map video-frame cycle bounds onto positions of the thinned frame grid."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    scaled: list[tuple[int, int]] = []
    for start, end in bounds:
        first = int(np.searchsorted(frame_ids, start, side="left"))
        last = int(np.searchsorted(frame_ids, end, side="left"))
        if last - first >= 2:
            scaled.append((first, last))
    return tuple(scaled)
