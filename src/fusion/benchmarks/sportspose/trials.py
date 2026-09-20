"""Build ``PosePairTrial`` sequences from the cached SportsPose predictions.

One sequence per subject, day and activity: the clips (trials of the same action)
are concatenated in clip-id order on a synthetic timeline, so that each clip
plays the role of one movement cycle for the cycle-aware model. Frame ids
of view A and view B are identical (hardware-synchronised cameras).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from common.skeletons.mhr70 import mhr_names
from fusion.keypoints.schema import PosePairTrial

from .dataset import load_calibration, load_reference, load_timing
from .sam3d import ViewPrediction, cache_path, load_derived_prediction, load_prediction
from .schema import NATIVE_FPS, SelectedViews, SportsPoseClip


def load_clip_predictions(
    clip: SportsPoseClip,
    views: SelectedViews,
    *,
    cache_root: Path | None = None,
    derived_root: Path | None = None,
    frame_stride: int = 3,
) -> tuple[ViewPrediction, ViewPrediction]:
    """The two selected views of one clip from either SAM3D source.

    ``derived_root`` (the external per-video cache, every camera and frame)
    takes precedence when given; the requested frames are then every
    ``frame_stride``-th reference frame. Otherwise the benchmark cache below
    ``cache_root`` is read as written by ``benchmark-sportspose infer``.
    """
    if derived_root is not None:
        cameras = load_calibration(clip.joints_path.parents[1])
        timing = load_timing(clip)
        frame_ids = np.arange(0, clip.frames, max(1, int(frame_stride)), dtype=np.int64)
        a = load_derived_prediction(derived_root, clip, cameras[views.view_a], frame_ids, timing["video_index"][cameras[views.view_a].index][frame_ids])
        b = load_derived_prediction(derived_root, clip, cameras[views.view_b], frame_ids, timing["video_index"][cameras[views.view_b].index][frame_ids])
    elif cache_root is not None:
        a = load_prediction(cache_path(cache_root, clip, views.view_a))
        b = load_prediction(cache_path(cache_root, clip, views.view_b))
    else:
        raise ValueError("either derived_root or cache_root is required")
    if not np.array_equal(a.frame_ids, b.frame_ids):
        raise ValueError(f"{clip.clip_id}: cached views cover different frames")
    return a, b


def build_sequence_trial(
    clips: Sequence[SportsPoseClip],
    predictions: Sequence[tuple[ViewPrediction, ViewPrediction]],
    views: SelectedViews,
    *,
    reference: bool = False,
) -> tuple[PosePairTrial, tuple[tuple[int, int], ...], np.ndarray | None]:
    """Concatenate the clips of one subject/day/activity into one trial.

    Returns:
        ``(trial, clip_bounds, reference)``; ``reference`` is the ``[T, 17, 3]``
        COCO17 sequence sampled at the cached frames when requested.
    """
    if len(clips) != len(predictions) or not clips:
        raise ValueError("one prediction pair per clip is required")
    faces, sides, valid_faces, valid_sides, refs = [], [], [], [], []
    bounds: list[tuple[int, int]] = []
    offset = 0
    fps = None
    for clip, (pred_a, pred_b) in zip(clips, predictions):
        n = len(pred_a.frame_ids)
        stride = int(pred_a.frame_ids[1] - pred_a.frame_ids[0]) if n > 1 else 1
        clip_fps = NATIVE_FPS / stride
        if fps is None:
            fps = clip_fps
        elif not np.isclose(fps, clip_fps):
            raise ValueError(f"{clip.clip_id}: frame stride differs from the other clips")
        faces.append(pred_a.points_3d)
        sides.append(pred_b.points_3d)
        valid_faces.append(pred_a.valid_3d)
        valid_sides.append(pred_b.valid_3d)
        if reference:
            refs.append(load_reference(clip)[pred_a.frame_ids])
        bounds.append((offset, offset + n))
        offset += n
    frame_ids = np.arange(offset, dtype=np.int32)
    trial = PosePairTrial(
        face=np.concatenate(faces).astype(np.float32),
        side=np.concatenate(sides).astype(np.float32),
        valid_face=np.concatenate(valid_faces),
        valid_side=np.concatenate(valid_sides),
        timestamps=frame_ids.astype(np.float64) / float(fps),
        face_map=frame_ids,
        side_map=frame_ids,
        joint_names=tuple(mhr_names),
        person_id=views.subject_key,
        trial_id=views.sequence_key,
        fps=float(fps),
        source_metadata={
            "dataset": "sportspose",
            "day": clips[0].day,
            "activity": clips[0].activity,
            "camera_reference": views.view_a,
            "view_a": views.view_a,
            "view_b": views.view_b,
            "offset_side_to_face": 0,
            "clips": [clip.clip_id for clip in clips],
            "clip_bounds": [list(b) for b in bounds],
        },
    )
    return trial, tuple(bounds), (np.concatenate(refs) if reference else None)
