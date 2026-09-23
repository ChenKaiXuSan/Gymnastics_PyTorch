"""Read the external per-video SAM3D-Body cache of the Fit3D recordings.

The predictions are produced outside this repository (the user's
``derived/normal_camera`` jobs) as one file per subject, camera and
exercise::

    <derived_root>/<split>/<subject>/<camera>/<action>.npz

with ``frame_ids``, ``person_rank``, ``points3d [N, 70, 3]``,
``points2d [N, 70, 2]``, ``valid3d``, ``valid2d`` and ``n_frames``. Every
frame of every camera is present, so this module only selects the rank-0
person and the requested frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import Fit3DSequence, ViewPrediction


def cache_path(derived_root: Path, sequence: Fit3DSequence, camera: str, *, split: str = "train") -> Path:
    """Location of one camera's predictions for one sequence."""
    return Path(derived_root) / split / sequence.subject / str(camera) / f"{sequence.action}.npz"


def cached_frames(derived_root: Path, sequence: Fit3DSequence, camera: str, *, split: str = "train") -> int:
    """Number of video frames the cache covers (0 when the file is missing)."""
    path = cache_path(derived_root, sequence, camera, split=split)
    if not path.is_file():
        return 0
    with np.load(path, allow_pickle=True) as payload:
        return int(payload["n_frames"])


def load_prediction(
    derived_root: Path,
    sequence: Fit3DSequence,
    camera: str,
    frame_ids: np.ndarray,
    *,
    split: str = "train",
) -> ViewPrediction:
    """Rank-0 person of one camera at ``frame_ids``.

    Frames the estimator did not produce (no detection) are returned as
    invalid rather than dropped, so both views and the reference stay on one
    frame grid.

    Raises:
        FileNotFoundError: If the cache file is missing.
    """
    path = cache_path(derived_root, sequence, camera, split=split)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as payload:
        rows_frames = np.asarray(payload["frame_ids"], dtype=np.int64)
        ranks = np.asarray(payload["person_rank"], dtype=np.int64)
        points_3d_all = np.asarray(payload["points3d"], dtype=np.float32)
        points_2d_all = np.asarray(payload["points2d"], dtype=np.float32)
        valid_3d_all = np.asarray(payload["valid3d"], dtype=bool)
        valid_2d_all = np.asarray(payload["valid2d"], dtype=bool)
    primary = ranks == 0
    row_of = {int(frame): int(row) for row, frame in zip(np.flatnonzero(primary), rows_frames[primary])}
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    count = len(frame_ids)
    points_3d = np.zeros((count, 70, 3), dtype=np.float32)
    points_2d = np.zeros((count, 70, 2), dtype=np.float32)
    valid_3d = np.zeros((count, 70), dtype=bool)
    valid_2d = np.zeros((count, 70), dtype=bool)
    failed: list[int] = []
    for slot, frame in enumerate(frame_ids):
        row = row_of.get(int(frame))
        if row is None:
            failed.append(int(frame))
            continue
        xyz, xy = points_3d_all[row], points_2d_all[row]
        ok_3d = valid_3d_all[row] & np.isfinite(xyz).all(axis=-1) & np.any(xyz != 0, axis=-1)
        ok_2d = valid_2d_all[row] & np.isfinite(xy).all(axis=-1)
        points_3d[slot] = np.where(ok_3d[:, None], xyz, 0.0)
        points_2d[slot] = np.where(ok_2d[:, None], xy, 0.0)
        valid_3d[slot] = ok_3d
        valid_2d[slot] = ok_2d
    return ViewPrediction(
        sequence_key=sequence.sequence_key,
        view_id=str(camera),
        frame_ids=frame_ids,
        points_3d=points_3d,
        points_2d=points_2d,
        valid_3d=valid_3d,
        valid_2d=valid_2d,
        failed_frames=tuple(failed),
    )
