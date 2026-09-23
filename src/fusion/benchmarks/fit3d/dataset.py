"""Discover Fit3D sequences, read their calibration, reference and repetitions.

Only the release itself is read here (videos are never opened); the SAM3D
predictions come from :mod:`fusion.benchmarks.fit3d.sam3d`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .schema import CAMERAS, TRAIN_SUBJECTS, Fit3DCamera, Fit3DSequence, SelectedViews


def discover_sequences(
    dataset_root: Path,
    *,
    split: str = "train",
    subjects: Sequence[str] | None = None,
    actions: Sequence[str] | None = None,
) -> tuple[Fit3DSequence, ...]:
    """Every subject/exercise of ``split`` that has a 3D reference.

    Args:
        dataset_root: Fit3D release root (holding ``train/`` and ``test/``).
        split: Release split; only ``train`` carries ``joints3d_25``.
        subjects: Optional subject ids (``s03`` ...); default: all.
        actions: Optional exercise names; default: all.

    Returns:
        Sequences sorted by subject and action.
    """
    root = Path(dataset_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"missing Fit3D split: {root}")
    wanted_subjects = {str(s) for s in subjects} if subjects else None
    wanted_actions = {str(a) for a in actions} if actions else None
    found: list[Fit3DSequence] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject = subject_dir.name
        if wanted_subjects is not None and subject not in wanted_subjects:
            continue
        reference_dir = subject_dir / "joints3d_25"
        if not reference_dir.is_dir():
            continue  # test subjects have no ground truth
        for reference_path in sorted(reference_dir.glob("*.json")):
            action = reference_path.stem
            if wanted_actions is not None and action not in wanted_actions:
                continue
            frames = len(load_reference(reference_path))
            cameras = tuple(c for c in CAMERAS if (subject_dir / "camera_parameters" / c / f"{action}.json").is_file())
            if len(cameras) < 2:
                continue
            found.append(Fit3DSequence(subject=subject, action=action, root=subject_dir, frames=frames, cameras=cameras))
    return tuple(found)


def load_reference(path: Path) -> np.ndarray:
    """``[T, 25, 3]`` reference joints in metres (world frame)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    points = np.asarray(payload["joints3d_25"], dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (25, 3):
        raise ValueError(f"{path}: expected [T, 25, 3] reference joints")
    return points


def load_camera(sequence: Fit3DSequence, camera: str) -> Fit3DCamera:
    """Calibration of one camera for one sequence."""
    payload = json.loads(sequence.camera_path(camera).read_text(encoding="utf-8"))
    extrinsics, intrinsics = payload["extrinsics"], payload["intrinsics_wo_distortion"]
    return Fit3DCamera(
        camera_id=str(camera),
        rotation=np.asarray(extrinsics["R"], dtype=np.float64).reshape(3, 3),
        center=np.asarray(extrinsics["T"], dtype=np.float64).reshape(3),
        focal=np.asarray(intrinsics["f"], dtype=np.float64).reshape(2),
        principal_point=np.asarray(intrinsics["c"], dtype=np.float64).reshape(2),
    )


def load_cameras(sequence: Fit3DSequence) -> dict[str, Fit3DCamera]:
    """Calibration of every camera of one sequence."""
    return {camera: load_camera(sequence, camera) for camera in sequence.cameras}


def load_repetitions(subject_root: Path) -> Mapping[str, tuple[int, ...]]:
    """``rep_ann.json`` of one subject: exercise -> repetition marks."""
    path = Path(subject_root) / "rep_ann.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(action): tuple(int(f) for f in marks) for action, marks in payload.items()}


def repetition_bounds(marks: Iterable[int], frames: int) -> tuple[tuple[int, int], ...]:
    """Consecutive repetition marks as half-open ``(start, end)`` cycles.

    ``rep_ann`` stores the frame of every repetition boundary, so ``n`` marks
    describe ``n - 1`` complete repetitions. Marks outside the sequence and
    non-increasing pairs are dropped.
    """
    ordered = sorted({int(m) for m in marks if 0 <= int(m) < int(frames)})
    return tuple((start, end) for start, end in zip(ordered, ordered[1:]) if end > start)


def body_azimuth(camera: Fit3DCamera, reference: np.ndarray, up_axis: int = 2) -> float:
    """Horizontal angle (degrees) of the camera around the subject.

    The subject's mean position over the sequence is the origin and the
    camera centre is projected onto the horizontal plane, so the angle is
    comparable between cameras of the same recording.
    """
    plane = [axis for axis in range(3) if axis != int(up_axis)]
    root = np.asarray(reference, dtype=np.float64)[:, 0].mean(axis=0)
    offset = camera.center - root
    return float(np.degrees(np.arctan2(offset[plane[1]], offset[plane[0]])))


def facing_azimuth(reference: np.ndarray, up_axis: int = 2, left_hip: int = 1, right_hip: int = 4) -> float:
    """Mean horizontal direction the subject faces, in degrees.

    The forward direction of a frame is ``up x (right_hip - left_hip)``; the
    mean over the sequence gives the facing direction of the recording, which
    defines which camera plays the "face" role.
    """
    points = np.asarray(reference, dtype=np.float64)
    up = np.zeros(3)
    up[int(up_axis)] = 1.0
    forward = np.cross(up, points[:, int(right_hip)] - points[:, int(left_hip)])
    mean = forward.mean(axis=0)
    plane = [axis for axis in range(3) if axis != int(up_axis)]
    return float(np.degrees(np.arctan2(mean[plane[1]], mean[plane[0]])))


def _angle_difference(first: float, second: float) -> float:
    """Absolute difference of two angles in degrees, in ``[0, 180]``."""
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def select_views(
    sequence: Fit3DSequence,
    reference: np.ndarray,
    *,
    target_separation_deg: float = 90.0,
    up_axis: int = 2,
    reps: int = 0,
) -> SelectedViews:
    """Choose the face/side camera pair of one sequence.

    ``view_a`` is the camera closest to the direction the subject faces
    (:func:`facing_azimuth`) and ``view_b`` the camera whose azimuth is
    closest to ``target_separation_deg`` away from it, exactly the roles of
    the private face/side rig.  The four Fit3D cameras sit at roughly
    ``+-25`` and ``+-157`` degrees, so the reachable separations are about
    46, 132 and 178 degrees and the selected pair is normally the 132-degree
    one; calibration is used for this choice only and never by the fusion
    methods.
    """
    cameras = load_cameras(sequence)
    angles = {name: body_azimuth(camera, reference, up_axis=up_axis) for name, camera in cameras.items()}
    facing = facing_azimuth(reference, up_axis=up_axis)
    view_a = min(sorted(angles), key=lambda name: _angle_difference(angles[name], facing))
    others = [name for name in sorted(angles) if name != view_a]
    view_b = min(others, key=lambda name: abs(_angle_difference(angles[name], angles[view_a]) - float(target_separation_deg)))
    return SelectedViews(
        subject_key=sequence.subject_key,
        sequence_key=sequence.sequence_key,
        view_a=view_a,
        view_b=view_b,
        azimuth_a_deg=angles[view_a],
        azimuth_b_deg=angles[view_b],
        separation_deg=_angle_difference(angles[view_a], angles[view_b]),
        frames=int(sequence.frames),
        reps=int(reps),
    )
