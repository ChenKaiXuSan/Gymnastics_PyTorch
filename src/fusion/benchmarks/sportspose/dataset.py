"""SportsPose release access: clips, calibrations, references and view selection.

Release layout (``paths.dataset_root``)::

    data/<day>/S<nn>/calib.pkl                       7 cameras (R, T, f, c, rot90)
    data/<day>/S<nn>/<activity>/<activity><nnnn>.npy   [F, 17, 3] COCO17 reference, metres, 90 fps
    data/<day>/S<nn>/<activity>/<activity><nnnn>_timing.pkl
        video_index [7, F]  video frame of each reference frame per camera
        times_ms    [7, F]  capture timestamps
        video_path  (day, S<nn>, Video_<date>_<time>)
    videos/<day>/S<nn>/Video_<date>_<time>/CAM<k>.avi   MJPEG 1936x1216, 90 fps, stored
        unrotated: rotate ``rot90_clockwise`` times clockwise to get the upright image

View selection mirrors the private face/side setup: per subject and
sequence (day + activity), the camera whose horizontal azimuth from the subject's mean facing
direction is smallest becomes view A, and the camera closest to
``target_separation_deg`` (default 90) from it becomes view B. The facing
direction is taken from the reference hips and shoulders and is used only to
choose cameras, never as a training signal.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .schema import ACTIVITIES, DAYS, NATIVE_FPS, NUM_CAMERAS, REFERENCE_JOINTS, SelectedViews, SportsPoseCamera, SportsPoseClip

_CLIP_FILE = re.compile(r"^(?P<activity>[a-z_]+)(?P<index>\d{4})\.npy$")
_SUBJECT_DIR = re.compile(r"^S\d{2}$")

# COCO17 indices used for the facing direction.
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 5, 6, 11, 12


def load_calibration(subject_dir: Path) -> dict[str, SportsPoseCamera]:
    """The seven cameras of ``<day>/S<nn>/calib.pkl`` keyed by ``cam<k>``."""
    path = Path(subject_dir) / "calib.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    entries = payload["calibration"]
    rotations = payload.get("numtimesrot90clockwise", [0] * len(entries))
    if len(entries) != NUM_CAMERAS:
        raise ValueError(f"{path}: expected {NUM_CAMERAS} cameras, found {len(entries)}")
    cameras = {}
    for index, entry in enumerate(entries):
        camera = SportsPoseCamera(index=index, rotation=entry["R"], translation=entry["T"], focal=entry["f"], center=entry["c"], rot90_clockwise=int(rotations[index]))
        cameras[camera.view_id] = camera
    return cameras


def load_timing(clip: SportsPoseClip) -> dict[str, np.ndarray]:
    """``video_index`` and ``times_ms`` arrays of shape ``[7, F]``."""
    with Path(clip.timing_path).open("rb") as handle:
        payload = pickle.load(handle)
    video_index = np.asarray(payload["video_index"], dtype=np.int64)
    times_ms = np.asarray(payload["times_ms"], dtype=np.int64)
    if video_index.shape != (NUM_CAMERAS, clip.frames) or times_ms.shape != (NUM_CAMERAS, clip.frames):
        raise ValueError(f"{clip.timing_path}: timing arrays must have shape [{NUM_CAMERAS}, {clip.frames}]")
    return {"video_index": video_index, "times_ms": times_ms}


def load_reference(clip: SportsPoseClip) -> np.ndarray:
    """``[F, 17, 3]`` COCO17 reference in metres (world frame of the calibration)."""
    points = np.asarray(np.load(clip.joints_path), dtype=np.float32)
    if points.shape != (clip.frames, REFERENCE_JOINTS, 3):
        raise ValueError(f"{clip.joints_path}: expected [{clip.frames}, {REFERENCE_JOINTS}, 3], got {points.shape}")
    return points


def discover_clips(
    dataset_root: Path,
    *,
    days: Iterable[str] | None = None,
    subjects: Iterable[str] | None = None,
    activities: Iterable[str] | None = None,
) -> list[SportsPoseClip]:
    """Enumerate every clip with a reference, a timing file and a video folder."""
    root = Path(dataset_root)
    wanted_days = set(days) if days else set(DAYS)
    wanted_subjects = set(subjects) if subjects else None
    wanted_activities = set(activities) if activities else set(ACTIVITIES)
    clips: list[SportsPoseClip] = []
    for day in sorted(wanted_days):
        day_dir = root / "data" / day
        if not day_dir.is_dir():
            continue
        for subject_dir in sorted(p for p in day_dir.iterdir() if p.is_dir() and _SUBJECT_DIR.match(p.name)):
            if wanted_subjects is not None and subject_dir.name not in wanted_subjects:
                continue
            for activity in sorted(wanted_activities):
                activity_dir = subject_dir / activity
                if not activity_dir.is_dir():
                    continue
                for joints_path in sorted(activity_dir.glob("*.npy")):
                    match = _CLIP_FILE.match(joints_path.name)
                    if match is None or match.group("activity") != activity:
                        continue
                    timing_path = joints_path.with_name(joints_path.stem + "_timing.pkl")
                    if not timing_path.is_file():
                        continue
                    with timing_path.open("rb") as handle:
                        video_parts = pickle.load(handle)["video_path"]
                    video_dir = root / "videos" / Path(*[str(part) for part in video_parts])
                    frames = int(np.load(joints_path, mmap_mode="r").shape[0])
                    clips.append(SportsPoseClip(day=day, subject=subject_dir.name, activity=activity, clip_id=joints_path.stem, joints_path=joints_path, timing_path=timing_path, video_dir=video_dir, frames=frames))
    return clips


def group_clips(clips: Sequence[SportsPoseClip]) -> dict[tuple[str, str], list[SportsPoseClip]]:
    """Clips keyed by ``(subject_key, sequence_key)`` in clip-id order."""
    groups: dict[tuple[str, str], list[SportsPoseClip]] = {}
    for clip in sorted(clips, key=lambda c: (c.subject, c.day, c.activity, c.clip_id)):
        groups.setdefault((clip.subject_key, clip.sequence_key), []).append(clip)
    return groups


def body_axes(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean ``(root, up, forward, right)`` of a COCO17 reference sequence."""
    points = np.asarray(reference, dtype=np.float64)
    hips = (points[:, _L_HIP] + points[:, _R_HIP]) / 2.0
    shoulders = (points[:, _L_SHOULDER] + points[:, _R_SHOULDER]) / 2.0
    up = np.nanmean(shoulders - hips, axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    left_to_right = np.nanmean(points[:, _R_SHOULDER] - points[:, _L_SHOULDER] + points[:, _R_HIP] - points[:, _L_HIP], axis=0)
    forward = np.cross(up, left_to_right)
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, up)
    return np.nanmean(hips, axis=0), up, forward, right


def camera_azimuths(cameras: Mapping[str, SportsPoseCamera], references: Sequence[np.ndarray]) -> dict[str, float]:
    """Horizontal azimuth (degrees, 0 = in front of the subject) of each camera.

    The subject frame is the mean body frame over ``references`` (one array
    per clip); azimuths are signed, positive toward the subject's right.
    """
    axes = [body_axes(reference) for reference in references]
    root = np.mean([a[0] for a in axes], axis=0)
    up = np.mean([a[1] for a in axes], axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    forward = np.mean([a[2] for a in axes], axis=0)
    forward -= np.dot(forward, up) * up
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, up)
    azimuths = {}
    for view_id, camera in cameras.items():
        offset = camera.position - root
        offset -= np.dot(offset, up) * up
        azimuths[view_id] = float(np.degrees(np.arctan2(np.dot(offset, right), np.dot(offset, forward))))
    return azimuths


def _angular_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def select_views(
    subject_key: str,
    sequence_key: str,
    clips: Sequence[SportsPoseClip],
    cameras: Mapping[str, SportsPoseCamera],
    *,
    target_separation_deg: float = 90.0,
    max_frontal_deg: float = 60.0,
) -> SelectedViews:
    """Face-like view A and side-like view B for one subject and sequence.

    Raises:
        ValueError: If no camera lies within ``max_frontal_deg`` of the
            subject's facing direction (the setup would not resemble the
            private face/side pair).
    """
    azimuths = camera_azimuths(cameras, [load_reference(clip) for clip in clips])
    view_a = min(azimuths, key=lambda v: (abs(azimuths[v]), v))
    if abs(azimuths[view_a]) > max_frontal_deg:
        raise ValueError(f"{subject_key}/{sequence_key}: most frontal camera {view_a} is {azimuths[view_a]:.1f} deg off the facing direction")
    candidates = {v: _angular_distance(azimuths[v], azimuths[view_a]) for v in azimuths if v != view_a}
    view_b = min(candidates, key=lambda v: (abs(candidates[v] - target_separation_deg), v))
    return SelectedViews(
        subject_key=subject_key,
        sequence_key=sequence_key,
        view_a=view_a,
        view_b=view_b,
        azimuth_a_deg=azimuths[view_a],
        azimuth_b_deg=azimuths[view_b],
        separation_deg=candidates[view_b],
        clips=tuple(clip.clip_id for clip in clips),
    )


def native_fps() -> float:
    return NATIVE_FPS
