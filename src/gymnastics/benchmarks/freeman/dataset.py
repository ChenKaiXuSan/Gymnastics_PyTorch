"""Validated loader for the official FreeMan release layout."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from .mapping import FREEMAN_COCO17_NAMES
from .schema import FreeManCamera, FreeManSession, ReferenceSequence


_VIEWS = tuple(f"c{view:02d}" for view in range(1, 9))
_SUBJECT_PATTERN = re.compile(r"_subj(\d+)$")


def _read_lines(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _validation_split_path(root: Path) -> Path:
    candidates = [
        path
        for path in (root / "valid.txt", root / "validation.txt")
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"validation split requires exactly one of valid.txt or validation.txt: {root}"
        )
    return candidates[0]


def _split_membership(root: Path) -> dict[str, str]:
    files = {
        "train": root / "train.txt",
        "validation": _validation_split_path(root),
        "test": root / "test.txt",
    }
    membership: dict[str, str] = {}
    for split, path in files.items():
        for session_id in _read_lines(path):
            if session_id in membership:
                raise ValueError(
                    f"session {session_id} appears in multiple official splits"
                )
            membership[session_id] = split
    return membership


def _subject_from_session(session_id: str) -> int:
    match = _SUBJECT_PATTERN.search(session_id)
    if match is None:
        raise ValueError(f"session does not end in _subjNN: {session_id}")
    subject = int(match.group(1))
    if subject < 1 or subject > 40:
        raise ValueError(f"session subject is outside 1..40: {session_id}")
    return subject


def _subject_from_root(subject_root: Path) -> int:
    match = re.fullmatch(r"subject_(\d+)", subject_root.name)
    if match is None:
        raise ValueError(f"subject workspace must be named subject_NN: {subject_root}")
    subject = int(match.group(1))
    if subject < 1 or subject > 40:
        raise ValueError(f"subject workspace is outside 1..40: {subject_root}")
    return subject


def _load_object_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=True)
    if value.shape == ():
        payload = value.item()
    elif len(value) == 1:
        payload = value[0]
    else:
        raise ValueError(f"annotation must contain one mapping: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"annotation payload must be a mapping: {path}")
    return payload


def _load_annotation_shapes(
    keypoints2d_path: Path,
    keypoints3d_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    payload2d = _load_object_mapping(keypoints2d_path)
    payload3d = _load_object_mapping(keypoints3d_path)
    if "keypoints2d" not in payload2d:
        raise ValueError(f"keypoints2d field missing: {keypoints2d_path}")
    if "keypoints3d_optim" not in payload3d:
        raise ValueError(
            f"keypoints3d_optim field missing: {keypoints3d_path}"
        )
    keypoints2d = np.asarray(payload2d["keypoints2d"], dtype=np.float32)
    keypoints3d = np.asarray(payload3d["keypoints3d_optim"], dtype=np.float32)
    if (
        keypoints2d.ndim != 4
        or keypoints2d.shape[0] != 8
        or keypoints2d.shape[2:] != (17, 3)
    ):
        raise ValueError("FreeMan keypoints2d must have shape [8,F,17,3]")
    if keypoints3d.ndim != 3 or keypoints3d.shape[1:] != (17, 3):
        raise ValueError("FreeMan keypoints3d_optim must have shape [F,17,3]")
    if not np.isfinite(keypoints2d).all() or not np.isfinite(keypoints3d).all():
        raise ValueError("FreeMan annotations must contain finite values")
    return keypoints2d, keypoints3d


def _load_cameras(path: Path) -> dict[str, FreeManCamera]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"FreeMan camera JSON must contain a list: {path}")
    cameras: dict[str, FreeManCamera] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"FreeMan camera entry must be a mapping: {path}")
        camera = FreeManCamera(
            name=str(item["name"]),
            size=tuple(int(value) for value in item["size"]),
            matrix=np.asarray(item["matrix"]),
            rotation=np.asarray(item["rotation"]),
            translation=np.asarray(item["translation"]),
            distortions=np.asarray(item["distortions"]),
        )
        if camera.name in cameras:
            raise ValueError(f"duplicate camera {camera.name}: {path}")
        cameras[camera.name] = camera
    if tuple(sorted(cameras)) != _VIEWS:
        raise ValueError(f"FreeMan session requires cameras c01 through c08: {path}")
    return cameras


def _video_metadata(path: Path, expected_fps: int) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open FreeMan video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if not np.isfinite(fps) or abs(fps - expected_fps) > 0.5:
        raise ValueError(
            f"FreeMan video FPS mismatch for {path}: {fps} != {expected_fps}"
        )
    if frames <= 0:
        raise ValueError(f"FreeMan video has no frames: {path}")
    return frames


def _subset_lists_root(shared_root: Path, subset_root: Path) -> Path:
    if (subset_root / "session_list.txt").is_file():
        return subset_root
    if (shared_root / "session_list.txt").is_file():
        return shared_root
    raise FileNotFoundError(subset_root / "session_list.txt")


def load_subject_sessions(
    subject_root: Path,
    shared_root: Path,
    fps_values: Sequence[int],
) -> tuple[FreeManSession, ...]:
    """Load every official session belonging to one extracted subject."""
    subject_path = Path(subject_root).resolve()
    shared_path = Path(shared_root).resolve()
    subject_id = _subject_from_root(subject_path)
    sessions: list[FreeManSession] = []
    seen: set[tuple[int, str]] = set()
    for fps_value in fps_values:
        fps = int(fps_value)
        if fps not in {30, 60}:
            raise ValueError("FreeMan FPS subset must be 30 or 60")
        subset = shared_path / f"{fps}FPS"
        videos_root = subject_path / f"{fps}FPS" / "videos"
        if not subset.exists() or not videos_root.exists():
            continue
        lists_root = _subset_lists_root(shared_path, subset)
        session_ids = _read_lines(lists_root / "session_list.txt")
        if len(set(session_ids)) != len(session_ids):
            raise ValueError(f"duplicate session ID in {lists_root / 'session_list.txt'}")
        membership = _split_membership(lists_root)
        for session_id in session_ids:
            session_subject = _subject_from_session(session_id)
            if session_subject != subject_id:
                continue
            identity = (fps, session_id)
            if identity in seen:
                raise ValueError(f"duplicate session identity: {identity}")
            seen.add(identity)
            session_video_root = videos_root / session_id / "vframes"
            video_paths = {
                view: session_video_root / f"{view}.mp4"
                for view in _VIEWS
            }
            missing_videos = [
                view for view, path in video_paths.items() if not path.is_file()
            ]
            if missing_videos:
                raise FileNotFoundError(
                    f"session {session_id} missing videos: {missing_videos}"
                )
            cameras = _load_cameras(subset / "cameras" / f"{session_id}.json")
            keypoints2d_path = subset / "keypoints2d" / f"{session_id}.npy"
            keypoints3d_path = subset / "keypoints3d" / f"{session_id}.npy"
            keypoints2d, keypoints3d = _load_annotation_shapes(
                keypoints2d_path,
                keypoints3d_path,
            )
            counts = {
                view: _video_metadata(path, fps)
                for view, path in video_paths.items()
            }
            counts["keypoints2d"] = int(keypoints2d.shape[1])
            counts["keypoints3d"] = int(keypoints3d.shape[0])
            common_frames = min(counts.values())
            if common_frames <= 0:
                raise ValueError(f"session {session_id} has no common frames")
            exclusions = {
                name: count - common_frames for name, count in counts.items()
            }
            sessions.append(
                FreeManSession(
                    session_id=session_id,
                    subject_id=subject_id,
                    fps=fps,
                    split=membership.get(session_id, "unassigned"),
                    scenario=None,
                    action=None,
                    video_paths=video_paths,
                    cameras=cameras,
                    keypoints2d_path=keypoints2d_path,
                    keypoints3d_path=keypoints3d_path,
                    frame_ids=np.arange(common_frames, dtype=np.int64),
                    excluded_trailing_frames=exclusions,
                )
            )
    return tuple(sorted(sessions, key=lambda item: (item.fps, item.session_id)))


def load_session_reference(
    session: FreeManSession,
    *,
    reference_scale_to_m: float,
) -> ReferenceSequence:
    """Load only the optimized FreeMan markerless 3D reference field."""
    scale = float(reference_scale_to_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference_scale_to_m must be positive and finite")
    payload = _load_object_mapping(session.keypoints3d_path)
    if "keypoints3d_optim" not in payload:
        raise ValueError(
            f"keypoints3d_optim field missing: {session.keypoints3d_path}"
        )
    all_points = np.asarray(payload["keypoints3d_optim"], dtype=np.float32)
    if all_points.ndim != 3 or all_points.shape[1:] != (17, 3):
        raise ValueError("FreeMan keypoints3d_optim must have shape [F,17,3]")
    points = all_points[session.frame_ids] * scale
    valid = np.isfinite(points).all(axis=-1)
    safe_points = np.where(valid[..., None], points, 0)
    return ReferenceSequence(
        session_id=session.session_id,
        subject_id=session.subject_id,
        fps=session.fps,
        split=session.split,
        scenario=session.scenario,
        action=session.action,
        reference_scale_to_m=scale,
        points_m=safe_points,
        valid=valid,
        frame_ids=session.frame_ids,
        joint_names=FREEMAN_COCO17_NAMES,
    )
