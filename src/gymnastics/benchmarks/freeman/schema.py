"""Immutable data contracts for the FreeMan benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ArchiveEntry:
    """One file published by the gated Hugging Face dataset repository."""

    path: str
    size: int
    sha256: str | None

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            not self.path
            or path.is_absolute()
            or ".." in path.parts
            or self.size <= 0
        ):
            raise ValueError(f"invalid archive entry: {self.path!r}")
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"invalid SHA256 for {self.path!r}")
            object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class PreflightReport:
    """Read-only result of authentication, access, and storage checks."""

    repo_id: str
    revision: str
    hf_executable: Path
    authenticated_user: str
    access_granted: bool
    archive_root: Path
    required_bytes: int
    free_bytes: int
    reserve_bytes: int
    entries: tuple[ArchiveEntry, ...]

    def __post_init__(self) -> None:
        if not self.repo_id or not self.revision or not self.authenticated_user:
            raise ValueError("repository, revision, and authenticated user are required")
        if min(self.required_bytes, self.free_bytes, self.reserve_bytes) < 0:
            raise ValueError("preflight byte counts must be non-negative")
        object.__setattr__(self, "hf_executable", Path(self.hf_executable).resolve())
        object.__setattr__(self, "archive_root", Path(self.archive_root).resolve())
        object.__setattr__(self, "entries", tuple(self.entries))


def _readonly_array(
    value: Any,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _path_mapping(value: Mapping[str, Path]) -> Mapping[str, Path]:
    return MappingProxyType(
        {str(name): Path(path).resolve() for name, path in value.items()}
    )


@dataclass(frozen=True)
class FreeManCamera:
    """One official OpenCV-style FreeMan camera calibration."""

    name: str
    size: tuple[int, int]
    matrix: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    distortions: np.ndarray

    def __post_init__(self) -> None:
        if not self.name or len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("camera name and positive image size are required")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        distortions = np.asarray(self.distortions, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera matrix must be finite with shape [3,3]")
        if rotation.shape != (3,) or not np.isfinite(rotation).all():
            raise ValueError("camera rotation must be a finite Rodrigues vector")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("camera translation must be a finite 3-vector")
        if distortions.ndim != 1 or not np.isfinite(distortions).all():
            raise ValueError("camera distortions must be a finite vector")
        object.__setattr__(self, "size", tuple(int(value) for value in self.size))
        object.__setattr__(self, "matrix", _readonly_array(matrix))
        object.__setattr__(self, "rotation", _readonly_array(rotation))
        object.__setattr__(self, "translation", _readonly_array(translation))
        object.__setattr__(self, "distortions", _readonly_array(distortions))


@dataclass(frozen=True)
class FreeManSession:
    """Validated synchronized FreeMan session metadata."""

    session_id: str
    subject_id: int
    fps: int
    split: str
    scenario: str | None
    action: str | None
    video_paths: Mapping[str, Path]
    cameras: Mapping[str, FreeManCamera]
    keypoints2d_path: Path
    keypoints3d_path: Path
    frame_ids: np.ndarray
    excluded_trailing_frames: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or self.subject_id < 1
            or self.subject_id > 40
            or self.fps not in {30, 60}
        ):
            raise ValueError("session ID, subject 1..40, and FPS 30/60 are required")
        videos = _path_mapping(self.video_paths)
        cameras = MappingProxyType(dict(self.cameras))
        if len(videos) < 2 or set(videos) != set(cameras):
            raise ValueError("session videos and cameras must share at least two view IDs")
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64)
        if (
            frame_ids.ndim != 1
            or len(frame_ids) == 0
            or np.any(frame_ids < 0)
            or (len(frame_ids) > 1 and not np.all(np.diff(frame_ids) > 0))
        ):
            raise ValueError("session frame_ids must be non-empty and increasing")
        exclusions = {
            str(name): int(count)
            for name, count in self.excluded_trailing_frames.items()
        }
        if any(count < 0 for count in exclusions.values()):
            raise ValueError("excluded trailing frame counts must be non-negative")
        object.__setattr__(self, "video_paths", videos)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "keypoints2d_path", Path(self.keypoints2d_path).resolve())
        object.__setattr__(self, "keypoints3d_path", Path(self.keypoints3d_path).resolve())
        object.__setattr__(self, "frame_ids", _readonly_array(frame_ids))
        object.__setattr__(
            self,
            "excluded_trailing_frames",
            MappingProxyType(exclusions),
        )


@dataclass(frozen=True)
class SelectedPair:
    """Deterministically selected near-orthogonal camera pair."""

    session_id: str
    view_a: str
    view_b: str
    reference_view: str
    separation_deg: float
    target_error_deg: float
    height_difference: float

    def __post_init__(self) -> None:
        values = (
            self.separation_deg,
            self.target_error_deg,
            self.height_difference,
        )
        if (
            not self.session_id
            or not self.view_a
            or self.view_a >= self.view_b
            or self.reference_view != self.view_a
            or not np.isfinite(values).all()
            or min(values) < 0
        ):
            raise ValueError("invalid selected camera pair")


@dataclass(frozen=True)
class ReferenceSequence:
    """FreeMan optimized markerless multi-view 3D reference sequence."""

    session_id: str
    points_m: np.ndarray
    valid: np.ndarray
    frame_ids: np.ndarray
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_m, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("reference points must have shape [T,J,3]")
        if valid.shape != points.shape[:2] or frame_ids.shape != (points.shape[0],):
            raise ValueError("reference masks and frame IDs must match points")
        if len(self.joint_names) != points.shape[1]:
            raise ValueError("reference joint names must match point count")
        object.__setattr__(self, "points_m", _readonly_array(points))
        object.__setattr__(self, "valid", _readonly_array(valid))
        object.__setattr__(self, "frame_ids", _readonly_array(frame_ids))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))


@dataclass(frozen=True)
class MappedPose:
    """MHR70 prediction restricted to the FreeMan COCO17 order."""

    points: np.ndarray
    valid: np.ndarray
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("mapped points must have shape [T,J,3]")
        if valid.shape != points.shape[:2] or len(self.joint_names) != points.shape[1]:
            raise ValueError("mapped validity and names must match points")
        object.__setattr__(self, "points", _readonly_array(points))
        object.__setattr__(self, "valid", _readonly_array(valid))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
