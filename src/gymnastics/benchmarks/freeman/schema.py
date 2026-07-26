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
    subject_id: int
    fps: int
    split: str
    scenario: str | None
    action: str | None
    reference_scale_to_m: float
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
        if (
            self.subject_id < 1
            or self.subject_id > 40
            or self.fps not in {30, 60}
            or not self.split
            or not np.isfinite(self.reference_scale_to_m)
            or self.reference_scale_to_m <= 0
        ):
            raise ValueError("reference subject, FPS, split, and unit scale are required")
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


@dataclass(frozen=True)
class ViewPrediction:
    """One selected FreeMan view's compact SAM3D prediction."""

    session_id: str
    subject_id: int
    fps: float
    view_id: str
    frame_ids: np.ndarray
    points3d: np.ndarray
    points2d: np.ndarray
    valid3d: np.ndarray
    valid2d: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64)
        points3d = np.asarray(self.points3d, dtype=np.float32)
        points2d = np.asarray(self.points2d, dtype=np.float32)
        valid3d = np.asarray(self.valid3d, dtype=bool)
        valid2d = np.asarray(self.valid2d, dtype=bool)
        frames = len(frame_ids)
        if points3d.shape != (frames, 70, 3):
            raise ValueError("SAM3D points3d must have shape [T,70,3]")
        if points2d.shape != (frames, 70, 2):
            raise ValueError("SAM3D points2d must have shape [T,70,2]")
        if valid3d.shape != (frames, 70) or valid2d.shape != (frames, 70):
            raise ValueError("SAM3D validity masks must have shape [T,70]")
        if (
            frames == 0
            or np.any(frame_ids < 0)
            or (frames > 1 and not np.all(np.diff(frame_ids) > 0))
        ):
            raise ValueError("SAM3D frame IDs must be non-empty and increasing")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("SAM3D prediction FPS must be positive")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("SAM3D prediction metadata must be a mapping")
        object.__setattr__(self, "frame_ids", _readonly_array(frame_ids))
        object.__setattr__(self, "points3d", _readonly_array(points3d))
        object.__setattr__(self, "points2d", _readonly_array(points2d))
        object.__setattr__(self, "valid3d", _readonly_array(valid3d))
        object.__setattr__(self, "valid2d", _readonly_array(valid2d))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class InferenceIdentity:
    """Fields that make a cached SAM3D view reusable."""

    session_id: str
    subject_id: int
    fps: int
    view_id: str
    source_video_sha256: str
    source_frame_count: int
    frame_stride: int
    sam3d_config_sha256: str
    checkpoint_id: str

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or not self.view_id
            or self.subject_id < 1
            or self.subject_id > 40
            or self.fps not in {30, 60}
            or self.source_frame_count < 1
            or self.frame_stride < 1
            or not self.checkpoint_id
        ):
            raise ValueError("invalid SAM3D inference identity")
        for name, digest in (
            ("source_video_sha256", self.source_video_sha256),
            ("sam3d_config_sha256", self.sam3d_config_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA256")


@dataclass(frozen=True)
class InferenceArtifact:
    """Published SAM3D prediction artifact summary."""

    path: Path
    session_id: str
    view_id: str
    frames: int
    valid_frames: int

    def __post_init__(self) -> None:
        if self.frames < 1 or self.valid_frames < 0 or self.valid_frames > self.frames:
            raise ValueError("invalid inference artifact frame counts")
        object.__setattr__(self, "path", Path(self.path).resolve())


@dataclass(frozen=True)
class PosePairInput:
    """Exact synchronized pair passed to fusion without reference 3D."""

    session_id: str
    subject_id: int
    fps: float
    view_a: ViewPrediction
    view_b: ViewPrediction

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or self.subject_id < 1
            or self.subject_id > 40
            or not np.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("pose pair requires session, subject, and positive FPS")
        for view in (self.view_a, self.view_b):
            if (
                view.session_id != self.session_id
                or view.subject_id != self.subject_id
                or not np.isclose(view.fps, self.fps)
            ):
                raise ValueError("pose pair view identity does not match pair identity")
        if self.view_a.view_id == self.view_b.view_id:
            raise ValueError("pose pair requires two distinct views")


@dataclass(frozen=True)
class MethodPrediction:
    """One fused or single-view prediction ready for evaluation."""

    method: str
    session_id: str
    subject_id: int
    fps: float
    points: np.ndarray
    valid: np.ndarray
    frame_ids: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64)
        if not self.method or not self.session_id:
            raise ValueError("method and session IDs are required")
        if points.ndim != 3 or points.shape[1:] != (70, 3):
            raise ValueError("method points must have shape [T,70,3]")
        if valid.shape != points.shape[:2] or frame_ids.shape != (points.shape[0],):
            raise ValueError("method validity and frame IDs must match points")
        if (
            len(frame_ids) == 0
            or (len(frame_ids) > 1 and not np.all(np.diff(frame_ids) > 0))
            or not np.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("method prediction requires increasing frames and positive FPS")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("method metadata must be a mapping")
        object.__setattr__(self, "points", _readonly_array(points))
        object.__setattr__(self, "valid", _readonly_array(valid))
        object.__setattr__(self, "frame_ids", _readonly_array(frame_ids))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
