"""Immutable contracts shared by Unity benchmark stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _readonly(value: np.ndarray, *, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class UnityCamera:
    camera_id: str
    image_size: tuple[int, int]
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    clip_projection: np.ndarray

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id is required")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must be positive (width, height)")
        for name in ("camera_to_world", "world_to_camera", "clip_projection"):
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must be a finite 4x4 matrix")
            object.__setattr__(self, name, _readonly(matrix, dtype=np.float64))


@dataclass(frozen=True)
class UnityFrame:
    sample_id: int
    sequence_id: str
    frame_index: int
    sample_type: str
    phase: str
    time_seconds: float
    actual_angle_deg: float
    image_paths: Mapping[str, Path]
    gt_world_m: np.ndarray
    gt_available: np.ndarray
    gt_pixels: Mapping[str, np.ndarray]
    visible: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.sample_id < 0 or self.frame_index < 0:
            raise ValueError("sample_id and frame_index must be non-negative")
        if not self.sequence_id or not self.sample_type:
            raise ValueError("sequence_id and sample_type are required")
        if not np.isfinite((self.time_seconds, self.actual_angle_deg)).all():
            raise ValueError("time and angle must be finite")
        world = np.asarray(self.gt_world_m, dtype=np.float32)
        available = np.asarray(self.gt_available, dtype=bool)
        if world.ndim != 2 or world.shape[1] != 3:
            raise ValueError("gt_world_m must have shape [J,3]")
        if available.shape != world.shape[:1]:
            raise ValueError("gt_available must have shape [J]")
        paths = {str(key): Path(value) for key, value in self.image_paths.items()}
        pixels = {
            str(key): _readonly(value, dtype=np.float32)
            for key, value in self.gt_pixels.items()
        }
        visible = {
            str(key): _readonly(value, dtype=bool)
            for key, value in self.visible.items()
        }
        for camera_id in paths:
            if camera_id not in pixels or pixels[camera_id].shape != (len(world), 2):
                raise ValueError(f"invalid 2D keypoints for {camera_id}")
            if camera_id not in visible or visible[camera_id].shape != (len(world),):
                raise ValueError(f"invalid visibility for {camera_id}")
        object.__setattr__(self, "image_paths", MappingProxyType(paths))
        object.__setattr__(self, "gt_world_m", _readonly(world, dtype=np.float32))
        object.__setattr__(self, "gt_available", _readonly(available, dtype=bool))
        object.__setattr__(self, "gt_pixels", MappingProxyType(pixels))
        object.__setattr__(self, "visible", MappingProxyType(visible))


@dataclass(frozen=True)
class UnityBenchmark:
    root: Path
    joint_names: tuple[str, ...]
    cameras: Mapping[str, UnityCamera]
    frames: tuple[UnityFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("benchmark must contain at least one frame")
        if len(self.joint_names) != self.frames[0].gt_world_m.shape[0]:
            raise ValueError("joint_names do not match frame joint count")
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(
            self, "cameras", MappingProxyType(dict(self.cameras))
        )
        object.__setattr__(self, "frames", tuple(self.frames))


@dataclass(frozen=True)
class MappedPose:
    points: np.ndarray
    valid: np.ndarray
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("points must end with [J,3]")
        if valid.shape != points.shape[:-1]:
            raise ValueError("valid must match points without xyz")
        if points.shape[-2] != len(self.joint_names):
            raise ValueError("joint names must match pose joint dimension")
        object.__setattr__(self, "points", _readonly(points, dtype=np.float32))
        object.__setattr__(self, "valid", _readonly(valid, dtype=bool))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))

    def index(self, name: str) -> int:
        return self.joint_names.index(name)


@dataclass(frozen=True)
class MethodSequence:
    method: str
    sequence_id: str
    sample_ids: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    joint_names: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        sample_ids = np.asarray(self.sample_ids, dtype=np.int64)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("method points must have shape [T,J,3]")
        if valid.shape != points.shape[:2]:
            raise ValueError("method valid mask must have shape [T,J]")
        if sample_ids.shape != (points.shape[0],):
            raise ValueError("sample_ids must have shape [T]")
        if len(self.joint_names) != points.shape[1]:
            raise ValueError("joint_names must match J")
        object.__setattr__(self, "points", _readonly(points, dtype=np.float32))
        object.__setattr__(self, "valid", _readonly(valid, dtype=bool))
        object.__setattr__(self, "sample_ids", _readonly(sample_ids, dtype=np.int64))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
