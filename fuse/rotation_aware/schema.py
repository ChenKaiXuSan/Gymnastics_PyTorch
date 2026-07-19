"""Immutable data contracts shared by rotation-aware fusion stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fuse.metadata.mhr70 import mhr_names


def valid_from_points(points: np.ndarray) -> np.ndarray:
    """Return validity for finite, non-zero xyz joints in a `[T, J, 3]` array."""
    array = np.asarray(points)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("points must have shape [T, J, 3]")
    return np.isfinite(array).all(axis=-1) & np.any(array != 0, axis=-1)


def _readonly_array(value: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PosePairTrial:
    """One split-cycle-aligned face/side SAM3D sequence."""

    face: np.ndarray
    side: np.ndarray
    valid_face: np.ndarray
    valid_side: np.ndarray
    timestamps: np.ndarray
    face_map: np.ndarray
    side_map: np.ndarray
    joint_names: tuple[str, ...]
    person_id: str
    trial_id: str
    fps: float

    def __post_init__(self) -> None:
        face = np.asarray(self.face)
        side = np.asarray(self.side)
        if face.ndim != 3 or face.shape[-1] != 3 or face.shape != side.shape:
            raise ValueError("face and side must have equal shape [T, J, 3]")
        if face.shape[0] == 0:
            raise ValueError("PosePairTrial must contain at least one frame")
        if tuple(self.joint_names) != tuple(mhr_names) or len(self.joint_names) != face.shape[1]:
            raise ValueError("joint_names must exactly match the MHR70 joint order and pose joint dimension")

        valid_face = np.asarray(self.valid_face, dtype=bool)
        valid_side = np.asarray(self.valid_side, dtype=bool)
        expected_mask_shape = face.shape[:2]
        if valid_face.shape != expected_mask_shape or valid_side.shape != expected_mask_shape:
            raise ValueError("valid masks must have shape [T, J]")
        if np.any(valid_face & ~valid_from_points(face)) or np.any(valid_side & ~valid_from_points(side)):
            raise ValueError("valid masks cannot mark non-finite or zero points as valid")

        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        face_map = np.asarray(self.face_map, dtype=np.int32)
        side_map = np.asarray(self.side_map, dtype=np.int32)
        if timestamps.shape != (face.shape[0],) or not np.isfinite(timestamps).all():
            raise ValueError("timestamps must be finite with shape [T]")
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")
        for name, frame_map in (("face_map", face_map), ("side_map", side_map)):
            if frame_map.shape != (face.shape[0],) or np.any(frame_map < 0):
                raise ValueError(f"{name} must contain non-negative frame ids with shape [T]")
            if len(frame_map) > 1 and not np.all(np.diff(frame_map) > 0):
                raise ValueError(f"{name} must be strictly increasing")
        if not self.person_id or not self.trial_id or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("person_id, trial_id, and positive finite fps are required")

        object.__setattr__(self, "face", _readonly_array(face, dtype=np.float32))
        object.__setattr__(self, "side", _readonly_array(side, dtype=np.float32))
        object.__setattr__(self, "valid_face", _readonly_array(valid_face, dtype=bool))
        object.__setattr__(self, "valid_side", _readonly_array(valid_side, dtype=bool))
        object.__setattr__(self, "timestamps", _readonly_array(timestamps, dtype=np.float64))
        object.__setattr__(self, "face_map", _readonly_array(face_map, dtype=np.int32))
        object.__setattr__(self, "side_map", _readonly_array(side_map, dtype=np.int32))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
