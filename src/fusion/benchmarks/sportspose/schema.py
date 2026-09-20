"""Typed records of the SportsPose release layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ACTIVITIES: tuple[str, ...] = ("jump", "soccer", "tennis", "throw_baseball", "volley")
DAYS: tuple[str, ...] = ("indoors", "outdoors")
NATIVE_FPS = 90.0
NUM_CAMERAS = 7
REFERENCE_JOINTS = 17  # COCO17 order, metres


def sequence_key(day: str, activity: str) -> str:
    """Sequence identifier of one subject's trials of an action on one day (``indoors_tennis``).

    S-ids name people: S00 and S10 were recorded both indoors and outdoors,
    so the subject id stays the S-id and the day goes into the sequence id.
    """
    return f"{day}_{activity}"


@dataclass(frozen=True)
class SportsPoseCamera:
    """One ``calib.pkl`` entry: OpenCV world-to-camera extrinsics and intrinsics."""

    index: int
    rotation: np.ndarray
    translation: np.ndarray
    focal: np.ndarray
    center: np.ndarray
    rot90_clockwise: int

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"camera {self.index}: rotation must be a proper 3x3 rotation matrix")
        if not np.isfinite(translation).all():
            raise ValueError(f"camera {self.index}: translation must be finite")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "focal", np.asarray(self.focal, dtype=np.float64).reshape(2))
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64).reshape(2))

    @property
    def view_id(self) -> str:
        return f"cam{self.index}"

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.rotation.T @ self.translation

    @property
    def optical_axis(self) -> np.ndarray:
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class SportsPoseClip:
    """One trial: a 3D reference file, its timing file and the seven videos."""

    day: str
    subject: str
    activity: str
    clip_id: str
    joints_path: Path
    timing_path: Path
    video_dir: Path
    frames: int

    def __post_init__(self) -> None:
        if self.day not in DAYS or self.activity not in ACTIVITIES or self.frames <= 0:
            raise ValueError(f"invalid SportsPose clip {self.day}/{self.subject}/{self.activity}/{self.clip_id}")
        for name in ("joints_path", "timing_path", "video_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))

    @property
    def subject_key(self) -> str:
        """Split-level subject identifier (the S-id)."""
        return self.subject

    @property
    def sequence_key(self) -> str:
        """Sequence identifier of the training adapter (``<day>_<activity>``)."""
        return sequence_key(self.day, self.activity)

    def video_path(self, camera_index: int) -> Path:
        return self.video_dir / f"CAM{int(camera_index)}.avi"


@dataclass(frozen=True)
class SelectedViews:
    """The two cameras used for one subject and sequence (day + activity).

    ``view_a`` faces the subject most directly (the "face" role of the private
    data); ``view_b`` is the camera closest to ``target_separation_deg`` of
    horizontal azimuth from it (the "side" role).
    """

    subject_key: str
    sequence_key: str
    view_a: str
    view_b: str
    azimuth_a_deg: float
    azimuth_b_deg: float
    separation_deg: float
    clips: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_key": self.subject_key,
            "sequence_key": self.sequence_key,
            "view_a": self.view_a,
            "view_b": self.view_b,
            "azimuth_a_deg": self.azimuth_a_deg,
            "azimuth_b_deg": self.azimuth_b_deg,
            "separation_deg": self.separation_deg,
            "clips": list(self.clips),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelectedViews":
        return cls(
            subject_key=str(payload["subject_key"]),
            sequence_key=str(payload["sequence_key"]),
            view_a=str(payload["view_a"]),
            view_b=str(payload["view_b"]),
            azimuth_a_deg=float(payload["azimuth_a_deg"]),
            azimuth_b_deg=float(payload["azimuth_b_deg"]),
            separation_deg=float(payload["separation_deg"]),
            clips=tuple(str(c) for c in payload.get("clips", ())),
        )
