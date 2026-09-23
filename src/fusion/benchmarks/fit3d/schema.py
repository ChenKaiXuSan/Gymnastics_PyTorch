"""Typed records of the Fit3D release layout.

Fit3D (Fieraru et al., CVPR 2021, "AIFit"): 8 training subjects with 3D
ground truth, 47 fitness exercises each, four synchronised calibrated
cameras at 50 fps, and -- unique among the public sets used here --
**repetition annotations**, so the movement cycles come from the dataset
instead of a detector.

Release layout (``<root>/train/<subject>/``)::

    camera_parameters/<camera>/<action>.json   extrinsics R, T (T = camera centre) + intrinsics
    joints3d_25/<action>.json                  {"joints3d_25": [T, 25, 3]} in metres, world frame
    rep_ann.json                               {action: [frame, frame, ...]} repetition marks
    videos/<camera>/<action>.mp4               900 x 900, 50 fps
    smplx/, gpp/                               not used here

The three test subjects carry no ``joints3d_25`` and are therefore not used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NATIVE_FPS = 50.0
REFERENCE_JOINTS = 25
TRAIN_SUBJECTS: tuple[str, ...] = ("s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11")
CAMERAS: tuple[str, ...] = ("50591643", "58860488", "60457274", "65906101")

# joints3d_25 = the Human3.6M 17-joint core (0-16) plus two foot and two hand
# joints per side (17-24).  The correspondence was established by projecting
# the reference into every camera and matching it against the SAM3D 2D
# keypoints (scratch: fit3d_mapping.py; >= 94 of 104 votes per joint, the
# runner-up never above 13).  Only the unambiguous joints are mapped: Fit3D
# has no heel, no eyes and no ears, and its foot joints (17-20) cannot be
# told apart reliably, so those stay invalid in the reference mask.
JOINTS3D_25_TO_MHR70: dict[str, int] = {
    "nose": 9,
    "neck": 8,
    "left-shoulder": 11,
    "right-shoulder": 14,
    "left-elbow": 12,
    "right-elbow": 15,
    "left-wrist": 13,
    "right-wrist": 16,
    "left-hip": 1,
    "right-hip": 4,
    "left-knee": 2,
    "right-knee": 5,
    "left-ankle": 3,
    "right-ankle": 6,
    "pelvis-approx": 0,  # informational; the MHR70 pelvis is the hip midpoint
}
"""MHR70 joint name -> index in ``joints3d_25`` (15 joints; ``pelvis-approx`` is not mapped)."""


def sequence_key(action: str) -> str:
    """Sequence identifier of one subject's recording of one exercise."""
    return str(action)


@dataclass(frozen=True)
class Fit3DCamera:
    """One ``camera_parameters/<camera>/<action>.json`` entry.

    Fit3D stores the extrinsics as the camera centre ``T`` and the rotation
    ``R`` with ``x_camera = (x_world - T) @ R.T`` (verified by reprojection
    against the SAM3D 2D keypoints), and the undistorted intrinsics as focal
    lengths ``f`` and principal point ``c``.
    """

    camera_id: str
    rotation: np.ndarray
    center: np.ndarray
    focal: np.ndarray
    principal_point: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        center = np.asarray(self.center, dtype=np.float64).reshape(3)
        if not np.isfinite(rotation).all() or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"camera {self.camera_id}: rotation must be a proper rotation matrix")
        if not np.isfinite(center).all():
            raise ValueError(f"camera {self.camera_id}: camera centre must be finite")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "focal", np.asarray(self.focal, dtype=np.float64).reshape(2))
        object.__setattr__(self, "principal_point", np.asarray(self.principal_point, dtype=np.float64).reshape(2))

    @property
    def view_id(self) -> str:
        return self.camera_id

    @property
    def optical_axis(self) -> np.ndarray:
        """Viewing direction in world coordinates (the camera's ``+z``)."""
        return self.rotation[2, :]

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points ``[..., 3]`` to pixels; returns ``(uv, depth)``."""
        camera_points = (np.asarray(points, dtype=np.float64) - self.center) @ self.rotation.T
        depth = camera_points[..., 2]
        uv = camera_points[..., :2] / np.clip(depth[..., None], 1e-6, None) * self.focal + self.principal_point
        return uv, depth


@dataclass(frozen=True)
class Fit3DSequence:
    """One subject performing one exercise, seen by every camera."""

    subject: str
    action: str
    root: Path
    frames: int
    cameras: tuple[str, ...] = CAMERAS

    def __post_init__(self) -> None:
        if not self.subject or not self.action or self.frames <= 0:
            raise ValueError(f"invalid Fit3D sequence {self.subject}/{self.action}")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "cameras", tuple(str(c) for c in self.cameras))

    @property
    def subject_key(self) -> str:
        return self.subject

    @property
    def sequence_key(self) -> str:
        return sequence_key(self.action)

    @property
    def reference_path(self) -> Path:
        return self.root / "joints3d_25" / f"{self.action}.json"

    def camera_path(self, camera: str) -> Path:
        return self.root / "camera_parameters" / str(camera) / f"{self.action}.json"

    def video_path(self, camera: str) -> Path:
        return self.root / "videos" / str(camera) / f"{self.action}.mp4"


@dataclass(frozen=True)
class SelectedViews:
    """The two cameras used for one subject and exercise.

    ``view_a`` faces the subject most directly (the "face" role of the
    private data) and ``view_b`` is the camera closest to
    ``target_separation_deg`` of horizontal azimuth from it (the "side"
    role).  Calibration is used for this choice only; the fusion methods
    never see it.
    """

    subject_key: str
    sequence_key: str
    view_a: str
    view_b: str
    azimuth_a_deg: float
    azimuth_b_deg: float
    separation_deg: float
    frames: int = 0
    reps: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_key": self.subject_key,
            "sequence_key": self.sequence_key,
            "view_a": self.view_a,
            "view_b": self.view_b,
            "azimuth_a_deg": self.azimuth_a_deg,
            "azimuth_b_deg": self.azimuth_b_deg,
            "separation_deg": self.separation_deg,
            "frames": self.frames,
            "reps": self.reps,
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
            frames=int(payload.get("frames", 0)),
            reps=int(payload.get("reps", 0)),
        )


@dataclass(frozen=True)
class ViewPrediction:
    """SAM3D-Body output of one camera at the frames of one sequence."""

    sequence_key: str
    view_id: str
    frame_ids: np.ndarray
    points_3d: np.ndarray
    points_2d: np.ndarray
    valid_3d: np.ndarray
    valid_2d: np.ndarray
    failed_frames: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        frames = len(self.frame_ids)
        if self.points_3d.shape != (frames, 70, 3) or self.valid_3d.shape != (frames, 70):
            raise ValueError(f"{self.view_id}: expected [T, 70, 3] predictions for {frames} frames")
