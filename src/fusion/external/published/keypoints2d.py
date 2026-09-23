"""Per-view 2D keypoint sequences (MHR70 image coordinates) for every dataset.

The published lifters consume 2D keypoints, which the training package never
touches; this module reads them from the SAM3D outputs each dataset already
has:

    gymnastics   per-frame ``<sam3d_root>/person/<id>/<view>/<frame>_sam3d_body.npz``
                 (``pred_keypoints_2d`` on the stored frame; the frame size is
                 taken from the stored image).  Reading every file also decodes
                 the full frame, so the result is cached once per person and
                 view below ``local/runs/external_published/keypoints2d``.
    freeman      ``prediction.npz`` of the benchmark cache (``points2d``); the
                 release videos are portrait 1080x1920.
    fit3d        the external per-video cache (rank-0 person, 900x900 frames).

Every loader returns :class:`View2D` with the keypoints indexed by video frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from common.paths import PROJECT_ROOT, SAM3D_RESULTS_ROOT

DEFAULT_CACHE_ROOT = PROJECT_ROOT / "local" / "runs" / "external_published" / "keypoints2d"


@dataclass(frozen=True)
class View2D:
    """2D keypoints of one video (view) indexed by frame."""

    frame_ids: np.ndarray  # [N] int64, sorted video frame indices
    points: np.ndarray  # [N, 70, 2] float32 image coordinates (upright frame)
    valid: np.ndarray  # [N, 70] bool
    width: int
    height: int
    name: str = ""  # stable identity of the video (cache key of lifted results)

    def __post_init__(self) -> None:
        if self.points.shape != (len(self.frame_ids), 70, 2) or self.valid.shape != (len(self.frame_ids), 70):
            raise ValueError("View2D arrays must be [N, 70, 2] / [N, 70] over N frames")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("View2D needs a positive frame size")

    def select(self, frame_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints and validity at ``frame_ids`` (missing frames are invalid zeros)."""
        wanted = np.asarray(list(frame_ids), dtype=np.int64)
        position = np.searchsorted(self.frame_ids, wanted)
        position = np.clip(position, 0, len(self.frame_ids) - 1)
        hit = self.frame_ids[position] == wanted
        points = np.where(hit[:, None, None], self.points[position], 0.0).astype(np.float32)
        valid = self.valid[position] & hit[:, None]
        return points, valid


def _cache_path(cache_root: Path, dataset: str, subject: str, view: str) -> Path:
    return Path(cache_root) / dataset / f"subject_{subject}" / f"{view}.npz"


def save_view(path: Path, view: View2D) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, frame_ids=view.frame_ids, points=view.points, valid=view.valid, size=np.array([view.width, view.height], dtype=np.int64))
    tmp.replace(path)


def load_view(path: Path, name: str = "") -> View2D:
    payload = np.load(path)
    width, height = (int(v) for v in payload["size"])
    return View2D(frame_ids=np.asarray(payload["frame_ids"], dtype=np.int64), points=np.asarray(payload["points"], dtype=np.float32), valid=np.asarray(payload["valid"], dtype=bool), width=width, height=height, name=name)


# ----------------------------------------------------------------------------- gymnastics
def read_gymnastics_view(person_id: str, view: str, *, sam3d_root: Path = SAM3D_RESULTS_ROOT) -> View2D:
    """Decode every per-frame SAM3D file of one private video (slow: full frames are stored)."""
    base = Path(sam3d_root) / "person" / str(person_id) / view
    files = sorted(base.glob("*_sam3d_body.npz"))
    if not files:
        raise FileNotFoundError(f"no SAM3D frames under {base}")
    frame_ids, points, size = [], [], None
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            output = data["output"].item()
            frame_ids.append(int(np.asarray(output["frame_idx"]).item()))
            points.append(np.asarray(output["pred_keypoints_2d"], dtype=np.float32).reshape(70, 2))
            if size is None:
                frame = output.get("frame")
                if frame is None:
                    raise ValueError(f"{path}: no stored frame to read the image size from")
                size = (int(np.asarray(frame).shape[1]), int(np.asarray(frame).shape[0]))
    order = np.argsort(frame_ids)
    pts = np.stack(points)[order]
    return View2D(frame_ids=np.asarray(frame_ids, dtype=np.int64)[order], points=pts, valid=np.isfinite(pts).all(axis=-1), width=size[0], height=size[1], name=f"gymnastics/{person_id}/{view}")


def gymnastics_view(person_id: str, view: str, *, cache_root: Path = DEFAULT_CACHE_ROOT, sam3d_root: Path = SAM3D_RESULTS_ROOT) -> View2D:
    """Cached :func:`read_gymnastics_view`."""
    path = _cache_path(cache_root, "gymnastics", str(person_id), view)
    if path.is_file():
        return load_view(path, name=f"gymnastics/{person_id}/{view}")
    view2d = read_gymnastics_view(person_id, view, sam3d_root=sam3d_root)
    save_view(path, view2d)
    return view2d


# ----------------------------------------------------------------------------- freeman
FREEMAN_FRAME_SIZE = (1080, 1920)  # (width, height): every FreeMan video is portrait


def freeman_view(benchmark_root: Path, subject_id: int, session_id: str, view_id: str) -> View2D:
    """2D keypoints of one selected FreeMan view from the benchmark cache."""
    from fusion.benchmarks.freeman.sam3d import load_inference

    path = Path(benchmark_root) / "sam3d" / f"subject_{int(subject_id):02d}" / session_id / view_id / "prediction.npz"
    prediction = load_inference(path)
    points = np.asarray(prediction.points2d, dtype=np.float32)
    valid = np.asarray(prediction.valid2d, dtype=bool)
    frame_ids = np.asarray(prediction.frame_ids, dtype=np.int64)
    return View2D(frame_ids=frame_ids, points=points, valid=valid, width=FREEMAN_FRAME_SIZE[0], height=FREEMAN_FRAME_SIZE[1], name=f"freeman/{int(subject_id):02d}/{session_id}/{view_id}")


# ----------------------------------------------------------------------------- fit3d
FIT3D_FRAME_SIZE = (900, 900)


def fit3d_view(derived_root: Path, sequence, camera: str, frame_ids: np.ndarray, *, split: str = "train") -> View2D:
    """2D keypoints of one Fit3D camera from the external per-video cache."""
    from fusion.benchmarks.fit3d.sam3d import load_prediction

    prediction = load_prediction(Path(derived_root), sequence, camera, frame_ids, split=split)
    return View2D(
        frame_ids=np.asarray(prediction.frame_ids, dtype=np.int64),
        points=np.asarray(prediction.points_2d, dtype=np.float32),
        valid=np.asarray(prediction.valid_2d, dtype=bool),
        width=FIT3D_FRAME_SIZE[0],
        height=FIT3D_FRAME_SIZE[1],
        name=f"fit3d/{sequence.subject}/{sequence.action}/{camera}",
    )


def write_manifest(cache_root: Path, dataset: str, entries: list[dict]) -> Path:
    path = Path(cache_root) / dataset / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": dataset, "views": entries}, indent=2), encoding="utf-8")
    return path
