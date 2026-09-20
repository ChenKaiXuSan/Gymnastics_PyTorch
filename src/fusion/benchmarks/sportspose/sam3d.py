"""SAM3D-Body inference on the two selected views of every SportsPose clip.

Cache layout (``paths.sam3d_cache_root``)::

    <day>/<S<nn>>/<activity>/<clip_id>/<view>.npz
        frame_ids   [N]        reference frame index of each processed frame
        video_frames [N]       decoded video frame index (timing.video_index)
        points_3d   [N, 70, 3] MHR70 camera-frame keypoints (metres)
        points_2d   [N, 70, 2] image keypoints on the upright frame
        valid_3d    [N, 70]    finite and non-zero
        valid_2d    [N, 70]
    <day>/<S<nn>>/<activity>/<clip_id>/<view>.json   identity + failed frames

Frames are decoded from the MJPEG video, rotated ``rot90_clockwise`` times
clockwise (the release stores the sensor orientation) and processed one at a
time; the largest detected person is kept. ``frame_stride`` thins the 90 fps
reference timeline (default 3 -> 30 fps).

A second source is the per-video cache of ``derived/normal_camera/sam3d_sportspose``
(every camera, every frame, every detected person, written outside this
repository): :func:`load_derived_prediction` converts one of those files
into the same :class:`ViewPrediction` (rank-0 person, requested frames only),
so the trial builder and the cycle records do not care which source produced
the poses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .dataset import load_timing
from .schema import SportsPoseCamera, SportsPoseClip

EstimatorFactory = Callable[[Path, int], Any]


@dataclass(frozen=True)
class ViewPrediction:
    """One cached view of one clip."""

    clip_id: str
    view_id: str
    frame_ids: np.ndarray
    video_frames: np.ndarray
    points_3d: np.ndarray
    points_2d: np.ndarray
    valid_3d: np.ndarray
    valid_2d: np.ndarray
    failed_frames: tuple[int, ...]


@dataclass(frozen=True)
class ClipInferenceSummary:
    clip_id: str
    view_id: str
    frames: int
    failed: int
    reused: bool
    path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(cache_root: Path, clip: SportsPoseClip, view_id: str) -> Path:
    return Path(cache_root) / clip.day / clip.subject / clip.activity / clip.clip_id / f"{view_id}.npz"


def _identity(clip: SportsPoseClip, view_id: str, config_sha: str, frame_stride: int) -> dict[str, Any]:
    return {"clip_id": clip.clip_id, "day": clip.day, "subject": clip.subject, "activity": clip.activity, "view_id": view_id, "frames": clip.frames, "frame_stride": int(frame_stride), "sam3d_config_sha256": config_sha}


def load_prediction(path: Path) -> ViewPrediction:
    path = Path(path)
    payload = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return ViewPrediction(
        clip_id=str(meta["identity"]["clip_id"]),
        view_id=str(meta["identity"]["view_id"]),
        frame_ids=np.asarray(payload["frame_ids"], dtype=np.int64),
        video_frames=np.asarray(payload["video_frames"], dtype=np.int64),
        points_3d=np.asarray(payload["points_3d"], dtype=np.float32),
        points_2d=np.asarray(payload["points_2d"], dtype=np.float32),
        valid_3d=np.asarray(payload["valid_3d"], dtype=bool),
        valid_2d=np.asarray(payload["valid_2d"], dtype=bool),
        failed_frames=tuple(int(f) for f in meta.get("failed_frames", ())),
    )


def derived_path(derived_root: Path, clip: SportsPoseClip, camera_index: int) -> Path:
    """``<derived_root>/<day>/<S>/<Video_dir>/CAM<k>.npz`` of the external per-video cache."""
    return Path(derived_root) / clip.day / clip.subject / clip.video_dir.name / f"CAM{int(camera_index)}.npz"


def load_derived_prediction(derived_root: Path, clip: SportsPoseClip, camera: SportsPoseCamera, frame_ids: np.ndarray, video_frames: np.ndarray) -> ViewPrediction:
    """Rank-0 person of the external per-video cache at the requested frames.

    Args:
        derived_root: Root of ``sam3d_sportspose`` (see :func:`derived_path`).
        clip: The clip.
        camera: The camera (its index names the file).
        frame_ids: Reference frame indices to keep.
        video_frames: Video frame index of each reference frame (``timing.video_index``).

    Raises:
        FileNotFoundError: If the per-video file is missing.
    """
    path = derived_path(derived_root, clip, camera.index)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    rows_frames = np.asarray(payload["frame_ids"], dtype=np.int64)
    ranks = np.asarray(payload["person_rank"], dtype=np.int64)
    primary = ranks == 0
    row_of = {int(f): i for i, f in zip(np.flatnonzero(primary), rows_frames[primary])}
    n = len(frame_ids)
    points_3d = np.zeros((n, 70, 3), dtype=np.float32)
    points_2d = np.zeros((n, 70, 2), dtype=np.float32)
    valid_3d = np.zeros((n, 70), dtype=bool)
    valid_2d = np.zeros((n, 70), dtype=bool)
    failed: list[int] = []
    src3 = np.asarray(payload["points3d"], dtype=np.float32)
    src2 = np.asarray(payload["points2d"], dtype=np.float32)
    for slot, video_frame in enumerate(np.asarray(video_frames, dtype=np.int64)):
        row = row_of.get(int(video_frame))
        if row is None:
            failed.append(int(frame_ids[slot]))
            continue
        xyz, xy = src3[row], src2[row]
        ok3 = np.isfinite(xyz).all(axis=-1) & np.any(xyz != 0, axis=-1)
        ok2 = np.isfinite(xy).all(axis=-1)
        points_3d[slot] = np.where(ok3[:, None], xyz, 0.0)
        points_2d[slot] = np.where(ok2[:, None], xy, 0.0)
        valid_3d[slot] = ok3
        valid_2d[slot] = ok2
    return ViewPrediction(clip_id=clip.clip_id, view_id=camera.view_id, frame_ids=np.asarray(frame_ids, dtype=np.int64), video_frames=np.asarray(video_frames, dtype=np.int64), points_3d=points_3d, points_2d=points_2d, valid_3d=valid_3d, valid_2d=valid_2d, failed_frames=tuple(failed))


def prediction_is_current(path: Path, identity: Mapping[str, Any], *, accepted_config_hashes: Sequence[str] = ()) -> bool:
    meta_path = Path(path).with_suffix(".json")
    if not Path(path).is_file() or not meta_path.is_file():
        return False
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8")).get("identity", {})
    except (OSError, ValueError):
        return False
    accepted = {identity["sam3d_config_sha256"], *accepted_config_hashes}
    return all(stored.get(k) == v for k, v in identity.items() if k != "sam3d_config_sha256") and stored.get("sam3d_config_sha256") in accepted


def _default_estimator_factory(config_path: Path, device: int) -> Any:
    from common.config import load_config as load_project_config
    from pose_estimation.infer import setup_sam_3d_body

    return setup_sam_3d_body(load_project_config(Path(config_path), [f"infer.gpu={int(device)}"]))


def _largest_person(outputs: Any) -> Mapping[str, Any] | None:
    best, best_area = None, -1.0
    for item in outputs or ():
        if not isinstance(item, Mapping):
            continue
        bbox = np.asarray(item.get("bbox", (0, 0, 0, 0)), dtype=np.float64).reshape(-1)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            continue
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if area > best_area:
            best, best_area = item, area
    return best


def _pose_array(output: Mapping[str, Any], field: str, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(output[field], dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.shape != shape:
        raise ValueError(f"SAM3D {field} must have shape {shape}, got {value.shape}")
    return value


def upright(frame_bgr: np.ndarray, camera: SportsPoseCamera) -> np.ndarray:
    """Rotate a decoded frame clockwise ``rot90_clockwise`` times."""
    k = int(camera.rot90_clockwise) % 4
    return np.ascontiguousarray(np.rot90(frame_bgr, k=-k)) if k else frame_bgr


def stream_clip_view(estimator: Any, clip: SportsPoseClip, camera: SportsPoseCamera, frame_ids: np.ndarray, video_frames: np.ndarray) -> ViewPrediction:
    """Run the estimator on the requested frames of one video."""
    wanted = {int(v): i for i, v in enumerate(video_frames)}
    n = len(frame_ids)
    points_3d = np.zeros((n, 70, 3), dtype=np.float32)
    points_2d = np.zeros((n, 70, 2), dtype=np.float32)
    valid_3d = np.zeros((n, 70), dtype=bool)
    valid_2d = np.zeros((n, 70), dtype=bool)
    failed: list[int] = []
    seen = 0
    path = clip.video_path(camera.index)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open SportsPose video {path}")
    try:
        index = 0
        last = int(video_frames.max())
        while index <= last:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            slot = wanted.get(index)
            if slot is not None:
                seen += 1
                frame_rgb = cv2.cvtColor(upright(frame_bgr, camera), cv2.COLOR_BGR2RGB)
                best = _largest_person(estimator.process_one_image(img=frame_rgb, bboxes=None))
                if best is None:
                    failed.append(int(frame_ids[slot]))
                else:
                    xyz = _pose_array(best, "pred_keypoints_3d", (70, 3))
                    xy = _pose_array(best, "pred_keypoints_2d", (70, 2))
                    ok3 = np.isfinite(xyz).all(axis=-1) & np.any(xyz != 0, axis=-1)
                    ok2 = np.isfinite(xy).all(axis=-1)
                    points_3d[slot] = np.where(ok3[:, None], xyz, 0.0)
                    points_2d[slot] = np.where(ok2[:, None], xy, 0.0)
                    valid_3d[slot] = ok3
                    valid_2d[slot] = ok2
            index += 1
    finally:
        capture.release()
    if seen != n:
        missing = [int(frame_ids[i]) for v, i in wanted.items() if v >= index]
        failed.extend(missing)
    return ViewPrediction(clip_id=clip.clip_id, view_id=camera.view_id, frame_ids=np.asarray(frame_ids, dtype=np.int64), video_frames=np.asarray(video_frames, dtype=np.int64), points_3d=points_3d, points_2d=points_2d, valid_3d=valid_3d, valid_2d=valid_2d, failed_frames=tuple(sorted(set(failed))))


def _write_prediction(path: Path, prediction: ViewPrediction, identity: Mapping[str, Any], *, config_path: Path, device: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez appends .npz to other names
    np.savez_compressed(tmp, frame_ids=prediction.frame_ids, video_frames=prediction.video_frames, points_3d=prediction.points_3d, points_2d=prediction.points_2d, valid_3d=prediction.valid_3d, valid_2d=prediction.valid_2d)
    tmp.replace(path)
    meta = {"identity": dict(identity), "sam3d_config": str(config_path), "device": int(device), "failed_frames": list(prediction.failed_frames), "frames_processed": int(len(prediction.frame_ids))}
    meta_tmp = path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta_tmp.replace(path.with_suffix(".json"))


def infer_clips(
    clips: Sequence[SportsPoseClip],
    views_by_group: Mapping[tuple[str, str], tuple[str, str]],
    cameras_by_day_subject: Mapping[tuple[str, str], Mapping[str, SportsPoseCamera]],
    *,
    cache_root: Path,
    config_path: Path,
    device: int,
    frame_stride: int = 3,
    force: bool = False,
    accepted_config_hashes: Sequence[str] = (),
    estimator_factory: EstimatorFactory | None = None,
) -> list[ClipInferenceSummary]:
    """Cache the two selected views of every clip (existing valid caches are reused).

    Args:
        clips: Clips to process.
        views_by_group: ``(subject_key, sequence_key) -> (view_a, view_b)``.
        cameras_by_day_subject: ``(day, subject) -> {view_id: camera}`` (calibrations are per day).
        cache_root: Cache root (see module docstring).
        config_path: SAM3D-Body Hydra config file.
        device: CUDA device index.
        frame_stride: Keep every ``frame_stride``-th reference frame.
        force: Recompute even when a current cache exists.
        accepted_config_hashes: Other SAM3D config hashes treated as equivalent.
        estimator_factory: ``(config_path, device) -> estimator`` (tests).
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    config_sha = _sha256(config_path)
    estimator = None
    summaries: list[ClipInferenceSummary] = []
    for clip in clips:
        views = views_by_group[(clip.subject_key, clip.sequence_key)]
        cameras = cameras_by_day_subject[(clip.day, clip.subject)]
        timing = load_timing(clip)
        frame_ids = np.arange(0, clip.frames, frame_stride, dtype=np.int64)
        for view_id in views:
            camera = cameras[view_id]
            path = cache_path(cache_root, clip, view_id)
            identity = _identity(clip, view_id, config_sha, frame_stride)
            if not force and prediction_is_current(path, identity, accepted_config_hashes=accepted_config_hashes):
                meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                summaries.append(ClipInferenceSummary(clip.clip_id, view_id, len(frame_ids), len(meta.get("failed_frames", ())), True, str(path)))
                continue
            if estimator is None:
                estimator = (estimator_factory or _default_estimator_factory)(Path(config_path), int(device))
            video_frames = timing["video_index"][camera.index][frame_ids]
            prediction = stream_clip_view(estimator, clip, camera, frame_ids, video_frames)
            _write_prediction(path, prediction, identity, config_path=Path(config_path), device=int(device))
            summaries.append(ClipInferenceSummary(clip.clip_id, view_id, len(frame_ids), len(prediction.failed_frames), False, str(path)))
    return summaries


def summary_rows(summaries: Sequence[ClipInferenceSummary]) -> list[dict[str, Any]]:
    return [asdict(s) for s in summaries]
