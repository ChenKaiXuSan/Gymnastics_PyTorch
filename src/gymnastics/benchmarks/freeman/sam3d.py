"""Streaming SAM3D inference and resumable FreeMan caches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from gymnastics.common.paths import PROJECT_ROOT

from .schema import (
    FreeManSession,
    InferenceArtifact,
    InferenceIdentity,
    SelectedPair,
    ViewPrediction,
)


EstimatorFactory = Callable[[Mapping[str, Any]], Any]
_MAX_TRAILING_DECODE_SHORTFALL = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_path(config: Mapping[str, Any]) -> Path:
    path = Path(config["sam3d"]["config"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _identity(
    session: FreeManSession,
    view_id: str,
    config: Mapping[str, Any],
    source_frame_count: int,
) -> InferenceIdentity:
    return InferenceIdentity(
        session_id=session.session_id,
        subject_id=session.subject_id,
        fps=session.fps,
        view_id=view_id,
        source_video_sha256=_sha256(session.video_paths[view_id]),
        source_frame_count=int(source_frame_count),
        frame_stride=int(config["dataset"]["frame_stride"]),
        sam3d_config_sha256=_sha256(_config_path(config)),
        checkpoint_id=str(config["sam3d"]["checkpoint_id"]),
    )


def _prediction_path(
    session: FreeManSession,
    view_id: str,
    config: Mapping[str, Any],
) -> Path:
    return (
        Path(config["paths"]["output_root"]).resolve()
        / "sam3d"
        / f"subject_{session.subject_id:02d}"
        / session.session_id
        / view_id
        / "prediction.npz"
    )


def _load_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name("metadata.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SAM3D metadata must be a mapping")
    return payload


def load_inference(path: Path) -> ViewPrediction:
    """Load and validate one compact SAM3D view prediction."""
    prediction_path = Path(path).resolve()
    metadata = _load_metadata(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as data:
        required = {
            "frame_ids",
            "points3d",
            "points2d",
            "valid3d",
            "valid2d",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"SAM3D cache missing arrays: {sorted(missing)}")
        prediction = ViewPrediction(
            session_id=str(metadata["session_id"]),
            subject_id=int(metadata["subject_id"]),
            fps=float(metadata["fps"]),
            view_id=str(metadata["view_id"]),
            frame_ids=np.asarray(data["frame_ids"]),
            points3d=np.asarray(data["points3d"]),
            points2d=np.asarray(data["points2d"]),
            valid3d=np.asarray(data["valid3d"]),
            valid2d=np.asarray(data["valid2d"]),
            metadata=metadata,
        )
    if (
        np.any(prediction.valid3d & ~np.isfinite(prediction.points3d).all(axis=-1))
        or np.any(prediction.valid2d & ~np.isfinite(prediction.points2d).all(axis=-1))
    ):
        raise ValueError("SAM3D cache marks non-finite keypoints valid")
    return prediction


def validate_inference(
    path: Path,
    expected: InferenceIdentity | None = None,
) -> bool:
    """Return whether an artifact is readable and matches its expected identity."""
    return _validate_inference(path, expected, allow_partial=False)


def _validate_inference(
    path: Path,
    expected: InferenceIdentity | None,
    *,
    allow_partial: bool,
) -> bool:
    try:
        prediction_path = Path(path).resolve()
        if not allow_partial and prediction_path.parent.name.endswith(".partial"):
            return False
        prediction = load_inference(prediction_path)
        metadata = dict(prediction.metadata)
        identity = metadata.get("identity")
        if not isinstance(identity, dict):
            return False
        if expected is not None and identity != asdict(expected):
            return False
        if (
            prediction.session_id != identity.get("session_id")
            or prediction.subject_id != identity.get("subject_id")
            or prediction.view_id != identity.get("view_id")
        ):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _default_estimator_factory(config: Mapping[str, Any]) -> Any:
    from gymnastics.common.config import load_config as load_project_config
    from gymnastics.sam3d.infer import setup_sam_3d_body

    device = int(config["sam3d"]["device"])
    estimator_config = load_project_config(
        _config_path(config),
        [f"infer.gpu={device}"],
    )
    return setup_sam_3d_body(estimator_config)


def _best_person(outputs: Any) -> Mapping[str, Any] | None:
    if not outputs:
        return None
    candidates = [item for item in outputs if isinstance(item, Mapping)]
    if not candidates:
        return None

    def area(item: Mapping[str, Any]) -> float:
        bbox = np.asarray(item.get("bbox", (0, 0, 0, 0)), dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            return float("-inf")
        return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))

    selected = max(candidates, key=area)
    return selected if np.isfinite(area(selected)) else None


def _pose_array(
    output: Mapping[str, Any],
    field: str,
    shape: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(output[field], dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.shape != shape:
        raise ValueError(f"SAM3D {field} must have shape {shape}, got {value.shape}")
    return value


def _stream_view(
    estimator: Any,
    session: FreeManSession,
    view_id: str,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    points3d: list[np.ndarray] = []
    points2d: list[np.ndarray] = []
    valid3d: list[np.ndarray] = []
    valid2d: list[np.ndarray] = []
    failed: list[int] = []
    wanted = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    capture = cv2.VideoCapture(str(session.video_paths[view_id]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open FreeMan video {session.video_paths[view_id]}")
    frame_index = 0
    try:
        while frame_index <= int(frame_ids[-1]):
            success, frame_bgr = capture.read()
            if not success:
                raise RuntimeError(
                    f"internal video decode failure at frame {frame_index}: "
                    f"{session.video_paths[view_id]}"
                )
            if frame_index in wanted:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                selected = _best_person(
                    estimator.process_one_image(img=frame_rgb, bboxes=None)
                )
                if selected is None:
                    xyz = np.zeros((70, 3), dtype=np.float32)
                    xy = np.zeros((70, 2), dtype=np.float32)
                    xyz_valid = np.zeros(70, dtype=bool)
                    xy_valid = np.zeros(70, dtype=bool)
                    failed.append(frame_index)
                else:
                    xyz = _pose_array(
                        selected,
                        "pred_keypoints_3d",
                        (70, 3),
                    )
                    xy = _pose_array(
                        selected,
                        "pred_keypoints_2d",
                        (70, 2),
                    )
                    xyz_valid = np.isfinite(xyz).all(axis=-1) & np.any(
                        xyz != 0,
                        axis=-1,
                    )
                    xy_valid = np.isfinite(xy).all(axis=-1)
                    xyz = np.where(xyz_valid[:, None], xyz, 0)
                    xy = np.where(xy_valid[:, None], xy, 0)
                points3d.append(xyz)
                points2d.append(xy)
                valid3d.append(xyz_valid)
                valid2d.append(xy_valid)
            frame_index += 1
    finally:
        capture.release()
    if len(points3d) != len(frame_ids):
        raise RuntimeError(
            f"decoded {len(points3d)} requested frames, expected {len(frame_ids)}"
        )
    return (
        np.stack(points3d),
        np.stack(points2d),
        np.stack(valid3d),
        np.stack(valid2d),
        failed,
    )


def _decodable_frame_count(path: Path, expected_frames: int) -> int:
    """Count readable frames, tolerating only a small container-tail mismatch."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open FreeMan video {path}")
    decoded = 0
    try:
        while decoded < expected_frames:
            success, _ = capture.read()
            if not success:
                break
            decoded += 1
    finally:
        capture.release()
    if decoded <= 0:
        raise RuntimeError(f"FreeMan video has no decodable frames: {path}")
    shortfall = expected_frames - decoded
    if shortfall > _MAX_TRAILING_DECODE_SHORTFALL:
        raise RuntimeError(
            f"video decode stopped {shortfall} frames before the expected end "
            f"at frame {decoded}: {path}"
        )
    return decoded


def _publish_prediction(
    path: Path,
    prediction: ViewPrediction,
    identity: InferenceIdentity,
    pair: SelectedPair,
    failed_frames: Sequence[int],
) -> Path:
    final_dir = path.parent
    partial = final_dir.with_name(final_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    partial.mkdir(parents=True)
    metadata = {
        "session_id": prediction.session_id,
        "subject_id": prediction.subject_id,
        "fps": prediction.fps,
        "view_id": prediction.view_id,
        "identity": asdict(identity),
        "selected_pair": asdict(pair),
        "failed_frames": [int(value) for value in failed_frames],
        "completion_status": "complete",
    }
    np.savez_compressed(
        partial / "prediction.npz",
        frame_ids=prediction.frame_ids,
        points3d=prediction.points3d,
        points2d=prediction.points2d,
        valid3d=prediction.valid3d,
        valid2d=prediction.valid2d,
    )
    (partial / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_prediction = partial / "prediction.npz"
    if not _validate_inference(
        temporary_prediction,
        identity,
        allow_partial=True,
    ):
        raise RuntimeError(
            f"new SAM3D cache failed validation: {temporary_prediction}"
        )
    partial.replace(final_dir)
    return path


def _artifact(path: Path) -> InferenceArtifact:
    prediction = load_inference(path)
    return InferenceArtifact(
        path=path,
        session_id=prediction.session_id,
        view_id=prediction.view_id,
        frames=len(prediction.frame_ids),
        valid_frames=int(prediction.valid3d.any(axis=1).sum()),
    )


def infer_subject_sessions(
    sessions: Sequence[FreeManSession],
    pairs: Mapping[str, SelectedPair],
    config: Mapping[str, Any],
    *,
    estimator_factory: EstimatorFactory | None = None,
) -> tuple[InferenceArtifact, ...]:
    """Stream selected videos and reuse every identity-valid view cache."""
    if not sessions:
        return ()
    subjects = {session.subject_id for session in sessions}
    if len(subjects) != 1:
        raise ValueError("infer_subject_sessions accepts exactly one subject")
    frame_stride = int(config["dataset"]["frame_stride"])
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    factory = estimator_factory or _default_estimator_factory
    estimator: Any | None = None
    artifacts: list[InferenceArtifact] = []
    for session in sorted(sessions, key=lambda item: (item.fps, item.session_id)):
        pair = pairs.get(session.session_id)
        if pair is None or pair.session_id != session.session_id:
            raise ValueError(f"missing selected pair for {session.session_id}")
        selected_views = (pair.view_a, pair.view_b)
        for view_id in selected_views:
            if view_id not in session.video_paths:
                raise ValueError(
                    f"selected view {view_id} is missing from {session.session_id}"
                )
        expected_frames = len(session.frame_ids)
        decodable_counts = {
            view_id: _decodable_frame_count(
                session.video_paths[view_id],
                expected_frames,
            )
            for view_id in selected_views
        }
        common_frame_count = min(decodable_counts.values())
        common_frame_ids = session.frame_ids[
            session.frame_ids < common_frame_count
        ][::frame_stride]
        for view_id in selected_views:
            identity = _identity(
                session,
                view_id,
                config,
                common_frame_count,
            )
            path = _prediction_path(session, view_id, config)
            if validate_inference(path, identity):
                artifacts.append(_artifact(path))
                continue
            if estimator is None:
                estimator = factory(config)
            frame_ids = np.array(
                common_frame_ids,
                dtype=np.int64,
                copy=True,
            )
            xyz, xy, xyz_valid, xy_valid, failed = _stream_view(
                estimator,
                session,
                view_id,
                frame_ids,
            )
            prediction = ViewPrediction(
                session_id=session.session_id,
                subject_id=session.subject_id,
                fps=float(session.fps),
                view_id=view_id,
                frame_ids=frame_ids,
                points3d=xyz,
                points2d=xy,
                valid3d=xyz_valid,
                valid2d=xy_valid,
                metadata={},
            )
            _publish_prediction(
                path,
                prediction,
                identity,
                pair,
                failed,
            )
            artifacts.append(_artifact(path))
    return tuple(artifacts)
