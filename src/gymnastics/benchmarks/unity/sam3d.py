"""Cached SAM3D-Body inference for Unity benchmark images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from .schema import UnityBenchmark


@dataclass(frozen=True)
class InferenceSummary:
    camera_id: str
    expected: int
    completed: int
    reused: int
    failed: tuple[Mapping[str, object], ...]
    summary_path: Path


@dataclass(frozen=True)
class CachedPose:
    camera_id: str
    sample_ids: np.ndarray
    points_3d: np.ndarray
    points_2d: np.ndarray
    valid_3d: np.ndarray
    valid_2d: np.ndarray
    failures: Mapping[int, str]


def _default_estimator_factory(config_path: Path, device: str):
    from omegaconf import OmegaConf

    from gymnastics.sam3d.infer import setup_sam_3d_body

    config = OmegaConf.load(config_path)
    config.infer.gpu = device
    return setup_sam_3d_body(config)


def _read_rgb_image(path: Path) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _cache_path(output_root: Path, camera_id: str, sample_id: int) -> Path:
    return Path(output_root) / camera_id / f"{sample_id:08d}.npz"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _select_largest(outputs) -> Mapping[str, object] | None:
    if not outputs:
        return None

    def area(output) -> float:
        box = np.asarray(output.get("bbox", (0, 0, 0, 0)), dtype=np.float64)
        return float(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))

    return max(outputs, key=area)


def _atomic_pose(
    path: Path,
    *,
    sample_id: int,
    camera_id: str,
    image_path: Path,
    output: Mapping[str, object],
    config_path: Path,
    device: str,
) -> None:
    points_3d = np.asarray(output["pred_keypoints_3d"], dtype=np.float32)
    points_2d = np.asarray(output["pred_keypoints_2d"], dtype=np.float32)
    if points_3d.shape != (70, 3) or points_2d.shape != (70, 2):
        raise ValueError(
            f"sample {sample_id} returned invalid SAM3D shapes "
            f"{points_3d.shape} and {points_2d.shape}"
        )
    valid_3d = np.isfinite(points_3d).all(axis=-1) & np.any(points_3d != 0, axis=-1)
    valid_2d = np.isfinite(points_2d).all(axis=-1) & np.any(points_2d != 0, axis=-1)
    metadata = {
        "sample_id": sample_id,
        "camera_id": camera_id,
        "source_image": str(image_path),
        "sam3d_config": str(config_path),
        "device": device,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pred_keypoints_3d=points_3d,
            pred_keypoints_2d=points_2d,
            valid_3d=valid_3d,
            valid_2d=valid_2d,
            sample_id=np.asarray(sample_id, dtype=np.int64),
            camera_id=np.asarray(camera_id),
            source_image=np.asarray(str(image_path)),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)


def run_sam3d_inference(
    benchmark: UnityBenchmark,
    camera_id: str,
    output_root: Path,
    config_path: Path,
    device: str,
    *,
    force: bool = False,
    sample_ids: Sequence[int] | None = None,
    estimator_factory: Callable[[Path, str], object] | None = None,
) -> InferenceSummary:
    if camera_id not in benchmark.cameras:
        raise ValueError(f"unknown Unity camera: {camera_id}")
    selected_ids = (
        tuple(int(value) for value in sample_ids)
        if sample_ids is not None
        else tuple(frame.sample_id for frame in benchmark.frames)
    )
    by_id = {frame.sample_id: frame for frame in benchmark.frames}
    unknown = [sample_id for sample_id in selected_ids if sample_id not in by_id]
    if unknown:
        raise ValueError(f"unknown Unity sample IDs: {unknown}")

    camera_root = Path(output_root) / camera_id
    summary_path = camera_root / "summary.json"
    prior_failures: dict[int, str] = {}
    if summary_path.is_file() and not force:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        prior_failures = {
            int(item["sample_id"]): str(item["reason"])
            for item in payload.get("failed", [])
        }
    reused = 0
    pending = []
    failures: dict[int, str] = {
        sample_id: reason
        for sample_id, reason in prior_failures.items()
        if sample_id in selected_ids
    }
    for sample_id in selected_ids:
        cache = _cache_path(output_root, camera_id, sample_id)
        if cache.is_file() and not force:
            reused += 1
        elif sample_id not in failures or force:
            pending.append(sample_id)

    estimator = None
    if pending:
        factory = estimator_factory or _default_estimator_factory
        estimator = factory(Path(config_path), device)
        for sample_id in pending:
            frame = by_id[sample_id]
            image_path = frame.image_paths[camera_id]
            image = _read_rgb_image(image_path)
            if image is None:
                failures[sample_id] = "image_read_failed"
                continue
            outputs = estimator.process_one_image(img=image, bboxes=None)
            best = _select_largest(outputs)
            if best is None:
                failures[sample_id] = "no_person_detected"
                continue
            try:
                _atomic_pose(
                    _cache_path(output_root, camera_id, sample_id),
                    sample_id=sample_id,
                    camera_id=camera_id,
                    image_path=image_path,
                    output=best,
                    config_path=Path(config_path),
                    device=device,
                )
            except (KeyError, TypeError, ValueError) as error:
                failures[sample_id] = f"invalid_output:{error}"
                continue
            failures.pop(sample_id, None)

    completed = sum(
        _cache_path(output_root, camera_id, sample_id).is_file()
        for sample_id in selected_ids
    )
    failed_records = tuple(
        {"sample_id": sample_id, "reason": failures[sample_id]}
        for sample_id in sorted(failures)
        if sample_id in selected_ids
    )
    summary = InferenceSummary(
        camera_id=camera_id,
        expected=len(selected_ids),
        completed=completed,
        reused=reused,
        failed=failed_records,
        summary_path=summary_path,
    )
    _atomic_json(
        summary_path,
        {
            **asdict(summary),
            "summary_path": str(summary_path),
            "sam3d_config": str(config_path),
            "device": device,
        },
    )
    del estimator
    return summary


def load_sam3d_camera_cache(
    root: Path, camera_id: str, sample_ids: Sequence[int]
) -> CachedPose:
    requested = np.asarray(tuple(int(value) for value in sample_ids), dtype=np.int64)
    points_3d = np.zeros((len(requested), 70, 3), dtype=np.float32)
    points_2d = np.zeros((len(requested), 70, 2), dtype=np.float32)
    valid_3d = np.zeros((len(requested), 70), dtype=bool)
    valid_2d = np.zeros((len(requested), 70), dtype=bool)
    summary_path = Path(root) / camera_id / "summary.json"
    failures: dict[int, str] = {}
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        failures = {
            int(item["sample_id"]): str(item["reason"])
            for item in payload.get("failed", [])
        }
    missing: list[int] = []
    for row, sample_id in enumerate(requested):
        path = _cache_path(root, camera_id, int(sample_id))
        if not path.is_file():
            if int(sample_id) not in failures:
                missing.append(int(sample_id))
            continue
        with np.load(path, allow_pickle=False) as data:
            if int(data["sample_id"]) != int(sample_id):
                raise ValueError(f"cache identity mismatch: {path}")
            points_3d[row] = np.asarray(data["pred_keypoints_3d"], dtype=np.float32)
            points_2d[row] = np.asarray(data["pred_keypoints_2d"], dtype=np.float32)
            valid_3d[row] = np.asarray(data["valid_3d"], dtype=bool)
            valid_2d[row] = np.asarray(data["valid_2d"], dtype=bool)
    if missing:
        raise FileNotFoundError(
            f"missing SAM3D {camera_id} cache without failure record: {missing}"
        )
    selected_failures = {
        int(sample_id): failures[int(sample_id)]
        for sample_id in requested
        if int(sample_id) in failures
    }
    return CachedPose(
        camera_id=camera_id,
        sample_ids=requested,
        points_3d=points_3d,
        points_2d=points_2d,
        valid_3d=valid_3d,
        valid_2d=valid_2d,
        failures=MappingProxyType(selected_failures),
    )
