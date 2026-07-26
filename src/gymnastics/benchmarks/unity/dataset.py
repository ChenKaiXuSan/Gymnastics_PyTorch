"""Read and validate the Unity benchmark manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .schema import UnityBenchmark, UnityCamera, UnityFrame


def _required_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"missing Unity benchmark file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _camera(record: dict[str, object]) -> UnityCamera:
    return UnityCamera(
        camera_id=str(record["camera_id"]),
        image_size=(int(record["image_width"]), int(record["image_height"])),
        camera_to_world=np.asarray(record["camera_to_world"], dtype=np.float64).reshape(4, 4),
        world_to_camera=np.asarray(record["world_to_camera"], dtype=np.float64).reshape(4, 4),
        clip_projection=np.asarray(record["projection_matrix"], dtype=np.float64).reshape(4, 4),
    )


def _world_points(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.full((len(records), 3), np.nan, dtype=np.float32)
    available = np.zeros((len(records),), dtype=bool)
    for index, record in enumerate(records):
        available[index] = bool(record.get("available", False))
        position = record.get("world_position_m")
        if available[index] and isinstance(position, dict):
            points[index] = (
                float(position["x"]),
                float(position["y"]),
                float(position["z"]),
            )
    return points, available


def _image_points(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.full((len(records), 2), np.nan, dtype=np.float32)
    visible = np.zeros((len(records),), dtype=bool)
    for index, record in enumerate(records):
        if bool(record.get("available", False)):
            points[index] = (float(record["x_px"]), float(record["y_px"]))
        visible[index] = bool(record.get("visible", False))
    return points, visible


def load_unity_benchmark(
    root: Path,
    *,
    sequence_ids: Sequence[str] | None = None,
) -> UnityBenchmark:
    root = Path(root).resolve()
    requested_sequences = (
        None
        if sequence_ids is None
        else frozenset(str(value) for value in sequence_ids)
    )
    if requested_sequences is not None and not requested_sequences:
        raise ValueError("sequence_ids must not be empty")
    skeleton = _required_json(root / "skeleton.json")
    camera_payload = _required_json(root / "cameras.json")
    if not isinstance(skeleton, dict) or not isinstance(camera_payload, dict):
        raise ValueError("Unity skeleton and camera payloads must be objects")
    joint_names = tuple(str(value) for value in skeleton["joint_names"])
    cameras = {
        camera.camera_id: camera
        for camera in (_camera(record) for record in camera_payload["cameras"])
    }
    if tuple(cameras) != ("cam0", "cam1"):
        raise ValueError("Unity benchmark must define cam0 and cam1")

    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Unity benchmark file: {manifest_path}")
    frames: list[UnityFrame] = []
    previous_id = -1
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = int(record["sample_id"])
        sequence_id = str(record["sequence_id"])
        if sample_id <= previous_id:
            raise ValueError(
                f"sample_id must be unique and increasing at line {line_number}"
            )
        previous_id = sample_id
        if (
            requested_sequences is not None
            and sequence_id not in requested_sequences
        ):
            continue
        images = {
            camera_id: (root / str(record["images"][camera_id])).resolve()
            for camera_id in cameras
        }
        missing = [str(path) for path in images.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing Unity images: {missing}")
        world, available = _world_points(record["keypoints_3d"])
        if len(world) != len(joint_names):
            raise ValueError(f"sample {sample_id} has wrong 3D joint count")
        pixels: dict[str, np.ndarray] = {}
        visible: dict[str, np.ndarray] = {}
        for camera_id in cameras:
            camera_points, camera_visible = _image_points(
                record[f"keypoints_2d_{camera_id}"]
            )
            if len(camera_points) != len(joint_names):
                raise ValueError(f"sample {sample_id} has wrong {camera_id} joint count")
            pixels[camera_id] = camera_points
            visible[camera_id] = camera_visible
        frames.append(
            UnityFrame(
                sample_id=sample_id,
                sequence_id=sequence_id,
                frame_index=int(record["frame_index"]),
                sample_type=str(record["sample_type"]),
                phase=str(record.get("phase", "")),
                time_seconds=float(record["time_seconds"]),
                actual_angle_deg=float(record["actual_angle_deg"]),
                image_paths=images,
                gt_world_m=world,
                gt_available=available,
                gt_pixels=pixels,
                visible=visible,
            )
        )
    return UnityBenchmark(root, joint_names, cameras, tuple(frames))


def group_evaluation_sequences(
    benchmark: UnityBenchmark,
) -> dict[str, tuple[UnityFrame, ...]]:
    grouped: dict[str, list[UnityFrame]] = {}
    static = [frame for frame in benchmark.frames if frame.sample_type == "static"]
    if static:
        grouped["static_sweep"] = sorted(static, key=lambda frame: frame.actual_angle_deg)
    for frame in benchmark.frames:
        if frame.sample_type == "static":
            continue
        grouped.setdefault(frame.sequence_id, []).append(frame)
    for sequence_id in tuple(grouped):
        if sequence_id != "static_sweep":
            grouped[sequence_id].sort(key=lambda frame: frame.frame_index)
    return {key: tuple(value) for key, value in grouped.items()}
