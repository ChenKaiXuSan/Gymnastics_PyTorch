from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gymnastics.benchmarks.unity.dataset import (
    group_evaluation_sequences,
    load_unity_benchmark,
)
from gymnastics.benchmarks.unity.mapping import (
    EVALUATION_JOINT_NAMES,
    map_mhr70_to_unity,
)


UNITY_JOINTS = (
    "Hips",
    "Spine",
    "Chest",
    "UpperChest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "LeftToes",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "RightToes",
)


def _camera(camera_id: str) -> dict[str, object]:
    identity = np.eye(4, dtype=float).reshape(-1).tolist()
    return {
        "camera_id": camera_id,
        "image_width": 16,
        "image_height": 8,
        "camera_to_world": identity,
        "world_to_camera": identity,
        "projection_matrix": identity,
    }


def _frame(
    sample_id: int,
    *,
    sequence_id: str,
    sample_type: str,
    frame_index: int,
    angle: float,
) -> dict[str, object]:
    points_3d = [
        {
            "available": True,
            "world_position_m": {"x": float(i), "y": float(i + 1), "z": float(i + 2)},
        }
        for i in range(22)
    ]
    points_2d = [
        {
            "available": True,
            "x_px": float(i),
            "y_px": float(i + 1),
            "visible": i % 2 == 0,
        }
        for i in range(22)
    ]
    return {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "frame_index": frame_index,
        "sample_type": sample_type,
        "phase": sample_type,
        "time_seconds": frame_index / 60.0,
        "actual_angle_deg": angle,
        "images": {
            "cam0": f"images/cam0/{sample_id:08d}.png",
            "cam1": f"images/cam1/{sample_id:08d}.png",
        },
        "keypoints_3d": points_3d,
        "keypoints_2d_cam0": points_2d,
        "keypoints_2d_cam1": points_2d,
    }


def _write_fixture(root: Path) -> Path:
    (root / "images/cam0").mkdir(parents=True)
    (root / "images/cam1").mkdir(parents=True)
    records = (
        _frame(0, sequence_id="static_000", sample_type="static", frame_index=0, angle=30),
        _frame(1, sequence_id="static_001", sample_type="static", frame_index=0, angle=-30),
        _frame(
            2,
            sequence_id="continuous_left_060_r00",
            sample_type="continuous",
            frame_index=0,
            angle=0,
        ),
    )
    for sample_id in range(3):
        for camera_id in ("cam0", "cam1"):
            (root / f"images/{camera_id}/{sample_id:08d}.png").write_bytes(b"png")
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    (root / "skeleton.json").write_text(
        json.dumps({"joint_names": UNITY_JOINTS}),
        encoding="utf-8",
    )
    (root / "cameras.json").write_text(
        json.dumps({"cameras": [_camera("cam0"), _camera("cam1")]}),
        encoding="utf-8",
    )
    return root


def test_loads_manifest_and_groups_static_samples(tmp_path: Path) -> None:
    benchmark = load_unity_benchmark(_write_fixture(tmp_path))

    assert benchmark.joint_names == UNITY_JOINTS
    assert [frame.sample_id for frame in benchmark.frames] == [0, 1, 2]
    assert benchmark.frames[0].gt_world_m.shape == (22, 3)
    assert benchmark.frames[0].visible["cam0"].shape == (22,)

    groups = group_evaluation_sequences(benchmark)
    assert tuple(groups) == ("static_sweep", "continuous_left_060_r00")
    assert [frame.sample_id for frame in groups["static_sweep"]] == [1, 0]
    assert groups["continuous_left_060_r00"][0].image_paths["cam0"].is_absolute()


def test_maps_exact_sixteen_homologous_joints() -> None:
    points = np.zeros((2, 70, 3), dtype=np.float32)
    for index in range(70):
        points[:, index] = index + 1

    mapped = map_mhr70_to_unity(points)

    assert mapped.points.shape == (2, 16, 3)
    assert mapped.joint_names == EVALUATION_JOINT_NAMES
    np.testing.assert_allclose(
        mapped.points[:, mapped.index("Hips")],
        0.5 * (points[:, 9] + points[:, 10]),
    )
    np.testing.assert_allclose(
        mapped.points[:, mapped.index("LeftToes")],
        0.5 * (points[:, 15] + points[:, 16]),
    )


def test_derived_mapping_requires_every_source_joint() -> None:
    points = np.ones((1, 70, 3), dtype=np.float32)
    valid = np.ones((1, 70), dtype=bool)
    valid[:, 10] = False
    valid[:, 16] = False

    mapped = map_mhr70_to_unity(points, valid)

    assert not mapped.valid[:, mapped.index("Hips")].any()
    assert not mapped.valid[:, mapped.index("LeftToes")].any()
    assert mapped.valid[:, mapped.index("Neck")].all()
