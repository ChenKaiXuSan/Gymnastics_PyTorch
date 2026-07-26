from __future__ import annotations

import numpy as np
import pytest

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.fusion import (
    build_pose_pair_trial,
    fuse_deterministic_sequence,
    load_rotation_aware_model,
    run_deterministic_fusion,
    run_rotation_aware_fusion,
)
from gymnastics.benchmarks.unity.schema import UnityBenchmark
from gymnastics.fusion.deterministic.experiment_matrix import (
    ALL_METHODS,
    current_body_average,
)


def _pose_pair(frames: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    face = rng.normal(size=(frames, 70, 3)).astype(np.float32)
    face[:, 9] = (-0.2, 0.9, 0.0)
    face[:, 10] = (0.2, 0.9, 0.0)
    face[:, 5] = (-0.3, 1.4, 0.0)
    face[:, 6] = (0.3, 1.4, 0.0)
    side = face @ np.asarray(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    side += np.asarray((2.0, 0.3, -1.0), dtype=np.float32)
    return face, side


@pytest.mark.parametrize("method", ALL_METHODS)
def test_runs_every_named_deterministic_method(method: str) -> None:
    face, side = _pose_pair()

    output, metadata = fuse_deterministic_sequence(
        method,
        face,
        side,
        leaky_weights=np.full((70, 2), 0.5, dtype=np.float32)
        if method == "sim3_face_stable_joint_weight"
        else None,
    )

    assert output.shape == face.shape
    assert np.isfinite(output).all()
    assert metadata["method"] == method


def test_body_average_matches_existing_implementation() -> None:
    face, side = _pose_pair()

    output, _ = fuse_deterministic_sequence("avg_body_current", face, side)

    np.testing.assert_allclose(output, current_body_average(face, side))


def test_unknown_deterministic_method_is_rejected() -> None:
    face, side = _pose_pair()

    with pytest.raises(ValueError, match="unsupported deterministic method"):
        fuse_deterministic_sequence("invented", face, side)


def _write_camera_cache(root, camera_id: str, sample_ids: list[int], points) -> None:
    camera_root = root / camera_id
    camera_root.mkdir(parents=True)
    for sample_id, pose in zip(sample_ids, points):
        np.savez_compressed(
            camera_root / f"{sample_id:08d}.npz",
            pred_keypoints_3d=pose,
            pred_keypoints_2d=np.ones((70, 2), dtype=np.float32),
            valid_3d=np.ones((70,), dtype=bool),
            valid_2d=np.ones((70,), dtype=bool),
            sample_id=np.asarray(sample_id),
        )


def test_runs_deterministic_adapter_and_saves_sequence(tmp_path) -> None:
    full = load_unity_benchmark(
        "/home/data/xchen/gymnastics/unity_benchmark"
    )
    benchmark = UnityBenchmark(
        full.root, full.joint_names, full.cameras, full.frames[:5]
    )
    sample_ids = [frame.sample_id for frame in benchmark.frames]
    face, side = _pose_pair(frames=5)
    cache_root = tmp_path / "sam3d"
    _write_camera_cache(cache_root, "cam0", sample_ids, face)
    _write_camera_cache(cache_root, "cam1", sample_ids, side)

    sequences = run_deterministic_fusion(
        benchmark,
        cache_root,
        tmp_path / "fusion",
        methods=("avg_body_current",),
    )

    assert len(sequences) == 1
    assert sequences[0].sequence_id == "static_sweep"
    assert sequences[0].points.shape == (5, 70, 3)
    assert (
        tmp_path
        / "fusion/deterministic/avg_body_current/static_sweep.npz"
    ).is_file()


def test_builds_zero_offset_rotation_aware_trial_without_unity_gt() -> None:
    face, side = _pose_pair(frames=5)

    trial = build_pose_pair_trial(
        "static_sweep",
        np.arange(5, dtype=np.int32),
        face,
        side,
        np.ones((5, 70), dtype=bool),
        np.ones((5, 70), dtype=bool),
        fps=60.0,
    )

    assert trial.person_id == "unity"
    assert trial.trial_id == "static_sweep"
    assert trial.face_map.tolist() == list(range(5))
    assert trial.side_map.tolist() == list(range(5))
    assert trial.source_metadata["offset_cam1_to_cam0"] == 0
    assert "gt" not in repr(dict(trial.source_metadata)).lower()


def test_loads_existing_rotation_checkpoint_metadata() -> None:
    checkpoint = (
        "local/runs/fuse_rotation_aware/runs/"
        "all137_a4_e100_seed0/checkpoints/best.pt"
    )

    loaded = load_rotation_aware_model(
        checkpoint,
        "configs/fusion/skeleton_mhr70.yaml",
        "cpu",
    )

    assert loaded.ablation == "A4"
    assert loaded.hidden_channels == 128
    assert loaded.checkpoint_path.name == "best.pt"
    assert len(loaded.checkpoint_sha256) == 64


def test_runs_existing_rotation_checkpoint_zero_shot(tmp_path) -> None:
    full = load_unity_benchmark(
        "/home/data/xchen/gymnastics/unity_benchmark"
    )
    benchmark = UnityBenchmark(
        full.root, full.joint_names, full.cameras, full.frames[:5]
    )
    sample_ids = [frame.sample_id for frame in benchmark.frames]
    face, side = _pose_pair(frames=5)
    cache_root = tmp_path / "sam3d"
    _write_camera_cache(cache_root, "cam0", sample_ids, face)
    _write_camera_cache(cache_root, "cam1", sample_ids, side)

    outputs = run_rotation_aware_fusion(
        benchmark,
        cache_root,
        tmp_path / "fusion",
        {
            "A4": (
                "local/runs/fuse_rotation_aware/runs/"
                "all137_a4_e100_seed0/checkpoints/best.pt"
            )
        },
        skeleton_path="configs/fusion/skeleton_mhr70.yaml",
        fps=60.0,
        device="cpu",
    )

    assert len(outputs) == 1
    assert outputs[0].method == "A4"
    assert outputs[0].metadata["unity_training"] is False
    assert outputs[0].points.shape == (5, 70, 3)
    assert (tmp_path / "fusion/rotation_aware/A4/static_sweep.npz").is_file()
