from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.geometry import (
    project_world,
    run_oracle_triangulation,
    run_sam3d_triangulation,
    triangulate_pixels,
)
from gymnastics.benchmarks.unity.mapping import map_mhr70_to_unity
from gymnastics.benchmarks.unity.schema import UnityBenchmark


UNITY_ROOT = Path("/home/data/xchen/gymnastics/unity_benchmark")


def test_projection_reproduces_manifest_pixels() -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)
    frame = benchmark.frames[0]

    for camera_id, camera in benchmark.cameras.items():
        pixels, depth = project_world(frame.gt_world_m, camera)
        mask = frame.gt_available
        np.testing.assert_allclose(
            pixels[mask], frame.gt_pixels[camera_id][mask], atol=1e-3
        )
        assert np.all(depth[mask] > 0)


def test_oracle_dlt_recovers_world_points() -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)
    frame = benchmark.frames[0]

    reconstructed = triangulate_pixels(
        frame.gt_pixels["cam0"],
        frame.gt_pixels["cam1"],
        benchmark.cameras["cam0"],
        benchmark.cameras["cam1"],
    )

    np.testing.assert_allclose(
        reconstructed[frame.gt_available],
        frame.gt_world_m[frame.gt_available],
        atol=1e-4,
    )


def test_oracle_triangulation_preserves_three_sequences(tmp_path: Path) -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)

    sequences = run_oracle_triangulation(benchmark, tmp_path)

    assert [sequence.sequence_id for sequence in sequences] == [
        "static_sweep",
        "continuous_left_060_r00",
        "continuous_right_060_r00",
    ]
    assert all(sequence.method == "triangulation_oracle2d" for sequence in sequences)
    assert all(sequence.points.shape[1:] == (16, 3) for sequence in sequences)
    assert (tmp_path / "oracle2d/static_sweep.npz").is_file()


def test_sam3d_pixel_triangulation_recovers_mhr_pose(tmp_path: Path) -> None:
    full = load_unity_benchmark(UNITY_ROOT)
    benchmark = UnityBenchmark(
        full.root, full.joint_names, full.cameras, full.frames[:5]
    )
    rng = np.random.default_rng(9)
    world = rng.normal(scale=0.2, size=(5, 70, 3)).astype(np.float32)
    world[..., 1] += 1.0
    cache_root = tmp_path / "sam3d"
    for camera_id, camera in benchmark.cameras.items():
        camera_root = cache_root / camera_id
        camera_root.mkdir(parents=True)
        pixels, _ = project_world(world, camera)
        for row, frame in enumerate(benchmark.frames):
            np.savez_compressed(
                camera_root / f"{frame.sample_id:08d}.npz",
                pred_keypoints_3d=world[row],
                pred_keypoints_2d=pixels[row],
                valid_3d=np.ones((70,), dtype=bool),
                valid_2d=np.ones((70,), dtype=bool),
                sample_id=np.asarray(frame.sample_id),
            )

    outputs = run_sam3d_triangulation(
        benchmark, cache_root, tmp_path / "triangulation"
    )

    assert len(outputs) == 1
    expected = map_mhr70_to_unity(world)
    np.testing.assert_allclose(outputs[0].points, expected.points, atol=1e-4)
    assert (
        tmp_path / "triangulation/sam3d2d/static_sweep.npz"
    ).is_file()
