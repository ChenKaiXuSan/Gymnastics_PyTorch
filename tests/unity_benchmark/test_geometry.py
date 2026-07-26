from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.geometry import (
    project_world,
    run_oracle_triangulation,
    triangulate_pixels,
)


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
