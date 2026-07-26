from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.sam3d import (
    load_sam3d_camera_cache,
    run_sam3d_inference,
)


UNITY_ROOT = Path("/home/data/xchen/gymnastics/unity_benchmark")
CONFIG = Path("configs/sam3d/sam3d_body.yaml")


class FakeEstimator:
    def __init__(self, *, detect: bool = True) -> None:
        self.detect = detect

    def process_one_image(self, *, img, bboxes=None):
        del img, bboxes
        if not self.detect:
            return []
        return [
            {
                "bbox": np.asarray([0, 0, 100, 100], dtype=np.float32),
                "pred_keypoints_3d": np.ones((70, 3), dtype=np.float32),
                "pred_keypoints_2d": np.full((70, 2), 2.0, dtype=np.float32),
            }
        ]


class Factory:
    def __init__(self, *, detect: bool = True) -> None:
        self.calls = 0
        self.detect = detect

    def __call__(self, config_path: Path, device: str):
        assert config_path == CONFIG
        assert device == "cpu"
        self.calls += 1
        return FakeEstimator(detect=self.detect)


def test_inference_loads_estimator_once_and_resumes(tmp_path: Path) -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)
    factory = Factory()

    first = run_sam3d_inference(
        benchmark,
        "cam0",
        tmp_path,
        CONFIG,
        "cpu",
        sample_ids=(0, 1),
        estimator_factory=factory,
    )
    second = run_sam3d_inference(
        benchmark,
        "cam0",
        tmp_path,
        CONFIG,
        "cpu",
        sample_ids=(0, 1),
        estimator_factory=factory,
    )
    cached = load_sam3d_camera_cache(tmp_path, "cam0", (1, 0))

    assert factory.calls == 1
    assert first.completed == 2
    assert first.reused == 0
    assert second.completed == 2
    assert second.reused == 2
    assert cached.sample_ids.tolist() == [1, 0]
    assert cached.points_3d.shape == (2, 70, 3)
    assert cached.valid_2d.all()


def test_detection_failure_is_explicit_and_keeps_frame_position(
    tmp_path: Path,
) -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)

    summary = run_sam3d_inference(
        benchmark,
        "cam1",
        tmp_path,
        CONFIG,
        "cpu",
        sample_ids=(0,),
        estimator_factory=Factory(detect=False),
    )
    cached = load_sam3d_camera_cache(tmp_path, "cam1", (0,))

    assert summary.completed == 0
    assert len(summary.failed) == 1
    assert summary.failed[0]["sample_id"] == 0
    assert cached.sample_ids.tolist() == [0]
    assert not cached.valid_3d.any()
    assert cached.failures == {0: "no_person_detected"}
