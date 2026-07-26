from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.sam3d import (
    _read_rgb_image,
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


class FallbackEstimator:
    def __init__(self) -> None:
        self.boxes: list[np.ndarray | None] = []

    def process_one_image(self, *, img, bboxes=None):
        del img
        self.boxes.append(bboxes)
        if bboxes is None:
            return []
        return [
            {
                "bbox": np.asarray(bboxes[0], dtype=np.float32),
                "pred_keypoints_3d": np.ones((70, 3), dtype=np.float32),
                "pred_keypoints_2d": np.full((70, 2), 2.0, dtype=np.float32),
            }
        ]


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


def test_detection_failure_retries_with_gt_independent_fixed_bbox(
    tmp_path: Path,
) -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)
    estimator = FallbackEstimator()

    summary = run_sam3d_inference(
        benchmark,
        "cam0",
        tmp_path,
        CONFIG,
        "cpu",
        sample_ids=(0,),
        estimator_factory=lambda *_: estimator,
        fallback_bbox_xyxy=(620.0, 280.0, 1300.0, 930.0),
    )
    cache_path = tmp_path / "cam0/00000000.npz"
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = str(payload["metadata"].item())

    assert summary.completed == 1
    assert not summary.failed
    assert estimator.boxes[0] is None
    np.testing.assert_allclose(
        estimator.boxes[1],
        np.asarray([[620.0, 280.0, 1300.0, 930.0]]),
    )
    assert '"proposal_source": "fixed_render_roi_fallback"' in metadata


def test_image_loader_converts_opencv_bgr_to_rgb(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "pixel.png"
    cv2.imwrite(
        str(path),
        np.asarray([[[1, 2, 3]]], dtype=np.uint8),
    )

    image = _read_rgb_image(path)

    assert image.tolist() == [[[3, 2, 1]]]
