from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnastics.benchmarks.freeman.dataset import load_subject_sessions
from gymnastics.benchmarks.freeman.sam3d import (
    infer_subject_sessions,
    load_inference,
    validate_inference,
)
from gymnastics.benchmarks.freeman import sam3d
from gymnastics.benchmarks.freeman.schema import SelectedPair


class FakeEstimator:
    def __init__(self, *, fail_calls: set[int] | None = None):
        self.calls = 0
        self.fail_calls = fail_calls or set()

    def process_one_image(self, *, img, bboxes):
        assert img.shape == (48, 64, 3)
        assert bboxes is None
        call = self.calls
        self.calls += 1
        if call in self.fail_calls:
            return []
        points3d = np.full((70, 3), call + 1, dtype=np.float32)
        points2d = np.full((70, 2), call + 2, dtype=np.float32)
        return [
            {
                "bbox": np.array([0.0, 0.0, 10.0, 10.0]),
                "pred_keypoints_3d": points3d,
                "pred_keypoints_2d": points2d,
                "pred_cam_t": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
        ]


def _pair(session) -> SelectedPair:
    return SelectedPair(
        session_id=session.session_id,
        view_a="c01",
        view_b="c03",
        reference_view="c01",
        separation_deg=90.0,
        target_error_deg=0.0,
        height_difference=0.0,
    )


def _config(tmp_path: Path, *, frame_stride: int = 1) -> dict:
    sam3d_config = tmp_path / "sam3d.yaml"
    sam3d_config.write_text("model: fixture\n", encoding="utf-8")
    return {
        "paths": {"output_root": tmp_path / "runs"},
        "dataset": {"frame_stride": frame_stride},
        "sam3d": {
            "config": sam3d_config,
            "device": 0,
            "checkpoint_id": "fixture-checkpoint",
        },
    }


def test_streams_selected_views_with_one_estimator_instance(
    freeman_fixture, tmp_path: Path
) -> None:
    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]
    estimator = FakeEstimator()
    factory_calls = 0

    def factory(config):
        nonlocal factory_calls
        factory_calls += 1
        return estimator

    artifacts = infer_subject_sessions(
        [session],
        {session.session_id: _pair(session)},
        _config(tmp_path),
        estimator_factory=factory,
    )

    assert factory_calls == 1
    assert estimator.calls == 6
    assert {item.view_id for item in artifacts} == {"c01", "c03"}
    assert all(item.frames == 3 and item.valid_frames == 3 for item in artifacts)
    predictions = [load_inference(item.path) for item in artifacts]
    assert all(item.points3d.shape == (3, 70, 3) for item in predictions)
    assert all(item.points2d.shape == (3, 70, 2) for item in predictions)


def test_failed_detection_keeps_frame_identity_and_false_masks(
    freeman_fixture, tmp_path: Path
) -> None:
    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]
    estimator = FakeEstimator(fail_calls={1})

    artifacts = infer_subject_sessions(
        [session],
        {session.session_id: _pair(session)},
        _config(tmp_path),
        estimator_factory=lambda _: estimator,
    )
    view_a = load_inference(next(item.path for item in artifacts if item.view_id == "c01"))

    np.testing.assert_array_equal(view_a.frame_ids, np.array([0, 1, 2]))
    assert not view_a.valid3d[1].any()
    assert not view_a.valid2d[1].any()
    np.testing.assert_array_equal(view_a.points3d[1], np.zeros((70, 3)))


def test_truncates_selected_pair_to_common_decodable_trailing_frame(
    freeman_fixture, tmp_path: Path, monkeypatch
) -> None:
    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]
    original_capture = sam3d.cv2.VideoCapture

    class TruncatedCapture:
        def __init__(self, path):
            self._capture = original_capture(path)
            self._truncate = str(path).endswith("c01.mp4")
            self._reads = 0

        def isOpened(self):
            return self._capture.isOpened()

        def read(self):
            if self._truncate and self._reads >= 2:
                return False, None
            success, frame = self._capture.read()
            if success:
                self._reads += 1
            return success, frame

        def release(self):
            self._capture.release()

    monkeypatch.setattr(sam3d.cv2, "VideoCapture", TruncatedCapture)

    artifacts = infer_subject_sessions(
        [session],
        {session.session_id: _pair(session)},
        _config(tmp_path),
        estimator_factory=lambda _: FakeEstimator(),
    )

    assert all(item.frames == 2 for item in artifacts)
    for artifact in artifacts:
        np.testing.assert_array_equal(
            load_inference(artifact.path).frame_ids,
            np.array([0, 1]),
        )


def test_valid_identical_cache_resumes_without_estimator_calls(
    freeman_fixture, tmp_path: Path
) -> None:
    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]
    config = _config(tmp_path)
    first = FakeEstimator()
    initial = infer_subject_sessions(
        [session],
        {session.session_id: _pair(session)},
        config,
        estimator_factory=lambda _: first,
    )

    def forbidden_factory(_):
        raise AssertionError("valid cache must not load SAM3D")

    resumed = infer_subject_sessions(
        [session],
        {session.session_id: _pair(session)},
        config,
        estimator_factory=forbidden_factory,
    )

    assert [item.path for item in resumed] == [item.path for item in initial]
    assert all(validate_inference(item.path) for item in resumed)


def test_frame_stride_identity_change_forces_scoped_recomputation(
    freeman_fixture, tmp_path: Path
) -> None:
    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]
    pair = _pair(session)
    infer_subject_sessions(
        [session],
        {session.session_id: pair},
        _config(tmp_path, frame_stride=1),
        estimator_factory=lambda _: FakeEstimator(),
    )
    replacement = FakeEstimator()

    artifacts = infer_subject_sessions(
        [session],
        {session.session_id: pair},
        _config(tmp_path, frame_stride=2),
        estimator_factory=lambda _: replacement,
    )

    assert replacement.calls == 4
    assert all(item.frames == 2 for item in artifacts)
    for artifact in artifacts:
        np.testing.assert_array_equal(
            load_inference(artifact.path).frame_ids,
            np.array([0, 2]),
        )


def test_corrupt_npz_is_replaced_but_partial_directory_is_not_reused(
    freeman_fixture, tmp_path: Path
) -> None:
    session = load_subject_sessions(
        freeman_fixture.subject_root,
        freeman_fixture.shared_root,
        fps_values=(30,),
    )[0]
    pair = _pair(session)
    config = _config(tmp_path)
    artifacts = infer_subject_sessions(
        [session],
        {session.session_id: pair},
        config,
        estimator_factory=lambda _: FakeEstimator(),
    )
    corrupt = next(item.path for item in artifacts if item.view_id == "c01")
    corrupt.write_bytes(b"not an npz")
    partial = corrupt.parent.with_name(corrupt.parent.name + ".partial")
    partial.mkdir()
    (partial / "prediction.npz").write_bytes(b"incomplete")
    replacement = FakeEstimator()

    repaired = infer_subject_sessions(
        [session],
        {session.session_id: pair},
        config,
        estimator_factory=lambda _: replacement,
    )

    assert replacement.calls == 3
    repaired_c01 = next(item.path for item in repaired if item.view_id == "c01")
    assert validate_inference(repaired_c01)
    assert not partial.exists()
