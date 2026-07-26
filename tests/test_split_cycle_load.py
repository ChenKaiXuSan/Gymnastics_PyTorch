import numpy as np

from gymnastics.alignment.load import load_sam3d_body_sequence


def _write_sam3d_frame(base, frame_idx, keypoint_value):
    base.mkdir(parents=True, exist_ok=True)
    output = {
        "frame_idx": np.int64(frame_idx),
        "pred_keypoints_3d": np.full(
            (70, 3), keypoint_value, dtype=np.float32
        ),
        "frame": np.full((8, 6, 3), frame_idx, dtype=np.uint8),
        "pred_vertices": np.full((4, 3), frame_idx, dtype=np.float32),
    }
    scalar = np.empty((), dtype=object)
    scalar[()] = output
    np.savez(base / f"{frame_idx:06d}_sam3d_body.npz", output=scalar)


def test_loader_returns_distinct_lightweight_metadata_and_stacked_keypoints(tmp_path):
    root = tmp_path / "sam3d_body_results"
    base = root / "person" / "69" / "face"
    _write_sam3d_frame(base, frame_idx=0, keypoint_value=1.0)
    _write_sam3d_frame(base, frame_idx=1, keypoint_value=2.0)

    all_info, keypoints = load_sam3d_body_sequence(
        root, person_id="69", subdir="face"
    )

    assert keypoints.shape == (2, 70, 3)
    np.testing.assert_array_equal(keypoints[0], np.ones((70, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        keypoints[1], np.full((70, 3), 2.0, dtype=np.float32)
    )
    assert all_info[0] is not all_info[1]
    assert [info["frame_idx"] for info in all_info] == [0, 1]
    assert all(set(info) == {"frame_idx", "pred_keypoints_3d"} for info in all_info)
    assert all("frame" not in info for info in all_info)
    assert all("pred_vertices" not in info for info in all_info)

    all_info[0]["frame_idx"] = 99
    assert all_info[1]["frame_idx"] == 1
