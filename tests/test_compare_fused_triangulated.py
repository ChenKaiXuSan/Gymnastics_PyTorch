import numpy as np

from analysis.compare_fused_triangulated import (
    build_fused_pair_index,
    compute_joint_errors,
    frame_pairs_from_summary,
    summarize_source_errors,
)


def test_build_fused_pair_index_uses_face_and_side_frame_ids():
    face_map = np.array([6, 7, 8], dtype=np.int32)
    side_map = np.array([0, 1, 2], dtype=np.int32)

    index = build_fused_pair_index(face_map, side_map)

    assert index[(6, 0)] == 0
    assert index[(7, 1)] == 1
    assert index[(8, 2)] == 2


def test_frame_pairs_from_summary_uses_cycle_frame_ranges():
    summary = {
        "face_video_frames": {"start": 360, "end": 516},
        "side_video_frames": {"start": 354, "end": 510},
        "processed_frames": 3,
    }

    pairs = frame_pairs_from_summary(summary)

    assert pairs == [(360, 354), (361, 355), (362, 356)]


def test_compute_joint_errors_filters_invalid_joints():
    fused = np.array(
        [
            [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    triangulated = np.array(
        [
            [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [2.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )

    errors, valid = compute_joint_errors(fused, triangulated)

    assert valid.tolist() == [[True, False], [True, True]]
    np.testing.assert_allclose(errors[valid], np.array([5.0, 0.0, 5.0]))


def test_summarize_source_errors_groups_by_person_and_source():
    values = {
        "face": [np.array([1.0, 3.0], dtype=np.float32)],
        "side": [np.array([2.0], dtype=np.float32), np.array([4.0], dtype=np.float32)],
        "fuse": [],
    }

    rows = summarize_source_errors(
        person_id="27",
        source_errors=values,
        matched_frames={"face": 1, "side": 2, "fuse": 0},
        missing_frames={"face": 0, "side": 1, "fuse": 3},
        scales={"face": [1.0], "side": [2.0, 4.0], "fuse": []},
    )

    assert [(row.person_id, row.source) for row in rows] == [
        ("27", "face"),
        ("27", "side"),
        ("27", "fuse"),
    ]
    assert rows[0].valid_points == 2
    assert rows[0].mpjpe == 2.0
    assert rows[1].matched_frames == 2
    assert rows[1].missing_frames == 1
    assert rows[1].scale == 3.0
    assert np.isnan(rows[2].mpjpe)
