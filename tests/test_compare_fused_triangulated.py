import numpy as np

from analysis.compare_fused_triangulated import (
    align_sequences,
    build_fused_pair_index,
    compute_joint_errors,
    finite_error_values,
    frame_pairs_from_summary,
    summarize_source_errors,
)


def _random_sequence(rng, frames=12, joints=17):
    return rng.normal(size=(frames, joints, 3)).astype(np.float32)


def test_similarity_alignment_removes_a_static_frame_and_scale_offset():
    rng = np.random.default_rng(0)
    triangulated = _random_sequence(rng)
    # A single rotation + scale + translation applied to the whole sequence is
    # exactly what one per-sequence Sim3 must undo, so the residual should vanish.
    theta = 0.7
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    candidate = (triangulated.reshape(-1, 3) @ rot.T * 1.8 + np.array([2.0, -1.0, 0.5])).reshape(
        triangulated.shape
    )
    values, scale = finite_error_values(candidate, triangulated, "similarity")
    assert np.max(values) < 1e-3
    assert np.isclose(scale, 1.0 / 1.8, rtol=1e-3)


def test_similarity_alignment_keeps_per_frame_pose_error():
    rng = np.random.default_rng(1)
    triangulated = _random_sequence(rng)
    candidate = triangulated.copy()
    candidate[5] += rng.normal(scale=0.3, size=candidate[5].shape)  # one corrupted frame
    root_values, _ = finite_error_values(candidate, triangulated, "similarity")
    # A per-frame Procrustes fit would absorb that corruption; the sequence-level
    # similarity fit must leave it in the residual.
    assert np.max(root_values) > 0.05


def test_similarity_alignment_preserves_nan_mask():
    rng = np.random.default_rng(2)
    triangulated = _random_sequence(rng)
    candidate = triangulated.copy()
    candidate[3, 4, :] = np.nan
    aligned, reference, _ = align_sequences(candidate, triangulated, "similarity")
    assert np.isnan(aligned[3, 4]).all()
    assert np.isfinite(aligned[0, 0]).all()


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
