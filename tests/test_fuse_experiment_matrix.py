import numpy as np
import pytest

from gymnastics.fusion.deterministic.experiment_matrix import (
    ALL_METHODS,
    AVAILABLE_METHODS,
    EXTRINSIC_METHODS,
    align_side_with_extrinsic_rotation,
    apply_sim3,
    build_aligned_timeline,
    bodypart_weights,
    estimate_joint_weights,
    estimate_sim3,
    fit_similarity,
    fuse_extrinsic_rotation,
    fuse_quality_weighted,
    fuse_weighted,
    iter_person_ids,
    joint_errors,
    load_aligned_cycle_cache,
    load_extrinsic_rotation,
    load_split_alignment_offset,
    root_align_to_reference,
    sam3d_person_root,
    smooth_sequence,
)


def _rotation_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_extrinsic_methods_are_available_but_not_in_the_legacy_default():
    assert set(EXTRINSIC_METHODS).issubset(AVAILABLE_METHODS)
    assert set(EXTRINSIC_METHODS).isdisjoint(ALL_METHODS)


def test_load_extrinsic_rotation_reads_valid_person_entry(tmp_path):
    path = tmp_path / "estimated_extrinsics.json"
    rotation = _rotation_z(0.7)
    path.write_text(
        (
            '{"persons": {"47": {"R": '
            + str(rotation.tolist()).replace("'", '"')
            + ', "t": [1, 2, 3], "method": "per_person"}}}'
        ),
        encoding="utf-8",
    )

    loaded, metadata = load_extrinsic_rotation(path, "47")

    np.testing.assert_allclose(loaded, rotation)
    assert metadata["method"] == "per_person"
    assert metadata["person_id"] == "47"


def test_load_extrinsic_rotation_rejects_missing_person(tmp_path):
    path = tmp_path / "estimated_extrinsics.json"
    path.write_text('{"persons": {}}', encoding="utf-8")

    with pytest.raises(KeyError, match="person 47"):
        load_extrinsic_rotation(path, "47")


def test_load_extrinsic_rotation_rejects_non_rotation(tmp_path):
    path = tmp_path / "estimated_extrinsics.json"
    path.write_text(
        '{"persons": {"47": {"R": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid rotation"):
        load_extrinsic_rotation(path, "47")


def test_load_aligned_cycle_cache_concatenates_manifest_trials(tmp_path):
    person_root = tmp_path / "person_47"
    generation = person_root / ".generations" / "generation_test"
    generation.mkdir(parents=True)
    manifest = {
        "generation": "generation_test",
        "person_id": "47",
        "source_hash": "source-hash",
        "config_hash": "config-hash",
        "source": {"offset_side_to_face": -4},
        "trials": ["cycle_000", "cycle_001"],
    }
    (person_root / "manifest.json").write_text(
        '{"generation": "generation_test"}', encoding="utf-8"
    )
    (generation / "manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )
    for index, start in enumerate((10, 20)):
        points = np.full((2, 70, 3), float(index), dtype=np.float32)
        np.savez_compressed(
            generation / f"cycle_{index:03d}.npz",
            face=points,
            side=points + 1,
            face_map=np.arange(start, start + 2, dtype=np.int32),
            side_map=np.arange(start + 4, start + 6, dtype=np.int32),
        )

    face, side, face_map, side_map, metadata = load_aligned_cycle_cache(
        tmp_path, "47"
    )

    assert face.shape == side.shape == (4, 70, 3)
    assert face_map.tolist() == [10, 11, 20, 21]
    assert side_map.tolist() == [14, 15, 24, 25]
    assert metadata["offset_side_to_face"] == -4
    assert metadata["generation"] == "generation_test"
    assert metadata["trial_lengths"] == [2, 2]
    assert metadata["sequence_scope"] == "split_cycles_concatenated"


def test_align_side_with_extrinsic_rotation_uses_side_to_face_convention():
    rng = np.random.default_rng(47)
    face = rng.normal(size=(3, 70, 3)).astype(np.float32)
    face_root = face[:, [9, 10], :].mean(axis=1, keepdims=True)
    rotation = _rotation_z(0.8).astype(np.float32)
    side_root = np.array([[[4.0, -2.0, 1.0]]], dtype=np.float32)
    side = (face - face_root) @ rotation.T + side_root

    aligned = align_side_with_extrinsic_rotation(side, face, rotation)

    np.testing.assert_allclose(aligned, face, atol=1e-5)


def test_fuse_extrinsic_rotation_averages_in_face_axes():
    face = np.zeros((1, 70, 3), dtype=np.float32)
    face[:, [9, 10], 0] = 2.0
    rotation = _rotation_z(np.pi / 2).astype(np.float32)
    side = np.zeros_like(face)
    side[:, :, 1] = -2.0
    side[:, [9, 10], :] = 0.0

    fused = fuse_extrinsic_rotation(face, side, rotation)

    expected_side = align_side_with_extrinsic_rotation(side, face, rotation)
    np.testing.assert_allclose(fused, 0.5 * (face + expected_side), atol=1e-6)


def test_fuse_quality_weighted_prefers_higher_quality_view():
    face = np.zeros((2, 3, 3), dtype=np.float32)
    side = np.full_like(face, 10.0)
    face_quality = np.array([3.0, 0.0], dtype=np.float32)
    side_quality = np.array([1.0, 0.0], dtype=np.float32)

    fused, weights = fuse_quality_weighted(
        face, side, face_quality, side_quality
    )

    np.testing.assert_allclose(fused[0], np.full((3, 3), 2.5))
    np.testing.assert_allclose(fused[1], np.full((3, 3), 5.0))
    np.testing.assert_allclose(weights, np.array([[0.75, 0.25], [0.5, 0.5]]))


def test_fit_similarity_recovers_a_known_scale_rotation_and_translation():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(40, 3))
    rotation = _rotation_z(0.7)
    target = 2.5 * (source @ rotation) + np.array([1.0, -2.0, 3.0])

    transform = fit_similarity(source, target)

    assert transform.scale == pytest.approx(2.5, rel=1e-9)
    assert apply_sim3(source, transform) == pytest.approx(target, abs=1e-6)


def test_similarity_alignment_removes_a_pure_frame_and_scale_mismatch():
    """A candidate differing only by world frame and scale must score ~0."""
    rng = np.random.default_rng(1)
    triangulated = rng.normal(size=(8, 12, 3))
    rotation = _rotation_z(0.9)
    candidate = (triangulated @ rotation.T) / 1.4 + np.array([0.3, -0.2, 0.5])

    root_errors, _ = joint_errors(candidate, triangulated, alignment="root")
    similarity_errors, valid = joint_errors(
        candidate, triangulated, alignment="similarity"
    )

    assert valid.all()
    assert root_errors.mean() > 0.5
    assert similarity_errors.mean() < 1e-6


def test_similarity_alignment_still_reports_genuine_pose_error():
    """Alignment must not absorb a real per-joint deformation."""
    rng = np.random.default_rng(2)
    triangulated = rng.normal(size=(8, 12, 3))
    candidate = triangulated.copy()
    candidate[:, 3] += np.array([0.4, 0.0, 0.0])

    errors, _ = joint_errors(candidate, triangulated, alignment="similarity")

    assert errors.mean() > 0.01
    assert errors[:, 3].mean() > errors.mean()


def test_root_alignment_remains_available_for_legacy_comparisons():
    triangulated = np.zeros((4, 12, 3))
    candidate = np.zeros((4, 12, 3))
    candidate[:, 2] += np.array([0.0, 0.0, 1.0])

    errors, valid = joint_errors(candidate, triangulated, alignment="root")

    assert valid.all()
    # The pelvis joints stay at the origin, so root centering is a no-op here.
    assert errors[:, 2] == pytest.approx(np.ones(4), abs=1e-9)
    assert errors[:, 0] == pytest.approx(np.zeros(4), abs=1e-9)


def test_joint_errors_rejects_an_unknown_alignment_mode():
    points = np.zeros((2, 4, 3))
    with pytest.raises(ValueError, match="alignment must be one of"):
        joint_errors(points, points, alignment="affine")


def test_estimate_sim3_recovers_scale_rotation_translation():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = 2.0 * (source @ rotation) + np.array([3.0, -1.0, 5.0], dtype=np.float32)

    transform = estimate_sim3(source, target, np.arange(len(source)))
    aligned = apply_sim3(source, transform)

    np.testing.assert_allclose(aligned, target, atol=1e-5)


def test_root_align_to_reference_translates_side_pelvis_to_face():
    face = np.zeros((1, 70, 3), dtype=np.float32)
    side = np.ones((1, 70, 3), dtype=np.float32) * 3.0
    face[:, [9, 10], :] = np.array([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=np.float32)
    side[:, [9, 10], :] = np.array([[[8.0, 1.0, 1.0], [10.0, 1.0, 1.0]]], dtype=np.float32)

    aligned = root_align_to_reference(side, face)

    np.testing.assert_allclose(aligned[:, [9, 10], :].mean(axis=1), face[:, [9, 10], :].mean(axis=1))


def test_estimate_joint_weights_prefers_lower_error_source():
    face_err = np.array([1.0, 4.0, 2.0], dtype=np.float32)
    side_err = np.array([3.0, 1.0, 2.0], dtype=np.float32)

    weights = estimate_joint_weights(face_err, side_err)

    assert weights.shape == (3, 2)
    assert weights[0, 0] > weights[0, 1]
    assert weights[1, 1] > weights[1, 0]
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(3))


def test_fuse_weighted_uses_joint_weights():
    face = np.zeros((1, 2, 3), dtype=np.float32)
    side = np.ones((1, 2, 3), dtype=np.float32) * 10.0
    weights = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32)

    fused = fuse_weighted(face, side, weights)

    np.testing.assert_allclose(fused[0, 0], np.array([2.5, 2.5, 2.5]))
    np.testing.assert_allclose(fused[0, 1], np.array([7.5, 7.5, 7.5]))


def test_build_aligned_timeline_converts_positions_to_frame_ids(monkeypatch):
    def fake_theta(kpts, idx):
        return np.arange(len(kpts), dtype=np.float32)

    def fake_offset(face_theta, side_theta):
        return -1

    monkeypatch.setattr("gymnastics.fusion.deterministic.experiment_matrix.compute_theta_unwrap_from_world", fake_theta)
    monkeypatch.setattr("gymnastics.fusion.deterministic.experiment_matrix.estimate_offset_by_dtw", fake_offset)

    face_by_frame = {
        10: np.full((2, 3), 10.0, dtype=np.float32),
        11: np.full((2, 3), 11.0, dtype=np.float32),
        12: np.full((2, 3), 12.0, dtype=np.float32),
    }
    side_by_frame = {
        20: np.full((2, 3), 20.0, dtype=np.float32),
        21: np.full((2, 3), 21.0, dtype=np.float32),
        22: np.full((2, 3), 22.0, dtype=np.float32),
    }

    face, side, face_map, side_map, offset = build_aligned_timeline(face_by_frame, side_by_frame)

    assert offset == -1
    assert face_map.tolist() == [11, 12]
    assert side_map.tolist() == [20, 21]
    np.testing.assert_allclose(face[:, 0, 0], np.array([11.0, 12.0]))
    np.testing.assert_allclose(side[:, 0, 0], np.array([20.0, 21.0]))


def test_build_aligned_timeline_uses_split_offset_override(monkeypatch):
    def fail_if_called(face_theta, side_theta):
        raise AssertionError("DTW should not run when split offset is provided")

    monkeypatch.setattr("gymnastics.fusion.deterministic.experiment_matrix.estimate_offset_by_dtw", fail_if_called)

    face_by_frame = {
        10: np.full((2, 3), 10.0, dtype=np.float32),
        11: np.full((2, 3), 11.0, dtype=np.float32),
        12: np.full((2, 3), 12.0, dtype=np.float32),
    }
    side_by_frame = {
        20: np.full((2, 3), 20.0, dtype=np.float32),
        21: np.full((2, 3), 21.0, dtype=np.float32),
        22: np.full((2, 3), 22.0, dtype=np.float32),
    }

    face, side, face_map, side_map, offset = build_aligned_timeline(
        face_by_frame, side_by_frame, offset_override=-1
    )

    assert offset == -1
    assert face_map.tolist() == [11, 12]
    assert side_map.tolist() == [20, 21]
    np.testing.assert_allclose(face[:, 0, 0], np.array([11.0, 12.0]))
    np.testing.assert_allclose(side[:, 0, 0], np.array([20.0, 21.0]))


def test_load_split_alignment_offset_reads_alignment_record(tmp_path):
    split_root = tmp_path / "split_cycle"
    person_root = split_root / "person_47"
    person_root.mkdir(parents=True)
    (person_root / "alignment_record_47.json").write_text(
        '{"metadata": {"offset_side_to_face": -11, "offset_source": "kpt_audio_avg"}}',
        encoding="utf-8",
    )

    offset, metadata = load_split_alignment_offset(split_root, "47")

    assert offset == -11
    assert metadata["offset_source"] == "kpt_audio_avg"


def test_load_split_alignment_offset_requires_alignment_record(tmp_path):
    split_root = tmp_path / "split_cycle"

    with pytest.raises(FileNotFoundError):
        load_split_alignment_offset(split_root, "47")


def test_load_split_alignment_offset_requires_offset_value(tmp_path):
    split_root = tmp_path / "split_cycle"
    person_root = split_root / "person_47"
    person_root.mkdir(parents=True)
    (person_root / "alignment_record_47.json").write_text(
        '{"metadata": {"offset_source": "kpt_audio_avg"}}',
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_split_alignment_offset(split_root, "47")


def test_smooth_sequence_reduces_center_spike():
    seq = np.zeros((5, 1, 3), dtype=np.float32)
    seq[2, 0, 0] = 10.0

    smoothed = smooth_sequence(seq, win=3)

    assert smoothed.shape == seq.shape
    assert smoothed[2, 0, 0] < 10.0
    assert smoothed[2, 0, 0] > 0.0


def test_bodypart_weights_are_valid_joint_weights():
    weights = bodypart_weights(70)

    assert weights.shape == (70, 2)
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(70))
    assert weights[41, 0] > weights[41, 1]
    assert weights[9, 0] == weights[9, 1]


def test_iter_person_ids_discovers_from_sam3d_person_root(tmp_path):
    sam3d_root = tmp_path / "sam3d_body_results"
    person_root = sam3d_root / "person"
    for name in ["10", "2", "notes", "46"]:
        (person_root / name).mkdir(parents=True)

    assert sam3d_person_root(sam3d_root) == person_root
    assert list(iter_person_ids(sam3d_root, None)) == ["2", "10", "46"]
    assert list(iter_person_ids(sam3d_root, ["46", "99"])) == ["46"]
