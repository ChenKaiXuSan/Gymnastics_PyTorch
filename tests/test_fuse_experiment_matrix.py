import numpy as np

from fuse.experiment_matrix import (
    apply_sim3,
    estimate_joint_weights,
    estimate_sim3,
    fuse_weighted,
    root_align_to_reference,
)


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
