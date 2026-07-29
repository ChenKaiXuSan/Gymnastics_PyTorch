from __future__ import annotations

import cv2
import numpy as np

from gymnastics.benchmarks.unity.camera_features import (
    build_camera_feature_sequence,
    fit_relative_camera_from_training_2d,
)
from gymnastics.triangulation.estimate_extrinsics import geodesic_deg


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        ),
        dtype=np.float64,
    )


def _project(
    points: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(rotation)
    pixels, _ = cv2.projectPoints(
        points,
        rotation_vector,
        translation,
        intrinsics,
        np.zeros(5),
    )
    return pixels.reshape(-1, 2)


def _known_rig() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(17)
    frames, joints = 12, 20
    intrinsics = np.stack(
        (
            np.array(
                ((900.0, 0.0, 320.0), (0.0, 910.0, 240.0), (0.0, 0.0, 1.0))
            ),
            np.array(
                ((880.0, 0.0, 315.0), (0.0, 895.0, 245.0), (0.0, 0.0, 1.0))
            ),
        )
    )
    rotation = _rotation_y(28.0)
    translation = np.array((0.8, 0.05, 0.12), dtype=np.float64)
    points = rng.normal(scale=0.35, size=(frames, joints, 3))
    points[..., 2] += 4.5
    pixels = np.stack(
        (
            np.stack(
                [
                    _project(frame, intrinsics[0], np.eye(3), np.zeros(3))
                    for frame in points
                ]
            ),
            np.stack(
                [
                    _project(frame, intrinsics[1], rotation, translation)
                    for frame in points
                ]
            ),
        ),
        axis=1,
    )
    valid = np.ones((frames, 2, joints), dtype=bool)
    sample_ids = np.arange(100, 100 + frames, dtype=np.int64)
    return pixels, valid, intrinsics, sample_ids, rotation


def test_fitted_camera_recovers_training_rig_and_records_only_training_ids() -> None:
    pixels, valid, intrinsics, sample_ids, expected_rotation = _known_rig()

    fitted = fit_relative_camera_from_training_2d(
        pixels,
        valid,
        intrinsics,
        sample_ids=sample_ids,
        threshold_px=1.0,
    )

    assert geodesic_deg(fitted.rotation_face_to_side, expected_rotation) < 1.0
    assert np.isclose(np.linalg.norm(fitted.translation_direction_face_to_side), 1.0)
    assert fitted.inlier_ratio > 0.95
    assert fitted.holdout_reprojection_px < 0.1
    assert fitted.fit_sample_ids.tolist() == sample_ids.tolist()


def test_camera_features_are_finite_masked_and_keep_documented_shapes() -> None:
    pixels, valid, intrinsics, sample_ids, _ = _known_rig()
    valid[3, 1, 7] = False
    pixels[3, 1, 7] = np.nan
    fitted = fit_relative_camera_from_training_2d(
        pixels,
        valid,
        intrinsics,
        sample_ids=sample_ids,
        threshold_px=1.0,
    )

    features = build_camera_feature_sequence(
        fitted,
        pixels,
        valid,
        intrinsics,
        image_sizes=np.array(((640.0, 480.0), (640.0, 480.0))),
    )

    assert features.global_features.shape == (19,)
    assert features.joint_features.shape == (12, 20, 8)
    assert features.valid.shape == (12, 20)
    assert len(features.global_schema) == 19
    assert len(features.joint_schema) == 8
    assert np.isfinite(features.global_features).all()
    assert np.isfinite(features.joint_features).all()
    assert not features.valid[3, 7]
    assert np.array_equal(features.joint_features[3, 7], np.zeros(8))
    assert features.joint_features[features.valid, 4].max() < 1e-3


def test_camera_fit_rejects_non_training_sample_identity_shape() -> None:
    pixels, valid, intrinsics, sample_ids, _ = _known_rig()

    try:
        fit_relative_camera_from_training_2d(
            pixels,
            valid,
            intrinsics,
            sample_ids=sample_ids[:-1],
        )
    except ValueError as error:
        assert "sample_ids" in str(error)
    else:
        raise AssertionError("mismatched sample IDs must be rejected")
