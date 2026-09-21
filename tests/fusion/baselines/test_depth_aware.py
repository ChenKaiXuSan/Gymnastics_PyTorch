from __future__ import annotations

import numpy as np
import pytest

from fusion.baselines.methods import (
    ALL_METHODS,
    DEPTH_AWARE_ALPHAS,
    NO_EXTRINSIC_METHODS,
    build_body_frame,
    camera_depth_axis_in_body,
    current_body_average,
    depth_aware_body_average,
    kpts_world_to_body,
)

FRAME_JOINTS = (5, 6, 9, 10)  # shoulders and hips define the body frame


def _skeleton_pose(rng: np.random.Generator, frames: int) -> np.ndarray:
    base = rng.normal(scale=0.3, size=(70, 3))
    base[9] = (-0.1, 0.9, 0.0)
    base[10] = (0.1, 0.9, 0.0)
    base[5] = (-0.2, 1.4, 0.0)
    base[6] = (0.2, 1.4, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, frames)[:, None, None]
    motion = 0.05 * np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=-1)[:, :, 0, :]
    return (base[None] + motion + np.array([0.0, 0.0, 3.0])).astype(np.float32)


def _depth_noise(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    """Per-joint noise along the camera optical axis only, zero on the frame joints."""
    noise = np.zeros(shape, dtype=np.float32)
    noise[..., 2] = rng.normal(scale=scale, size=shape[:-1])
    noise[:, list(FRAME_JOINTS)] = 0.0
    return noise


@pytest.fixture
def orthogonal_views() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face camera along body z, side camera along body x, depth-only noise."""
    rng = np.random.default_rng(7)
    clean = _skeleton_pose(rng, 60)
    face = clean + _depth_noise(rng, clean.shape, 0.05)
    # Rotate the world by 90 degrees about y so the side camera's optical axis is the body x axis.
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    side = clean @ rotation + _depth_noise(rng, clean.shape, 0.05)
    return clean, face, side


def test_registry_contains_depth_aware_methods() -> None:
    assert set(DEPTH_AWARE_ALPHAS) <= set(NO_EXTRINSIC_METHODS) <= set(ALL_METHODS)
    assert DEPTH_AWARE_ALPHAS["avg_body_depthaware_hard"] == (1.0, 1.0)
    assert all(0.0 < a <= 1.0 for alphas in DEPTH_AWARE_ALPHAS.values() for a in alphas)


def test_depth_axis_is_the_optical_axis_in_body_coordinates(orthogonal_views) -> None:
    clean, face, side = orthogonal_views
    _, face_rotation = build_body_frame(face)
    _, side_rotation = build_body_frame(side)
    # R^T e_z: the camera optical axis rotated into the body frame.
    np.testing.assert_allclose(camera_depth_axis_in_body(face_rotation), face_rotation[:, 2, :])
    np.testing.assert_allclose(np.abs(camera_depth_axis_in_body(face_rotation)[:, 2]), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.abs(camera_depth_axis_in_body(side_rotation)[:, 0]), 1.0, atol=1e-5)


def test_alpha_zero_is_the_plain_body_average(orthogonal_views) -> None:
    _, face, side = orthogonal_views
    np.testing.assert_allclose(
        depth_aware_body_average(face, side, 0.0, 0.0), current_body_average(face, side), atol=1e-5
    )


def test_hard_variant_removes_depth_only_noise_exactly(orthogonal_views) -> None:
    clean, face, side = orthogonal_views
    fused = depth_aware_body_average(face, side, 1.0, 1.0)
    target = kpts_world_to_body(clean)
    err_fused = np.linalg.norm(kpts_world_to_body(fused) - target, axis=-1).mean()
    err_plain = np.linalg.norm(kpts_world_to_body(current_body_average(face, side)) - target, axis=-1).mean()
    assert err_plain > 0.01
    assert err_fused < 1e-4
    # Restored in the face view's world frame at the face pelvis.
    np.testing.assert_allclose(fused[:, [9, 10]], face[:, [9, 10]], atol=1e-5)


def test_soft_variant_interpolates_between_average_and_hard(orthogonal_views) -> None:
    clean, face, side = orthogonal_views
    target = kpts_world_to_body(clean)

    def error(alpha: float) -> float:
        fused = depth_aware_body_average(face, side, alpha, alpha)
        return float(np.linalg.norm(kpts_world_to_body(fused) - target, axis=-1).mean())

    assert error(0.0) > error(0.5) > error(1.0)


def test_parallel_cameras_fall_back_to_the_average() -> None:
    rng = np.random.default_rng(3)
    clean = _skeleton_pose(rng, 30)
    face = clean + rng.normal(scale=0.01, size=clean.shape).astype(np.float32)
    side = clean + rng.normal(scale=0.01, size=clean.shape).astype(np.float32)
    fused = depth_aware_body_average(face, side, 1.0, 1.0)
    assert np.isfinite(fused).all()
    # The shared depth direction is unobservable: the cap keeps the result
    # within a few cm of the plain average instead of amplifying frame noise.
    deviation = np.linalg.norm(fused - current_body_average(face, side), axis=-1)
    assert deviation.mean() < 0.02 and deviation.max() < 0.1


def test_nan_frames_and_joints_stay_nan(orthogonal_views) -> None:
    _, face, side = orthogonal_views
    face = face.copy()
    side = side.copy()
    face[3:5] = np.nan
    side[10, 40] = np.nan
    fused = depth_aware_body_average(face, side, 1.0, 1.0)
    assert np.isnan(fused[3:5]).all()
    assert np.isnan(fused[10, 40]).all()
    assert np.isfinite(fused[10, :40]).all()
    assert np.isfinite(fused[:3]).all() and np.isfinite(fused[5:10]).all()
