from __future__ import annotations

import numpy as np
import pytest

from fusion.baselines.classical_baselines import (
    BASELINE_METHODS,
    CLASSICAL_METHODS,
    _kalman_channels,
    fuse_baseline,
    jitter_weighted_body_average,
    kalman_body_fusion,
)
from fusion.baselines.methods import (
    ALL_METHODS,
    current_body_average,
    kpts_world_to_body,
    NO_EXTRINSIC_METHODS,
)


def _skeleton_pose(rng: np.random.Generator, frames: int) -> np.ndarray:
    """A plausible moving MHR70 world sequence with a well-defined body frame."""
    base = rng.normal(scale=0.3, size=(70, 3))
    base[9] = (-0.1, 0.9, 0.0)  # left hip
    base[10] = (0.1, 0.9, 0.0)  # right hip
    base[5] = (-0.2, 1.4, 0.0)  # left shoulder
    base[6] = (0.2, 1.4, 0.0)  # right shoulder
    t = np.linspace(0.0, 2.0 * np.pi, frames)[:, None, None]
    motion = 0.05 * np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=-1)[:, :, 0, :]
    return (base[None] + motion + np.array([0.0, 0.0, 3.0])).astype(np.float32)


@pytest.fixture
def views() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260918)
    clean = _skeleton_pose(rng, 120)
    face = clean + rng.normal(scale=0.01, size=clean.shape).astype(np.float32)
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    side = clean @ rotation + rng.normal(scale=0.03, size=clean.shape).astype(np.float32)
    return face, side


def test_registry_extends_existing_methods() -> None:
    assert ALL_METHODS[: len(NO_EXTRINSIC_METHODS)] == NO_EXTRINSIC_METHODS
    assert ALL_METHODS[len(NO_EXTRINSIC_METHODS) :] == BASELINE_METHODS
    assert set(CLASSICAL_METHODS) < set(BASELINE_METHODS)


@pytest.mark.parametrize("method", BASELINE_METHODS)
def test_every_baseline_returns_world_sequence(method: str, views) -> None:
    face, side = views
    fused, extra = fuse_baseline(method, face, side, fps=60.0)
    assert fused.shape == face.shape and fused.dtype == np.float32
    assert np.isfinite(fused).all()
    assert extra["fusion_frame"] == "pelvis_centred_body_frame"
    # Restored at the face pelvis like avg_body_current; smoothing may move the
    # hip joints themselves slightly, so only require agreement within 5 cm.
    reference = current_body_average(face, side)
    assert np.abs(fused[:, [9, 10]].mean(axis=1) - reference[:, [9, 10]].mean(axis=1)).max() < 0.05


@pytest.mark.parametrize("method", BASELINE_METHODS)
def test_short_sequences_are_supported(method: str, views) -> None:
    face, side = views
    fused, _ = fuse_baseline(method, face[:6], side[:6], fps=30.0)
    assert fused.shape == (6, 70, 3) and np.isfinite(fused).all()


def test_missing_joints_are_masked_not_propagated(views) -> None:
    face, side = views
    face = face.copy()
    face[10:20, 40] = np.nan  # a wrist missing in the face view only
    side = side.copy()
    side[30:35] = np.nan  # whole frames missing in the side view
    for method in CLASSICAL_METHODS:
        fused, _ = fuse_baseline(method, face, side, fps=60.0)
        assert np.isfinite(fused[10:20, 40]).all() or method == "butterworth_body_average"
        assert np.isfinite(fused[30:35, :9]).all() or method == "butterworth_body_average"
        assert np.isfinite(fused[:10]).all()


def test_kalman_reduces_noise_relative_to_plain_average(views) -> None:
    rng = np.random.default_rng(1)
    clean = _skeleton_pose(rng, 240)
    face = clean + rng.normal(scale=0.01, size=clean.shape).astype(np.float32)
    side = clean + rng.normal(scale=0.03, size=clean.shape).astype(np.float32)
    plain = kpts_world_to_body(current_body_average(face, side))
    fused, extra = kalman_body_fusion(face, side, fps=60.0, smoother=True)
    fused_body = kpts_world_to_body(fused)
    target = kpts_world_to_body(clean)
    err_plain = np.linalg.norm(plain - target, axis=-1).mean()
    err_kalman = np.linalg.norm(fused_body - target, axis=-1).mean()
    assert err_kalman < err_plain
    assert extra["mean_measurement_noise_side"] > extra["mean_measurement_noise_face"]


def test_kalman_channels_track_constant_velocity_exactly() -> None:
    frames = 50
    truth = np.linspace(0.0, 1.0, frames)[:, None]
    z = (truth, truth)
    m = (np.ones((frames, 1), dtype=bool), np.ones((frames, 1), dtype=bool))
    r = (np.full(1, 1e-6), np.full(1, 1e-6))
    out = _kalman_channels(z, m, r, fps=50.0, sigma_a2=1e-4, smoother=True)
    np.testing.assert_allclose(out[5:], truth[5:], atol=2e-3)


def test_jitter_weights_favour_the_steadier_view() -> None:
    rng = np.random.default_rng(2)
    clean = _skeleton_pose(rng, 200)
    face = clean + rng.normal(scale=0.002, size=clean.shape).astype(np.float32)
    side = clean + rng.normal(scale=0.05, size=clean.shape).astype(np.float32)
    _, extra = jitter_weighted_body_average(face, side)
    assert extra["mean_face_weight"] > 0.7
