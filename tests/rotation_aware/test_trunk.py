import torch

from fuse.rotation_aware.trunk import (
    axial_rotation_angle,
    axial_rotation_angle_from_points,
    extract_trunk_features,
    relative_rotation,
)
from tests.rotation_aware.test_geometry import SPEC, synthetic_mhr70_pose


def test_known_thorax_rotation_is_thirty_degrees():
    pose, valid = synthetic_mhr70_pose(theta_deg=30.0)

    theta, theta_valid = axial_rotation_angle_from_points(pose, valid, SPEC)

    assert theta_valid.all()
    torch.testing.assert_close(theta, torch.deg2rad(torch.tensor(30.0)).expand_as(theta), atol=8.7e-3, rtol=0)


def test_relative_rotation_and_axial_angle_follow_so3_convention():
    angle = torch.deg2rad(torch.tensor(30.0))
    pelvis = torch.eye(3).expand(1, 3, 3).clone()
    thorax = torch.tensor(
        [[torch.cos(angle), 0.0, torch.sin(angle)], [0.0, 1.0, 0.0], [-torch.sin(angle), 0.0, torch.cos(angle)]]
    ).expand(1, 3, 3)

    relative = relative_rotation(pelvis, thorax)

    torch.testing.assert_close(axial_rotation_angle(relative), angle.expand(1), atol=1e-6, rtol=0)


def test_trunk_features_wrap_angle_velocity_at_circular_boundary():
    pose, valid = synthetic_mhr70_pose(theta_deg=179.0, frames=2)
    second, _ = synthetic_mhr70_pose(theta_deg=-179.0, frames=1)
    pose[:, 1] = second[:, 0]
    valid[:, 1] = True

    features = extract_trunk_features(pose, valid, SPEC, dt=1.0)

    torch.testing.assert_close(features.omega[:, 1], torch.deg2rad(torch.tensor(2.0)).expand(1), atol=1e-4, rtol=0)
    assert features.omega_valid[:, 1].all()


def test_trunk_acceleration_does_not_wrap_angular_velocity():
    pose, valid = synthetic_mhr70_pose(theta_deg=0.0, frames=3)
    middle, _ = synthetic_mhr70_pose(theta_deg=172.0, frames=1)
    pose[:, 1] = middle[:, 0]

    features = extract_trunk_features(pose, valid, SPEC, dt=1.0)

    torch.testing.assert_close(features.alpha[:, 2], torch.tensor([-2.0 * torch.deg2rad(torch.tensor(172.0))]), atol=1e-4, rtol=0)
    assert features.alpha_valid[:, 2].all()


def test_trunk_features_never_emit_nan_for_masked_roles():
    pose, valid = synthetic_mhr70_pose()
    valid[:] = False

    features = extract_trunk_features(pose, valid, SPEC, dt=1.0)

    assert not features.angle_valid.any()
    assert not features.omega_valid.any()
    assert not features.alpha_valid.any()
    assert torch.isfinite(features.angle).all()
    assert torch.isfinite(features.omega).all()
    assert torch.isfinite(features.alpha).all()


def test_trunk_omega_gradient_is_finite_when_dt_has_a_zero_padded_frame():
    # Regression for the A9 (改法3) divergence: omega = d(angle)/dt divides by the
    # per-frame interval, which is 0 at padded frames. The value was already zeroed
    # for those frames, but the division's backward is grad * (1/dt) = 0 * inf = nan
    # -- which poisoned every upstream gradient and diverged training to all-nan in
    # ~1 epoch. Differentiating omega through a dt=0 frame must stay finite.
    pose, valid = synthetic_mhr70_pose(theta_deg=15.0, frames=4)
    pose = pose.clone().requires_grad_(True)
    dt = torch.tensor([[1.0, 1.0, 0.0, 1.0]])  # frame 2 is padded: dt = 0

    features = extract_trunk_features(pose, valid, SPEC, dt=dt)
    # differentiate through omega (the derivative that divides by dt)
    features.omega.sum().backward()

    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()
