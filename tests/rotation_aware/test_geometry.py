from pathlib import Path

import torch

from fuse.rotation_aware.config import load_skeleton_spec
from fuse.rotation_aware.geometry import canonicalize_pose, restore_pose


SPEC = load_skeleton_spec(Path("configs/fuse/skeleton_mhr70.yaml"))


def synthetic_mhr70_pose(*, theta_deg: float = 0.0, batch: int = 1, frames: int = 3):
    """An upright torso whose shoulders twist around the pelvis-to-thorax axis."""
    points = torch.zeros(batch, frames, len(SPEC.joint_names), 3)
    valid = torch.zeros(batch, frames, len(SPEC.joint_names), dtype=torch.bool)
    theta = torch.deg2rad(torch.tensor(theta_deg))
    shoulder_axis = torch.stack((torch.cos(theta), torch.zeros_like(theta), -torch.sin(theta)))
    roles = {
        "left-hip": torch.tensor([-1.0, 0.0, 0.0]),
        "right-hip": torch.tensor([1.0, 0.0, 0.0]),
        "left-acromion": torch.tensor([0.0, 2.0, 0.0]) - shoulder_axis,
        "right-acromion": torch.tensor([0.0, 2.0, 0.0]) + shoulder_axis,
        "neck": torch.tensor([0.0, 2.2, 0.0]),
    }
    for name, value in roles.items():
        index = SPEC.joint_index(name)
        points[:, :, index] = value
        valid[:, :, index] = True
    return points, valid


def test_canonical_round_trip_preserves_valid_points():
    points, valid = synthetic_mhr70_pose(theta_deg=30.0)

    canonical = canonicalize_pose(points, valid, SPEC)
    restored = restore_pose(canonical.points, canonical.transform)

    torch.testing.assert_close(restored[valid], points[valid], atol=1e-5, rtol=0)
    assert torch.equal(canonical.valid, valid)


def test_canonicalization_is_invariant_to_translation_rotation_and_trial_scale():
    points, valid = synthetic_mhr70_pose(theta_deg=20.0)
    angle = torch.deg2rad(torch.tensor(47.0))
    rotation = torch.tensor(
        [[torch.cos(angle), 0.0, torch.sin(angle)], [0.0, 1.0, 0.0], [-torch.sin(angle), 0.0, torch.cos(angle)]]
    )
    transformed = points @ rotation.T * 3.5 + torch.tensor([5.0, -2.0, 7.0])

    original = canonicalize_pose(points, valid, SPEC)
    changed = canonicalize_pose(transformed, valid, SPEC)

    torch.testing.assert_close(changed.points[valid], original.points[valid], atol=1e-5, rtol=0)


def test_degenerate_frames_are_finite_and_use_previous_valid_transform():
    points, valid = synthetic_mhr70_pose()
    left_hip = SPEC.joint_index("left-hip")
    right_hip = SPEC.joint_index("right-hip")
    points[:, 1, right_hip] = points[:, 1, left_hip]

    canonical = canonicalize_pose(points, valid, SPEC)

    assert not canonical.transform.valid[:, 1].any()
    assert torch.isfinite(canonical.points).all()
    torch.testing.assert_close(canonical.transform.rotation[:, 1], canonical.transform.rotation[:, 0])


def test_canonicalization_has_finite_gradients():
    points, valid = synthetic_mhr70_pose(theta_deg=15.0)
    points.requires_grad_()

    canonical = canonicalize_pose(points, valid, SPEC)
    canonical.points.square().sum().backward()

    assert torch.isfinite(points.grad).all()
