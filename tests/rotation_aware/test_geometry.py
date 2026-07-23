from pathlib import Path

import torch

from fuse.rotation_aware.config import SkeletonSpec, load_skeleton_spec
from fuse.rotation_aware.geometry import build_pelvis_frame, build_thorax_frame, canonicalize_pose, restore_pose


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


def test_canonicalization_clears_valid_mask_for_nonfinite_coordinates():
    points, valid = synthetic_mhr70_pose()
    index = SPEC.joint_index("neck")
    points[:, 1, index] = torch.tensor([float("nan"), 0.0, 0.0])

    canonical = canonicalize_pose(points, valid, SPEC)

    assert not canonical.valid[:, 1, index].any()
    assert torch.isfinite(canonical.points).all()


def test_thorax_frame_falls_back_to_shoulders_when_acromions_are_invalid():
    points, valid = synthetic_mhr70_pose()
    for acromion, shoulder in (("left-acromion", "left-shoulder"), ("right-acromion", "right-shoulder")):
        points[:, :, SPEC.joint_index(shoulder)] = points[:, :, SPEC.joint_index(acromion)]
        valid[:, :, SPEC.joint_index(shoulder)] = True
        valid[:, :, SPEC.joint_index(acromion)] = False

    frame = build_thorax_frame(points, valid, SPEC)

    assert frame.valid.all()
    assert torch.isfinite(frame.rotation).all()


def test_thorax_frame_uses_shoulders_when_acromion_roles_are_absent():
    points, valid = synthetic_mhr70_pose()
    for acromion, shoulder in (("left-acromion", "left-shoulder"), ("right-acromion", "right-shoulder")):
        points[:, :, SPEC.joint_index(shoulder)] = points[:, :, SPEC.joint_index(acromion)]
        valid[:, :, SPEC.joint_index(shoulder)] = True
    roles = {name: role for name, role in SPEC.roles.items() if name not in {"left_acromion", "right_acromion"}}
    shoulder_only_spec = SkeletonSpec(SPEC.name, SPEC.joint_names, SPEC.bones, roles, SPEC.required_roles)

    frame = build_thorax_frame(points, valid, shoulder_only_spec)

    assert frame.valid.all()


def test_constructed_frames_are_orthogonal_right_handed_rotations():
    points, valid = synthetic_mhr70_pose(theta_deg=31.0)

    for frame in (build_pelvis_frame(points, valid, SPEC), build_thorax_frame(points, valid, SPEC)):
        identity = torch.eye(3).expand_as(frame.rotation)
        torch.testing.assert_close(frame.rotation.transpose(-1, -2) @ frame.rotation, identity, atol=1e-5, rtol=0)
        torch.testing.assert_close(torch.linalg.det(frame.rotation), torch.ones_like(frame.valid, dtype=points.dtype), atol=1e-5, rtol=0)


def test_thorax_frame_uses_neck_to_thorax_as_vertical_hint():
    points, valid = synthetic_mhr70_pose()
    neck = SPEC.joint_index("neck")
    points[:, :, neck] = torch.tensor([0.5, 3.0, 0.25])

    frame = build_thorax_frame(points, valid, SPEC)

    expected = torch.tensor([0.0, 1.0, 0.25])
    expected = expected / torch.linalg.vector_norm(expected)
    torch.testing.assert_close(frame.rotation[:, :, :, 1], expected.expand_as(frame.rotation[:, :, :, 1]), atol=1e-5, rtol=0)


def test_thorax_frame_is_directly_invalid_when_neck_is_invalid():
    points, valid = synthetic_mhr70_pose()
    valid[:, :, SPEC.joint_index("neck")] = False

    frame = build_thorax_frame(points, valid, SPEC)

    assert not frame.valid.any()
