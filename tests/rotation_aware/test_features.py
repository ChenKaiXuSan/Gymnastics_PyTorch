from pathlib import Path

import torch

from gymnastics.fusion.rotation_aware.config import SkeletonSpec, load_skeleton_spec
from gymnastics.fusion.rotation_aware.features import (
    QualityConfig,
    compute_disagreement_features,
    compute_quality_features,
    extract_pose_features,
)
from gymnastics.fusion.rotation_aware.trunk import extract_trunk_features
from tests.rotation_aware.test_geometry import synthetic_mhr70_pose


SPEC = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))


def test_identical_views_have_zero_disagreement():
    pose, valid = synthetic_mhr70_pose(theta_deg=20.0)
    trunk = extract_trunk_features(pose, valid, SPEC, dt=1.0)

    features = compute_disagreement_features(pose, pose, trunk, trunk, valid, valid)

    torch.testing.assert_close(features.coordinate_abs_delta, torch.zeros_like(pose))
    torch.testing.assert_close(features.angle_abs_delta, torch.zeros_like(trunk.angle))
    torch.testing.assert_close(features.rotation_distance, torch.zeros_like(trunk.angle))


def test_pose_features_are_mask_aware_and_use_dynamic_skeleton_bones():
    pose, valid = synthetic_mhr70_pose(frames=2)
    pose[:, 1] += 2.0
    valid[:, 1, SPEC.joint_index("neck")] = False

    features = extract_pose_features(pose, valid, SPEC, dt=1.0)

    assert features.bone_lengths.shape == (1, 2, len(SPEC.bones))
    assert features.velocity.shape == pose.shape
    assert not features.velocity_valid[:, 0].any()
    assert not features.valid[:, 1, SPEC.joint_index("neck")].any()
    assert torch.isfinite(features.velocity).all()
    assert torch.isfinite(features.bone_lengths).all()


def test_quality_is_finite_config_driven_and_detached():
    pose, valid = synthetic_mhr70_pose(theta_deg=15.0)
    pose.requires_grad_()
    trunk = extract_trunk_features(pose, valid, SPEC, dt=1.0)

    quality = compute_quality_features(pose, valid, trunk, SPEC)

    assert quality.score.shape == pose.shape[:2]
    assert not quality.score.requires_grad
    assert not quality.loss_weight.requires_grad
    assert torch.isfinite(quality.score).all()
    assert torch.isfinite(quality.loss_weight).all()
    assert (quality.score >= 0).all() and (quality.score <= 1).all()


def test_quality_uses_explicit_fixed_configuration_weights():
    pose, valid = synthetic_mhr70_pose(theta_deg=15.0)
    neck = SPEC.joint_index("neck")
    pose[:, 1, neck] += torch.tensor([4.0, 0.0, 0.0])
    trunk = extract_trunk_features(pose, valid, SPEC, dt=1.0)

    unweighted = compute_quality_features(pose, valid, trunk, SPEC)
    rigidity_weighted = compute_quality_features(pose, valid, trunk, SPEC, QualityConfig(rigidity_weight=10.0))

    assert rigidity_weighted.score[:, 1].item() < unweighted.score[:, 1].item()


def test_rigidity_residual_uses_only_valid_bones():
    pose, valid = synthetic_mhr70_pose(frames=3)
    pose[:, 2, SPEC.joint_index("right-hip")] += torch.tensor([1.0, 0.0, 0.0])
    trunk = extract_trunk_features(pose, valid, SPEC, dt=1.0)
    valid_bone_spec = SkeletonSpec(SPEC.name, SPEC.joint_names, (SPEC.bones[0],), SPEC.roles, SPEC.required_roles)
    masked_bone_spec = SkeletonSpec(SPEC.name, SPEC.joint_names, (SPEC.bones[0], SPEC.bones[1]), SPEC.roles, SPEC.required_roles)

    valid_only = compute_quality_features(pose, valid, trunk, valid_bone_spec)
    with_invalid_bone = compute_quality_features(pose, valid, trunk, masked_bone_spec)

    torch.testing.assert_close(with_invalid_bone.rigidity_residual, valid_only.rigidity_residual)


def test_static_pose_with_different_bone_lengths_has_zero_rigidity_and_high_quality():
    pose, valid = synthetic_mhr70_pose(frames=3)
    trunk = extract_trunk_features(pose, valid, SPEC, dt=1.0)

    quality = compute_quality_features(pose, valid, trunk, SPEC)

    torch.testing.assert_close(quality.rigidity_residual, torch.zeros_like(quality.rigidity_residual), atol=1e-6, rtol=0)
    assert (quality.score > 0.05).all()


def test_disagreement_masks_invalid_coordinates_and_is_swap_symmetric():
    face, valid_face = synthetic_mhr70_pose(theta_deg=10.0)
    side, valid_side = synthetic_mhr70_pose(theta_deg=-10.0)
    valid_side[:, :, SPEC.joint_index("neck")] = False
    face_trunk = extract_trunk_features(face, valid_face, SPEC, dt=1.0)
    side_trunk = extract_trunk_features(side, valid_side, SPEC, dt=1.0)

    forward = compute_disagreement_features(face, side, face_trunk, side_trunk, valid_face, valid_side)
    reverse = compute_disagreement_features(side, face, side_trunk, face_trunk, valid_side, valid_face)

    assert not forward.coordinate_valid[:, :, SPEC.joint_index("neck")].any()
    assert not forward.coordinate_abs_delta[:, :, SPEC.joint_index("neck")].any()
    torch.testing.assert_close(forward.coordinate_abs_delta, reverse.coordinate_abs_delta)
    torch.testing.assert_close(forward.rotation_distance, reverse.rotation_distance)


def test_disagreement_center_requires_common_valid_coordinates():
    face, face_valid = synthetic_mhr70_pose()
    side, side_valid = synthetic_mhr70_pose()
    face_valid[:] = False
    side_valid[:] = False
    face_valid[:, :, SPEC.joint_index("left-hip")] = True
    side_valid[:, :, SPEC.joint_index("right-hip")] = True
    face_trunk = extract_trunk_features(face, face_valid, SPEC, dt=1.0)
    side_trunk = extract_trunk_features(side, side_valid, SPEC, dt=1.0)

    features = compute_disagreement_features(face, side, face_trunk, side_trunk, face_valid, side_valid)

    assert not features.trunk_displacement_valid.any()
    torch.testing.assert_close(features.trunk_displacement_abs_delta, torch.zeros_like(features.trunk_displacement_abs_delta))
