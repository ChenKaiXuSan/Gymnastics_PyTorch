from pathlib import Path

import torch

from fuse.rotation_aware.config import load_skeleton_spec
from fuse.rotation_aware.features import (
    QualityConfig,
    compute_disagreement_features,
    compute_quality_features,
    extract_pose_features,
)
from fuse.rotation_aware.trunk import extract_trunk_features
from tests.rotation_aware.test_geometry import synthetic_mhr70_pose


SPEC = load_skeleton_spec(Path("configs/fuse/skeleton_mhr70.yaml"))


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
