from __future__ import annotations

from pathlib import Path

import torch

from fuse.rotation_aware.config import load_skeleton_spec
from fuse.rotation_aware.features import (
    FeatureBundle,
    compute_disagreement_features,
    compute_quality_features,
    extract_pose_features,
)
from fuse.rotation_aware.model import RotationAwareFusionModel
from fuse.rotation_aware.trunk import extract_trunk_features
from tests.rotation_aware.test_geometry import synthetic_mhr70_pose


SPEC = load_skeleton_spec(Path("configs/fuse/skeleton_mhr70.yaml"))


def _feature_bundle(points: torch.Tensor, valid: torch.Tensor) -> FeatureBundle:
    trunk = extract_trunk_features(points, valid, SPEC, dt=1.0)
    return FeatureBundle(
        pose=extract_pose_features(points, valid, SPEC, dt=1.0),
        quality=compute_quality_features(points, valid, trunk, SPEC),
    )


def _inputs(*, frames: int = 5) -> tuple[
    torch.Tensor,
    torch.Tensor,
    FeatureBundle,
    FeatureBundle,
    object,
    torch.Tensor,
    torch.Tensor,
]:
    face, valid_face = synthetic_mhr70_pose(theta_deg=20.0, frames=frames)
    side, valid_side = synthetic_mhr70_pose(theta_deg=-15.0, frames=frames)
    face[:, :, SPEC.joint_index("neck"), 0] += torch.linspace(0.0, 0.2, frames)
    side[:, :, SPEC.joint_index("neck"), 2] += torch.linspace(0.0, 0.15, frames)
    face_features = _feature_bundle(face, valid_face)
    side_features = _feature_bundle(side, valid_side)
    cross = compute_disagreement_features(
        face,
        side,
        extract_trunk_features(face, valid_face, SPEC, dt=1.0),
        extract_trunk_features(side, valid_side, SPEC, dt=1.0),
        valid_face,
        valid_side,
    )
    return face, side, face_features, side_features, cross, valid_face, valid_side


def test_model_uses_shared_encoders_and_the_required_noncausal_tcn_schedule() -> None:
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)

    assert hasattr(model, "view_encoder")
    assert not hasattr(model, "face_encoder")
    assert not hasattr(model, "side_encoder")
    assert [block.dilation for block in model.tcn.blocks] == [1, 2, 4, 8, 16, 32]
    for block in model.tcn.blocks:
        assert block.conv1.padding[0] == block.dilation
        assert block.conv2.padding[0] == block.dilation


def test_model_is_view_swap_invariant() -> None:
    torch.manual_seed(7)
    face, side, face_features, side_features, cross, valid_face, valid_side = _inputs()
    model = RotationAwareFusionModel(SPEC, hidden_channels=16).eval()

    out_lr = model(face, side, face_features, side_features, cross, valid_face, valid_side)
    swapped_cross = compute_disagreement_features(
        side,
        face,
        extract_trunk_features(side, valid_side, SPEC, dt=1.0),
        extract_trunk_features(face, valid_face, SPEC, dt=1.0),
        valid_side,
        valid_face,
    )
    out_rl = model(side, face, side_features, face_features, swapped_cross, valid_side, valid_face)

    torch.testing.assert_close(out_lr.fused_kpts, out_rl.fused_kpts, atol=1e-5, rtol=0)
    torch.testing.assert_close(out_lr.base_kpts, out_rl.base_kpts, atol=1e-5, rtol=0)
    torch.testing.assert_close(out_lr.delta_kpts, out_rl.delta_kpts, atol=1e-5, rtol=0)


def test_model_supports_dynamic_joint_count_and_per_joint_bounded_residuals() -> None:
    face, side, face_features, side_features, cross, valid_face, valid_side = _inputs()
    limits = torch.linspace(0.01, 0.05, len(SPEC.joint_names))
    model = RotationAwareFusionModel(SPEC, hidden_channels=16, max_delta_by_joint=limits)

    out = model(face, side, face_features, side_features, cross, valid_face, valid_side)

    assert out.fused_kpts.shape == face.shape
    torch.testing.assert_close(out.delta_kpts.abs(), out.delta_kpts.abs().clamp_max(limits[None, None, :, None]))


def test_model_preserves_exact_single_view_base_behavior() -> None:
    face, side, face_features, side_features, cross, valid_face, valid_side = _inputs()
    valid_side = torch.zeros_like(valid_side)
    side_features = _feature_bundle(side, valid_side)
    cross = compute_disagreement_features(
        face,
        side,
        extract_trunk_features(face, valid_face, SPEC, dt=1.0),
        extract_trunk_features(side, valid_side, SPEC, dt=1.0),
        valid_face,
        valid_side,
    )
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)

    out = model(face, side, face_features, side_features, cross, valid_face, valid_side)

    torch.testing.assert_close(out.fused_kpts, out.base_kpts, atol=0, rtol=0)
    torch.testing.assert_close(out.delta_kpts, torch.zeros_like(out.delta_kpts), atol=0, rtol=0)


def test_model_output_is_base_plus_delta_has_finite_gradients_and_recomputes_trunk() -> None:
    torch.manual_seed(11)
    face, side, face_features, side_features, cross, valid_face, valid_side = _inputs()
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)

    out = model(face, side, face_features, side_features, cross, valid_face, valid_side)
    expected_trunk = extract_trunk_features(out.fused_kpts, out.valid, SPEC, dt=1.0)

    torch.testing.assert_close(out.fused_kpts, out.base_kpts + out.delta_kpts)
    torch.testing.assert_close(out.fused_theta, expected_trunk.angle)
    torch.testing.assert_close(out.fused_r_pt, expected_trunk.rotation)
    torch.testing.assert_close(out.fused_theta_valid, expected_trunk.angle_valid)
    out.fused_kpts.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
