from __future__ import annotations

from pathlib import Path

import pytest
import torch

import fuse.rotation_aware.model as rotation_model
from fuse.rotation_aware.config import RoleSpec, SkeletonSpec, load_skeleton_spec
from fuse.rotation_aware.features import (
    DisagreementFeatures,
    FeatureBundle,
    compute_disagreement_features,
    compute_quality_features,
    extract_pose_features,
)
from fuse.rotation_aware.model import RotationAwareFusionModel
from fuse.rotation_aware.trunk import extract_trunk_features
from tests.rotation_aware.test_geometry import synthetic_mhr70_pose


SPEC = load_skeleton_spec(Path("configs/fuse/skeleton_mhr70.yaml"))


def _feature_bundle(points: torch.Tensor, valid: torch.Tensor, spec: SkeletonSpec = SPEC) -> FeatureBundle:
    trunk = extract_trunk_features(points, valid, spec, dt=1.0)
    return FeatureBundle(
        pose=extract_pose_features(points, valid, spec, dt=1.0),
        quality=compute_quality_features(points, valid, trunk, spec),
    )


def _inputs(*, frames: int = 5) -> tuple[
    torch.Tensor,
    torch.Tensor,
    FeatureBundle,
    FeatureBundle,
    DisagreementFeatures,
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


def _tiny_spec() -> SkeletonSpec:
    joint_names = (
        "left-hip",
        "right-hip",
        "left-acromion",
        "right-acromion",
        "neck",
        "left-wrist",
        "right-wrist",
    )
    roles = {
        "left_hip": RoleSpec("joint", ("left-hip",)),
        "right_hip": RoleSpec("joint", ("right-hip",)),
        "pelvis": RoleSpec("midpoint", ("left-hip", "right-hip")),
        "thorax": RoleSpec("midpoint", ("left-acromion", "right-acromion")),
        "left_shoulder": RoleSpec("joint", ("left-acromion",)),
        "right_shoulder": RoleSpec("joint", ("right-acromion",)),
        "left_acromion": RoleSpec("joint", ("left-acromion",)),
        "right_acromion": RoleSpec("joint", ("right-acromion",)),
        "neck": RoleSpec("joint", ("neck",)),
    }
    return SkeletonSpec("tiny", joint_names, ((0, 1), (2, 3), (0, 2), (1, 3)), roles, tuple(roles))


def _tiny_inputs() -> tuple[
    SkeletonSpec,
    torch.Tensor,
    torch.Tensor,
    FeatureBundle,
    FeatureBundle,
    DisagreementFeatures,
    torch.Tensor,
    torch.Tensor,
]:
    spec = _tiny_spec()
    pose = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.2, 0.0],
            [-1.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
        ]
    )
    face = pose[None, None].repeat(1, 4, 1, 1)
    side = face.clone()
    side[:, :, 2:4, 2] = torch.tensor([-0.2, 0.2])
    valid_face = torch.ones(face.shape[:-1], dtype=torch.bool)
    valid_side = torch.ones_like(valid_face)
    face_features = _feature_bundle(face, valid_face, spec)
    side_features = _feature_bundle(side, valid_side, spec)
    cross = compute_disagreement_features(
        face,
        side,
        extract_trunk_features(face, valid_face, spec, dt=1.0),
        extract_trunk_features(side, valid_side, spec, dt=1.0),
        valid_face,
        valid_side,
    )
    return spec, face, side, face_features, side_features, cross, valid_face, valid_side


def _rebuild_inputs(
    face: torch.Tensor,
    side: torch.Tensor,
    valid_face: torch.Tensor,
    valid_side: torch.Tensor,
    spec: SkeletonSpec = SPEC,
) -> tuple[FeatureBundle, FeatureBundle, DisagreementFeatures]:
    face_features = _feature_bundle(face, valid_face, spec)
    side_features = _feature_bundle(side, valid_side, spec)
    cross = compute_disagreement_features(
        face,
        side,
        extract_trunk_features(face, valid_face, spec, dt=1.0),
        extract_trunk_features(side, valid_side, spec, dt=1.0),
        valid_face,
        valid_side,
    )
    return face_features, side_features, cross


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

    for field in ("fused_kpts", "base_kpts", "delta_kpts", "fused_theta", "fused_r_pt"):
        torch.testing.assert_close(getattr(out_lr, field), getattr(out_rl, field), atol=1e-5, rtol=0)
    for field in ("valid", "fused_theta_valid", "fused_r_pt_valid"):
        assert torch.equal(getattr(out_lr, field), getattr(out_rl, field))


def test_model_supports_dynamic_joint_count_and_per_joint_bounded_residuals() -> None:
    spec, face, side, face_features, side_features, cross, valid_face, valid_side = _tiny_inputs()
    limits = torch.linspace(0.01, 0.05, len(spec.joint_names))
    model = RotationAwareFusionModel(spec, hidden_channels=16, max_delta_by_joint=limits)

    out = model(face, side, face_features, side_features, cross, valid_face, valid_side)

    assert out.fused_kpts.shape == face.shape
    torch.testing.assert_close(out.delta_kpts.abs(), out.delta_kpts.abs().clamp_max(limits[None, None, :, None]))


def test_invalid_padded_suffix_does_not_change_valid_prefix_outputs() -> None:
    torch.manual_seed(19)
    face, side, face_features, side_features, cross, valid_face, valid_side = _inputs()
    model = RotationAwareFusionModel(SPEC, hidden_channels=16).eval()
    reference = model(face, side, face_features, side_features, cross, valid_face, valid_side)
    suffix = 3
    padded_face = torch.cat((face, torch.randn_like(face[:, :suffix])), dim=1)
    padded_side = torch.cat((side, torch.randn_like(side[:, :suffix])), dim=1)
    padded_valid_face = torch.cat((valid_face, torch.zeros_like(valid_face[:, :suffix])), dim=1)
    padded_valid_side = torch.cat((valid_side, torch.zeros_like(valid_side[:, :suffix])), dim=1)
    padded_face_features, padded_side_features, padded_cross = _rebuild_inputs(
        padded_face,
        padded_side,
        padded_valid_face,
        padded_valid_side,
    )

    padded = model(
        padded_face,
        padded_side,
        padded_face_features,
        padded_side_features,
        padded_cross,
        padded_valid_face,
        padded_valid_side,
    )

    prefix = slice(None, face.shape[1])
    for field in ("fused_kpts", "base_kpts", "delta_kpts", "fused_theta", "fused_r_pt"):
        torch.testing.assert_close(getattr(reference, field), getattr(padded, field)[:, prefix], atol=1e-7, rtol=0)
    for field in ("valid", "fused_theta_valid", "fused_r_pt_valid"):
        assert torch.equal(getattr(reference, field), getattr(padded, field)[:, prefix])


def test_explicit_temporal_valid_mask_excludes_padded_frames() -> None:
    face, side, _, _, _, valid_face, valid_side = _inputs()
    temporal_valid = torch.ones(face.shape[:2], dtype=torch.bool)
    temporal_valid[:, -1] = False
    face_features, side_features, cross = _rebuild_inputs(
        face,
        side,
        valid_face & temporal_valid[..., None],
        valid_side & temporal_valid[..., None],
    )
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)

    out = model(
        face,
        side,
        face_features,
        side_features,
        cross,
        valid_face,
        valid_side,
        temporal_valid=temporal_valid,
    )

    assert not out.valid[:, -1].any()
    torch.testing.assert_close(out.delta_kpts[:, -1], torch.zeros_like(out.delta_kpts[:, -1]), atol=0, rtol=0)


def test_temporal_mask_rejects_features_built_before_masking() -> None:
    face, side, face_features, side_features, cross, valid_face, valid_side = _inputs()
    temporal_valid = torch.ones(face.shape[:2], dtype=torch.bool)
    temporal_valid[:, -1] = False
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)

    with pytest.raises(ValueError, match="features must be recomputed after temporal masking"):
        model(
            face,
            side,
            face_features,
            side_features,
            cross,
            valid_face,
            valid_side,
            temporal_valid=temporal_valid,
        )


def test_model_rejects_features_from_different_source_points() -> None:
    face, side, _, side_features, cross, valid_face, valid_side = _inputs()
    shifted_face = face.clone()
    shifted_face[:, :, SPEC.joint_index("neck"), 0] += 0.5
    shifted_face_features = _feature_bundle(shifted_face, valid_face)
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)

    with pytest.raises(ValueError, match="pose.points must match the supplied effective points"):
        model(face, side, shifted_face_features, side_features, cross, valid_face, valid_side)


def test_recomputed_masked_features_ignore_excluded_coordinates() -> None:
    torch.manual_seed(29)
    face, side, _, _, _, valid_face, valid_side = _inputs()
    temporal_valid = torch.ones(face.shape[:2], dtype=torch.bool)
    temporal_valid[:, -1] = False
    effective_face_valid = valid_face & temporal_valid[..., None]
    effective_side_valid = valid_side & temporal_valid[..., None]
    noisy_face = face.clone()
    noisy_side = side.clone()
    noisy_face[:, -1] = torch.randn_like(noisy_face[:, -1]) * 1000
    noisy_side[:, -1] = torch.randn_like(noisy_side[:, -1]) * 1000
    clean_face_features, clean_side_features, clean_cross = _rebuild_inputs(
        face,
        side,
        effective_face_valid,
        effective_side_valid,
    )
    noisy_face_features, noisy_side_features, noisy_cross = _rebuild_inputs(
        noisy_face,
        noisy_side,
        effective_face_valid,
        effective_side_valid,
    )
    model = RotationAwareFusionModel(SPEC, hidden_channels=16).eval()

    clean = model(
        face,
        side,
        clean_face_features,
        clean_side_features,
        clean_cross,
        valid_face,
        valid_side,
        temporal_valid=temporal_valid,
    )
    noisy = model(
        noisy_face,
        noisy_side,
        noisy_face_features,
        noisy_side_features,
        noisy_cross,
        valid_face,
        valid_side,
        temporal_valid=temporal_valid,
    )

    prefix = slice(None, -1)
    for field in ("fused_kpts", "base_kpts", "delta_kpts", "fused_theta", "fused_r_pt"):
        torch.testing.assert_close(getattr(clean, field)[:, prefix], getattr(noisy, field)[:, prefix], atol=0, rtol=0)
    for field in ("valid", "fused_theta_valid", "fused_r_pt_valid"):
        assert torch.equal(getattr(clean, field)[:, prefix], getattr(noisy, field)[:, prefix])


def test_internal_invalid_coordinates_do_not_change_valid_outputs() -> None:
    torch.manual_seed(23)
    face, side, _, _, _, valid_face, valid_side = _inputs()
    invalid_face = valid_face.clone()
    invalid_side = valid_side.clone()
    neck = SPEC.joint_index("neck")
    invalid_face[:, 2, neck] = False
    invalid_side[:, 2, neck] = False
    clean_face = face.clone()
    clean_side = side.clone()
    clean_face[:, 2, neck] = 0
    clean_side[:, 2, neck] = 0
    noisy_face = clean_face.clone()
    noisy_side = clean_side.clone()
    noisy_face[:, 2, neck] = torch.tensor([999.0, -999.0, 777.0])
    noisy_side[:, 2, neck] = torch.tensor([-555.0, 444.0, -333.0])
    clean_face_features, clean_side_features, clean_cross = _rebuild_inputs(
        clean_face,
        clean_side,
        invalid_face,
        invalid_side,
    )
    noisy_face_features, noisy_side_features, noisy_cross = _rebuild_inputs(
        noisy_face,
        noisy_side,
        invalid_face,
        invalid_side,
    )
    model = RotationAwareFusionModel(SPEC, hidden_channels=16).eval()

    clean = model(
        clean_face,
        clean_side,
        clean_face_features,
        clean_side_features,
        clean_cross,
        invalid_face,
        invalid_side,
    )
    noisy = model(
        noisy_face,
        noisy_side,
        noisy_face_features,
        noisy_side_features,
        noisy_cross,
        invalid_face,
        invalid_side,
    )

    valid = clean.valid
    torch.testing.assert_close(clean.fused_kpts[valid], noisy.fused_kpts[valid], atol=0, rtol=0)
    torch.testing.assert_close(clean.base_kpts[valid], noisy.base_kpts[valid], atol=0, rtol=0)
    torch.testing.assert_close(clean.delta_kpts[valid], noisy.delta_kpts[valid], atol=0, rtol=0)


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
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_model_propagates_physical_dt_to_every_trunk_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    face, side, face_features, side_features, cross, valid_face, valid_side = _inputs()
    dt = torch.full(face.shape[:2], 1 / 120)
    seen: list[torch.Tensor] = []
    original = rotation_model.extract_trunk_features

    def capture(points: torch.Tensor, valid: torch.Tensor, spec: SkeletonSpec, dt: float | torch.Tensor):
        if isinstance(dt, torch.Tensor):
            seen.append(dt)
        return original(points, valid, spec, dt)

    monkeypatch.setattr(rotation_model, "extract_trunk_features", capture)
    model = RotationAwareFusionModel(SPEC, hidden_channels=8)

    model(face, side, face_features, side_features, cross, valid_face, valid_side, dt=dt)

    assert len(seen) == 3
    for received in seen:
        torch.testing.assert_close(received, dt)
