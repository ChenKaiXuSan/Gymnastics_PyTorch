from __future__ import annotations

import torch

from fusion.archive.rotation_aware.cli import (
    LEARNED_ABLATIONS,
    _training_config_for_ablation,
    architecture_for_training,
    build_fusion_model,
    loss_config_for_ablation,
    model_metadata_for_training,
)
from fusion.keypoints.config import load_skeleton_spec
from fusion.archive.rotation_aware.evaluation import ABLATION_REGISTRY, LEARNED_ABLATION_IDS
from fusion.keypoints.features import (
    compute_disagreement_features,
    compute_quality_features,
    extract_pose_features,
)
from fusion.archive.rotation_aware.losses import LossConfig
from fusion.archive.rotation_aware.model import FusionOutput, RotationAwareFusionModel
from fusion.archive.rotation_aware.plain_tcn import PlainTemporalFusionModel
from fusion.keypoints.trunk import extract_trunk_features
from fusion.keypoints.features import FeatureBundle

SKELETON = load_skeleton_spec("src/configs/shared/skeleton_mhr70.yaml")


def _bundle(points: torch.Tensor, valid: torch.Tensor):
    pose = extract_pose_features(points, valid, SKELETON, dt=1.0 / 60.0)
    trunk = extract_trunk_features(points, valid, SKELETON, dt=1.0 / 60.0)
    quality = compute_quality_features(points, valid, trunk, SKELETON)
    return FeatureBundle(pose=pose, quality=quality), trunk


def _inputs(batch: int = 2, frames: int = 40):
    torch.manual_seed(0)
    face = torch.randn(batch, frames, 70, 3) * 0.3
    side = face + torch.randn_like(face) * 0.02
    valid_face = torch.ones(batch, frames, 70, dtype=torch.bool)
    valid_side = torch.ones(batch, frames, 70, dtype=torch.bool)
    valid_side[:, 5:8, 40] = False
    face_features, face_trunk = _bundle(face, valid_face)
    side_features, side_trunk = _bundle(side, valid_side)
    cross = compute_disagreement_features(
        face, side, face_trunk, side_trunk, valid_face, valid_side
    )
    return face, side, face_features, side_features, cross, valid_face, valid_side


def test_b1_is_registered_everywhere() -> None:
    assert "B1" in LEARNED_ABLATIONS
    assert loss_config_for_ablation("B1") == LossConfig()
    assert ABLATION_REGISTRY["B1"] == "plain_temporal_baseline"
    assert "B1" in LEARNED_ABLATION_IDS
    training = _training_config_for_ablation({"training": {"epochs": 3}}, "B1")
    assert training["architecture"] == "plain_tcn"
    assert model_metadata_for_training(training)["architecture"] == "plain_tcn"
    assert architecture_for_training({"ablation": "A6"}) == "rotation_aware"
    assert isinstance(build_fusion_model(SKELETON, training), PlainTemporalFusionModel)
    assert isinstance(build_fusion_model(SKELETON, {"ablation": "A6"}), RotationAwareFusionModel)


def test_plain_tcn_returns_full_fusion_output_with_union_mask() -> None:
    face, side, ff, sf, cross, vf, vs = _inputs()
    model = PlainTemporalFusionModel(SKELETON, hidden_channels=16)
    output = model(face, side, ff, sf, cross, vf, vs, dt=1.0 / 60.0)
    assert isinstance(output, FusionOutput)
    assert output.fused_kpts.shape == face.shape
    assert torch.equal(output.valid, vf | vs)
    assert torch.isfinite(output.fused_kpts).all()
    torch.testing.assert_close(output.delta_kpts, output.fused_kpts - output.base_kpts)
    # The base pose is the same quality-weighted mean the rotation-aware model uses.
    reference = RotationAwareFusionModel(SKELETON, hidden_channels=16)
    torch.testing.assert_close(
        output.base_kpts,
        reference(face, side, ff, sf, cross, vf, vs, dt=1.0 / 60.0).base_kpts,
    )


def test_plain_tcn_is_not_view_order_invariant_and_trains() -> None:
    face, side, ff, sf, cross, vf, vs = _inputs()
    model = PlainTemporalFusionModel(SKELETON, hidden_channels=16)
    forward = model(face, side, ff, sf, cross, vf, vs, dt=1.0 / 60.0).fused_kpts
    swapped = model(side, face, sf, ff, cross, vs, vf, dt=1.0 / 60.0).fused_kpts
    assert not torch.allclose(forward, swapped, atol=1e-5)
    loss = forward.square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_plain_tcn_rejects_camera_features() -> None:
    face, side, ff, sf, cross, vf, vs = _inputs(frames=8)
    model = PlainTemporalFusionModel(SKELETON, hidden_channels=8)
    try:
        model(face, side, ff, sf, cross, vf, vs, camera_features=object())
    except ValueError as error:
        assert "camera" in str(error)
    else:  # pragma: no cover
        raise AssertionError("camera features must be rejected")


def test_b2_uses_bounded_residual_and_b1_does_not() -> None:
    b1 = _training_config_for_ablation({"training": {"epochs": 1}}, "B1")
    b2 = _training_config_for_ablation({"training": {"epochs": 1}}, "B2")
    assert b1["max_delta"] == 0.0 and b2["max_delta"] == 0.05
    model_b1 = build_fusion_model(SKELETON, b1)
    model_b2 = build_fusion_model(SKELETON, b2)
    assert isinstance(model_b2, PlainTemporalFusionModel) and model_b2.max_delta == 0.05
    face, side, ff, sf, cross, vf, vs = _inputs(frames=16)
    with torch.no_grad():
        for parameter in model_b2.head.parameters():
            parameter.fill_(5.0)
        out = model_b2(face, side, ff, sf, cross, vf, vs, dt=1.0 / 60.0)
    assert out.delta_kpts.abs().max() <= 0.05 + 1e-6
    assert ABLATION_REGISTRY["B2"] == "plain_temporal_baseline_bounded"
    assert model_b1.max_delta == 0.0
