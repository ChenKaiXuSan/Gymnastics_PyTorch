from __future__ import annotations

import pytest
import torch

from gymnastics.fusion.rotation_aware.camera import (
    CameraConditioningConfig,
    CameraFeatureBundle,
)
from gymnastics.fusion.rotation_aware.model import RotationAwareFusionModel
from tests.rotation_aware.test_model import SPEC, _inputs


def _camera_bundle(
    face: torch.Tensor,
    *,
    global_channels: int = 19,
    joint_channels: int = 8,
) -> CameraFeatureBundle:
    batch, frames, joints = face.shape[:3]
    generator = torch.Generator().manual_seed(91)
    return CameraFeatureBundle(
        global_features=torch.randn(
            batch, global_channels, generator=generator
        ),
        joint_features=torch.randn(
            batch, frames, joints, joint_channels, generator=generator
        ),
        valid=torch.ones((batch, frames, joints), dtype=torch.bool),
    )


def _run(
    model: RotationAwareFusionModel,
    camera_features: CameraFeatureBundle | None = None,
):
    face, side, face_features, side_features, cross, valid_face, valid_side = (
        _inputs()
    )
    return model(
        face,
        side,
        face_features,
        side_features,
        cross,
        valid_face,
        valid_side,
        camera_features=camera_features,
    )


def test_camera_disabled_model_is_exactly_legacy_compatible() -> None:
    torch.manual_seed(4)
    legacy = RotationAwareFusionModel(SPEC, hidden_channels=16)
    explicit = RotationAwareFusionModel(
        SPEC, hidden_channels=16, camera_config=None
    )
    explicit.load_state_dict(legacy.state_dict())

    legacy_output = _run(legacy)
    explicit_output = _run(explicit)

    torch.testing.assert_close(
        explicit_output.fused_kpts,
        legacy_output.fused_kpts,
        atol=0,
        rtol=0,
    )
    assert not hasattr(explicit, "camera_conditioner")


@pytest.mark.parametrize("mode", ("additive", "film"))
def test_zero_initialized_camera_model_starts_as_exact_a6_and_receives_gradient(
    mode: str,
) -> None:
    torch.manual_seed(8)
    base = RotationAwareFusionModel(SPEC, hidden_channels=16)
    conditioned = RotationAwareFusionModel(
        SPEC,
        hidden_channels=16,
        camera_config=CameraConditioningConfig(
            global_channels=19,
            joint_channels=8,
            mode=mode,
        ),
    )
    missing, unexpected = conditioned.load_state_dict(
        base.state_dict(), strict=False
    )
    assert missing
    assert all(
        name.startswith(("camera_conditioner.", "camera_delta_head."))
        for name in missing
    )
    assert not unexpected
    face = _inputs()[0]
    camera = _camera_bundle(face)

    base_output = _run(base)
    conditioned_output = _run(conditioned, camera)

    torch.testing.assert_close(
        conditioned_output.fused_kpts,
        base_output.fused_kpts,
        atol=0,
        rtol=0,
    )
    conditioned_output.fused_kpts.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in conditioned.camera_conditioner.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().sum() > 0 for value in gradients)


def test_camera_model_requires_matching_finite_feature_shapes() -> None:
    face = _inputs()[0]
    model = RotationAwareFusionModel(
        SPEC,
        hidden_channels=16,
        camera_config=CameraConditioningConfig(19, 8, mode="film"),
    )
    camera = _camera_bundle(face)
    wrong = CameraFeatureBundle(
        global_features=camera.global_features,
        joint_features=camera.joint_features[:, :-1],
        valid=camera.valid[:, :-1],
    )

    with pytest.raises(ValueError, match="camera joint features"):
        _run(model, wrong)
    with pytest.raises(ValueError, match="requires camera_features"):
        _run(model)


def test_camera_bundle_zeros_invalid_nonfinite_rows() -> None:
    face = _inputs()[0]
    camera = _camera_bundle(face)
    joint = camera.joint_features.clone()
    valid = camera.valid.clone()
    valid[:, 2, 5] = False
    joint[:, 2, 5] = torch.nan
    sanitized = CameraFeatureBundle(
        camera.global_features,
        joint,
        valid,
    ).validated(
        batch=face.shape[0],
        frames=face.shape[1],
        joints=face.shape[2],
        global_channels=19,
        joint_channels=8,
    )

    assert torch.equal(
        sanitized.joint_features[:, 2, 5],
        torch.zeros_like(sanitized.joint_features[:, 2, 5]),
    )


def test_camera_residual_bypasses_a_saturated_a6_delta_head() -> None:
    """Unity can saturate A6's tanh; camera gradients must still reach output."""
    torch.manual_seed(12)
    model = RotationAwareFusionModel(
        SPEC,
        hidden_channels=16,
        camera_config=CameraConditioningConfig(19, 8, mode="film"),
    )
    with torch.no_grad():
        model.delta_head.weight.zero_()
        model.delta_head.bias.fill_(-100.0)
        model.camera_delta_head.bias.fill_(0.25)
    face = _inputs()[0]
    camera = _camera_bundle(face)

    output = _run(model, camera)
    loss = output.fused_kpts.square().mean()
    loss.backward()

    assert model.camera_delta_head.weight.grad is not None
    assert model.camera_delta_head.weight.grad.abs().sum() > 0
    saturated_a6 = -model.max_delta_by_joint[None, None, :, None]
    assert not torch.allclose(output.delta_kpts, saturated_a6)
