from __future__ import annotations

import pytest
import torch

from gymnastics.fusion.model import CycleAwareFusionModel, CycleAwareModelConfig
from gymnastics.fusion.outputs import PoseFusionOutput


def test_forward_shapes_and_identities(tiny_config, tiny_batch, skeleton):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config)
    out = model(**tiny_batch)
    B, T, J = tiny_batch["pose_a"].shape[:3]
    D = tiny_config.hidden_dim
    assert isinstance(out, PoseFusionOutput)
    assert out.pose.shape == (B, T, J, 3)
    assert out.weight_a.shape == (B, T, J, 1) and out.reliability_logits.shape == (B, T, J, 2)
    for name in ("pose_feature_a", "short_motion_b", "long_motion_a", "motion_feature_b", "guided_feature_a", "cross_feature_b"):
        assert getattr(out, name).shape == (B, T, J, D)
    torch.testing.assert_close(out.pose, out.base_pose + out.delta_pose)
    torch.testing.assert_close(out.weight_a + out.weight_b, torch.ones(B, T, J, 1))
    assert torch.isfinite(out.pose).all()
    # Padding frames and doubly-invalid joints are invalid in the output.
    assert not out.valid[1, 12:].any()
    assert not out.valid[1, 5:9, 7].any()
    assert out.valid[0, :, 3].all()  # valid in B only
    torch.testing.assert_close(out.weight_b[0, :, 3], torch.ones(T, 1))


def test_zero_init_model_equals_weighted_fusion(tiny_config, tiny_batch):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config)
    out = model(**tiny_batch)
    expected = out.weight_a * tiny_batch["pose_a"] + out.weight_b * tiny_batch["pose_b"]
    expected = torch.where(out.valid[..., None], expected, torch.zeros_like(expected))
    torch.testing.assert_close(out.pose, expected)
    assert torch.equal(out.delta_pose, torch.zeros_like(out.delta_pose))


def test_backward_pass_reaches_every_parameter(tiny_config, tiny_batch):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config)
    out = model(**tiny_batch)
    loss = out.pose.square().mean() + out.reliability_logits.square().mean() + out.motion_feature_a.square().mean()
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_view_swap_symmetry(tiny_config, tiny_batch):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config).eval()
    with torch.no_grad():
        out = model(**tiny_batch)
        swapped = model(
            tiny_batch["pose_b"], tiny_batch["pose_a"], tiny_batch["valid_b"], tiny_batch["valid_a"],
            tiny_batch["delta_t"], tiny_batch["phase"], tiny_batch["phase_valid"], tiny_batch["frame_mask"],
        )
    torch.testing.assert_close(out.pose, swapped.pose, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out.weight_a, swapped.weight_b, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"short_motion": {"enabled": False}},
        {"long_motion": {"enabled": False}},
        {"phase_encoding": {"enabled": False}},
        {"film": {"enabled": False}},
        {"cross_view": {"enabled": False}},
        {"reliability": {"enabled": False}},
        {"residual": {"enabled": False}},
        {"residual": {"max_delta": None}},
    ],
)
def test_ablation_switches(tiny_config, tiny_batch, overrides):
    config = tiny_config.to_dict()
    for key, value in overrides.items():
        config[key].update(value)
    model = CycleAwareFusionModel(config)
    out = model(**tiny_batch)
    assert torch.isfinite(out.pose).all()
    if overrides.get("film", {}).get("enabled") is False:
        torch.testing.assert_close(out.guided_feature_a, out.pose_feature_a)
    if overrides.get("reliability", {}).get("enabled") is False:
        assert torch.equal(out.reliability_logits, torch.zeros_like(out.reliability_logits))


def test_config_rejects_unknown_fields_and_bad_values():
    with pytest.raises(ValueError):
        CycleAwareModelConfig.from_mapping({"hidden_dims": 8})
    with pytest.raises(ValueError):
        CycleAwareModelConfig(hidden_dim=10, num_heads=4)
    assert CycleAwareModelConfig.from_mapping({"samples_per_cycle": 16, "long_motion": {"num_cycles": 1.5}}).window_length == 24


def test_forward_without_phase_and_scalar_dt(tiny_config, tiny_batch):
    model = CycleAwareFusionModel(tiny_config)
    out = model(tiny_batch["pose_a"], tiny_batch["pose_b"], tiny_batch["valid_a"], tiny_batch["valid_b"], torch.tensor(1 / 30))
    assert torch.isfinite(out.pose).all()
    with pytest.raises(ValueError):
        model(tiny_batch["pose_a"], tiny_batch["pose_b"], tiny_batch["valid_a"], tiny_batch["valid_b"], torch.tensor(1 / 30), phase=tiny_batch["phase"])
