from __future__ import annotations

import pytest
import torch

from gymnastics.fusion.cycle_aware.corruptions import (
    CORRUPTION_FAMILIES,
    CorruptionConfig,
    corrupt_view,
    corrupt_window,
    stable_seed,
)


def _clean(frames=24, joints=20):
    torch.manual_seed(0)
    pose = torch.randn(frames, joints, 3)
    return pose, torch.ones(frames, joints, dtype=torch.bool)


@pytest.mark.parametrize("family", CORRUPTION_FAMILIES)
def test_each_family_changes_input_and_reports_mask(skeleton, family):
    pose, valid = _clean()
    config = CorruptionConfig(
        families=(family,),
        joint_mask_probability=0.5,
        distal_mask_probability=1.0,
        gaussian_noise_probability=1.0,
        depth_probability=1.0,
        depth_shift_std=0.05,
        temporal_dropout_probability=0.5,
        contiguous_dropout_probability=1.0,
    )
    generator = torch.Generator().manual_seed(1)
    out, out_valid, mask = corrupt_view(pose, valid, generator, config, skeleton)
    assert out.shape == pose.shape and out_valid.shape == valid.shape
    assert mask.any(), family
    assert not (out_valid & ~torch.isfinite(out).all(dim=-1)).any()
    if family in {"joint_mask", "distal_mask", "temporal_dropout", "contiguous_dropout"}:
        assert (~out_valid).any()
        assert torch.equal(out[~out_valid], torch.zeros_like(out[~out_valid]))
    if family == "distal_mask":
        changed_joints = set(mask.any(dim=0).nonzero().flatten().tolist())
        assert changed_joints <= set(skeleton.distal_indices)
    if family == "contiguous_dropout":
        dropped = (~out_valid).all(dim=1).nonzero().flatten()
        assert len(dropped) == config.contiguous_block_length
        assert dropped.tolist() == list(range(dropped[0], dropped[0] + len(dropped)))
    # The clean input is untouched.
    assert torch.isfinite(pose).all() and valid.all()


def test_corrupt_window_is_deterministic_and_keeps_clean(skeleton):
    pose, valid = _clean()
    config = CorruptionConfig(view_probability=1.0)
    first = corrupt_window(pose, pose.clone(), valid, valid.clone(), seed=stable_seed(0, "w", 0), config=config, skeleton=skeleton)
    second = corrupt_window(pose, pose.clone(), valid, valid.clone(), seed=stable_seed(0, "w", 0), config=config, skeleton=skeleton)
    for key in first:
        assert torch.equal(first[key], second[key])
    other = corrupt_window(pose, pose.clone(), valid, valid.clone(), seed=stable_seed(0, "w", 1), config=config, skeleton=skeleton)
    assert any(not torch.equal(first[k], other[k]) for k in ("pose_a", "pose_b"))
    disabled = corrupt_window(pose, pose, valid, valid, seed=3, config=CorruptionConfig(enabled=False), skeleton=skeleton)
    assert torch.equal(disabled["pose_a"], pose) and not disabled["corruption_mask_a"].any()


def test_config_validation_and_mapping():
    with pytest.raises(ValueError):
        CorruptionConfig(families=("nope",))
    with pytest.raises(ValueError):
        CorruptionConfig(joint_mask_probability=1.5)
    config = CorruptionConfig.from_mapping({"families": ["joint_mask"], "joint_mask_probability": 0.2})
    assert config.families == ("joint_mask",) and config.to_dict()["joint_mask_probability"] == 0.2
    assert CorruptionConfig.from_mapping(None) == CorruptionConfig()
