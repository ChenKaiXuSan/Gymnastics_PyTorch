"""Architecture v1.1: depth-aware base pose and the depth-axis data contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from common.skeletons.mhr70 import MHR70_INDEX, MHR70_NAMES
from fusion.baselines.methods import depth_aware_body_average
from fusion.cycle_target import ViewConsensusConfig, view_consensus
from fusion.data import build_datamodule
from fusion.data.sample_cache import load_samples, save_samples
from fusion.data.windows import CycleWindowDataset, WindowConfig
from fusion.keypoints.schema import PosePairTrial
from fusion.lightning_module import CycleAwareFusionModule
from fusion.losses import pseudo_target
from fusion.metrics import per_joint_error
from fusion.model import ARCHITECTURE_VERSION, CycleAwareFusionModel, CycleAwareModelConfig
from fusion.modules import depth_aware_pose_fusion, weighted_pose_fusion
from fusion.phase import normalize_sample_to_phase
from fusion.sample import CanonicalTransformRecord, camera_depth_axes, sample_from_pose_pair_trial
from fusion.train import compose_config

from .conftest import make_sample

E_X = torch.tensor([1.0, 0.0, 0.0])
E_Z = torch.tensor([0.0, 0.0, 1.0])


def _views(seed: int = 0, batch: int = 2, frames: int = 6, joints: int = 5):
    torch.manual_seed(seed)
    pose_a = torch.randn(batch, frames, joints, 3)
    pose_b = pose_a + 0.1 * torch.randn(batch, frames, joints, 3)
    valid = torch.ones(batch, frames, joints, dtype=torch.bool)
    half = torch.full((batch, frames, joints, 1), 0.5)
    depth_a = E_Z.expand(batch, frames, 3).clone()
    depth_b = E_X.expand(batch, frames, 3).clone()
    return pose_a, pose_b, valid, half, depth_a, depth_b


def test_alpha_zero_is_bit_identical_to_v1_0():
    pose_a, pose_b, valid, half, depth_a, depth_b = _views()
    weight_a = torch.rand_like(half)
    weight_b = 1.0 - weight_a
    fused, fused_valid = depth_aware_pose_fusion(pose_a, pose_b, weight_a, weight_b, valid, valid, depth_a, depth_b, alpha=0.0)
    expected, expected_valid = weighted_pose_fusion(pose_a, pose_b, weight_a, weight_b, valid, valid)
    assert torch.equal(fused, expected) and torch.equal(fused_valid, expected_valid)
    # Missing depth axes fall back to the same rule, whatever alpha is.
    fused, _ = depth_aware_pose_fusion(pose_a, pose_b, weight_a, weight_b, valid, valid, None, None, alpha=0.8)
    assert torch.equal(fused, expected)


def test_orthogonal_axes_give_the_per_axis_closed_form():
    pose_a, pose_b, valid, half, depth_a, depth_b = _views()
    alpha = 0.8
    fused, _ = depth_aware_pose_fusion(pose_a, pose_b, half, half, valid, valid, depth_a, depth_b, alpha=alpha)
    # View A looks along z (its depth), View B along x: x comes mostly from A,
    # z mostly from B, y is averaged; a view keeps precision 1 - alpha along its depth.
    expected = torch.stack(
        (
            (pose_a[..., 0] + (1 - alpha) * pose_b[..., 0]) / (2 - alpha),
            0.5 * (pose_a[..., 1] + pose_b[..., 1]),
            ((1 - alpha) * pose_a[..., 2] + pose_b[..., 2]) / (2 - alpha),
        ),
        dim=-1,
    )
    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-5)
    # Unequal scalar weights scale each view's whole precision matrix.
    weight_a = torch.full_like(half, 0.75)
    fused, _ = depth_aware_pose_fusion(pose_a, pose_b, weight_a, 1 - weight_a, valid, valid, depth_a, depth_b, alpha=alpha)
    x = (0.75 * pose_a[..., 0] + 0.25 * (1 - alpha) * pose_b[..., 0]) / (0.75 + 0.25 * (1 - alpha))
    torch.testing.assert_close(fused[..., 0], x, atol=1e-6, rtol=1e-5)


def test_single_valid_view_is_returned_exactly_and_invalid_joints_are_zero():
    pose_a, pose_b, valid, half, depth_a, depth_b = _views()
    valid_a, valid_b = valid.clone(), valid.clone()
    valid_a[0, :, 1] = False
    valid_b[1, 2:4, 3] = False
    valid_a[1, 2:4, 3] = False
    # Weights as the reliability head would produce them are irrelevant here:
    # invalid views must contribute nothing whatever the weights say.
    weight_a = torch.full_like(half, 0.3)
    fused, fused_valid = depth_aware_pose_fusion(pose_a, pose_b, weight_a, 1 - weight_a, valid_a, valid_b, depth_a, depth_b, alpha=0.8)
    torch.testing.assert_close(fused[0, :, 1], pose_b[0, :, 1], atol=1e-6, rtol=1e-5)
    assert not fused_valid[1, 2:4, 3].any() and torch.equal(fused[1, 2:4, 3], torch.zeros(2, 3))
    assert fused_valid[0, :, 1].all()


def test_parallel_axes_are_capped_and_stay_close_to_the_average():
    pose_a, pose_b, valid, half, depth_a, _ = _views()
    fused, _ = depth_aware_pose_fusion(pose_a, pose_b, half, half, valid, valid, depth_a, depth_a, alpha=0.8)
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused, 0.5 * (pose_a + pose_b), atol=1e-6, rtol=1e-5)


def test_gradient_reaches_the_reliability_weights():
    pose_a, pose_b, valid, half, depth_a, depth_b = _views()
    logits = torch.zeros(*half.shape[:-1], 2, requires_grad=True)
    weights = torch.softmax(logits, dim=-1)
    fused, _ = depth_aware_pose_fusion(pose_a, pose_b, weights[..., :1], weights[..., 1:], valid, valid, depth_a, depth_b, alpha=0.8)
    fused.square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_torch_rule_matches_the_deterministic_baseline():
    """Equal weights + alpha reproduce ``avg_body_depthaware`` on world keypoints."""
    rng = np.random.default_rng(3)
    frames = 12
    body = rng.normal(scale=0.3, size=(frames, 70, 3)).astype(np.float32)
    body[:, 9] = (-0.1, 0.0, 0.0)  # hips define the body frame (pelvis at the origin)
    body[:, 10] = (0.1, 0.0, 0.0)
    body[:, 5] = (-0.2, 0.5, 0.0)  # shoulders
    body[:, 6] = (0.2, 0.5, 0.0)
    noise_a = np.zeros_like(body)
    noise_a[..., 2] = rng.normal(scale=0.05, size=(frames, 70))
    noise_b = np.zeros_like(body)
    noise_b[..., 0] = rng.normal(scale=0.05, size=(frames, 70))
    for j in (5, 6, 9, 10):
        noise_a[:, j] = noise_b[:, j] = 0.0
    face_world = body + noise_a  # camera frame == body frame, depth = body z
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    side_world = (body + noise_b) @ rotation  # side camera looks along body x
    baseline = depth_aware_body_average(face_world, side_world, 0.8, 0.8)
    pose_a = torch.from_numpy(face_world)[None]
    pose_b = torch.from_numpy(body + noise_b)[None]
    valid = torch.ones(1, frames, 70, dtype=torch.bool)
    half = torch.full((1, frames, 70, 1), 0.5)
    depth_a = E_Z.expand(1, frames, 3).clone()
    depth_b = E_X.expand(1, frames, 3).clone()
    fused, _ = depth_aware_pose_fusion(pose_a, pose_b, half, half, valid, valid, depth_a, depth_b, alpha=0.8)
    np.testing.assert_allclose(fused[0].numpy(), baseline, atol=2e-5)


def test_model_uses_the_depth_aware_base_and_alpha_zero_reproduces_v1_0(tiny_config, tiny_batch):
    torch.manual_seed(0)
    B, T = tiny_batch["pose_a"].shape[:2]
    depth_a = E_Z.expand(B, T, 3).clone()
    depth_b = E_X.expand(B, T, 3).clone()
    model = CycleAwareFusionModel(tiny_config)
    assert tiny_config.fusion.depth_alpha == 0.8 and tiny_config.architecture_version == ARCHITECTURE_VERSION == "1.1"
    out = model(**tiny_batch, depth_a=depth_a, depth_b=depth_b)
    expected, _ = depth_aware_pose_fusion(
        torch.where(tiny_batch["valid_a"][..., None], tiny_batch["pose_a"], 0.0),
        torch.where(tiny_batch["valid_b"][..., None], tiny_batch["pose_b"], 0.0),
        out.weight_a, out.weight_b, out.valid & tiny_batch["valid_a"], out.valid & tiny_batch["valid_b"], depth_a, depth_b, alpha=0.8,
    )
    torch.testing.assert_close(out.base_pose, torch.where(out.valid[..., None], expected, torch.zeros_like(expected)), atol=1e-6, rtol=1e-5)
    # Along the shared (y) axis the base is the plain weighted average.
    plain = out.weight_a * tiny_batch["pose_a"] + out.weight_b * tiny_batch["pose_b"]
    both = (tiny_batch["valid_a"] & tiny_batch["valid_b"] & out.valid)
    torch.testing.assert_close(out.base_pose[..., 1][both], plain[..., 1][both], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(out.base_pose[..., 0][both], plain[..., 0][both])
    # alpha = 0 is the v1.0 architecture, bit for bit, and reports itself as such.
    v10 = CycleAwareFusionModel(replace(tiny_config, fusion=replace(tiny_config.fusion, depth_alpha=0.0)))
    v10.load_state_dict(model.state_dict())
    assert v10.config.architecture_version == "1.0"
    with torch.no_grad():
        old = v10(**tiny_batch, depth_a=depth_a, depth_b=depth_b)
        without_axes = model(**tiny_batch)
    torch.testing.assert_close(old.base_pose, torch.where(old.valid[..., None], plain, torch.zeros_like(plain)))
    assert torch.equal(old.base_pose, without_axes.base_pose)
    # Config plumbing and validation.
    cfg = CycleAwareModelConfig.from_mapping({"hidden_dim": 16, "num_heads": 2, "fusion": {"depth_alpha": 0.5}})
    assert cfg.fusion.depth_alpha == 0.5 and cfg.to_dict()["fusion"]["min_precision"] == 0.5
    with pytest.raises(ValueError):
        CycleAwareModelConfig.from_mapping({"fusion": {"depth_alpha": 1.0}})
    with pytest.raises(ValueError):
        model(**tiny_batch, depth_a=depth_a[:, :3], depth_b=depth_b)


def test_camera_depth_axes_and_transform_b_through_the_data_pipeline(skeleton, tmp_path):
    frames, joints = 40, 70
    rng = np.random.default_rng(1)
    face = rng.normal(scale=0.3, size=(frames, joints, 3)).astype(np.float32) + np.array([0, 0, 3.0], dtype=np.float32)
    # The pelvis frame is built from the hips and the acromion midpoint (thorax).
    for name, position in (("left-hip", (-0.1, 0.0, 3.0)), ("right-hip", (0.1, 0.0, 3.0)), ("left-acromion", (-0.2, 0.5, 3.0)), ("right-acromion", (0.2, 0.5, 3.0)), ("left-shoulder", (-0.18, 0.48, 3.0)), ("right-shoulder", (0.18, 0.48, 3.0)), ("neck", (0.0, 0.6, 3.0))):
        face[:, MHR70_INDEX[name]] = position
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    side = (face - np.array([0, 0, 3.0], dtype=np.float32)) @ rotation + np.array([0, 0, 3.0], dtype=np.float32)
    valid = np.ones((frames, joints), dtype=bool)
    trial = PosePairTrial(
        face=face,
        side=side,
        valid_face=valid,
        valid_side=valid.copy(),
        timestamps=np.arange(frames, dtype=np.float64) / 30.0,
        face_map=np.arange(frames, dtype=np.int32),
        side_map=np.arange(frames, dtype=np.int32),
        joint_names=tuple(MHR70_NAMES),
        person_id="p1",
        trial_id="t0",
        fps=30.0,
        source_metadata={},
    )
    sample = sample_from_pose_pair_trial(trial, skeleton, dataset="gymnastics", cycle_bounds=((0, 16), (16, 32)), cycle_mids=(6, 22))
    assert sample.transform_a is not None and sample.transform_b is not None
    depth_a, depth_b = sample.depth_a, sample.depth_b
    assert depth_a.shape == depth_b.shape == (frames, 3)
    np.testing.assert_allclose(np.linalg.norm(depth_a, axis=-1), 1.0, atol=1e-5)
    # The face camera looks along the body's z axis, the side camera along its x axis.
    np.testing.assert_allclose(np.abs(depth_a[:, 2]), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.abs(depth_b[:, 0]), 1.0, atol=1e-4)
    # Depth axes are exactly row 2 of the canonical rotations; unobserved frames give zero.
    np.testing.assert_allclose(depth_a, sample.transform_a.rotation[:, 2, :], atol=1e-6)
    broken = CanonicalTransformRecord(rotation=sample.transform_b.rotation, origin=sample.transform_b.origin, scale=sample.transform_b.scale, valid=np.zeros(frames, dtype=bool))
    assert not camera_depth_axes(broken, frames).any() and not camera_depth_axes(None, frames).any()
    # Phase normalisation resamples both transforms.
    normalised = normalize_sample_to_phase(sample, 8)
    assert normalised.transform_b is not None and normalised.depth_b.shape == (normalised.num_frames, 3)
    np.testing.assert_allclose(np.abs(normalised.depth_b[:, 0]), 1.0, atol=1e-4)
    # The cache round-trips transform_b.
    save_samples(tmp_path / "cache", [sample], config={})
    loaded = load_samples(tmp_path / "cache")[0]
    np.testing.assert_allclose(loaded.transform_b.rotation, sample.transform_b.rotation)
    np.testing.assert_allclose(loaded.depth_b, depth_b)
    # Windows ship depth axes (zero on padding); the view consensus uses the same rule.
    dataset = CycleWindowDataset([sample], skeleton=skeleton, window=WindowConfig(num_cycles=4, samples_per_cycle=8), split="train")
    item = dataset[0]
    assert item["depth_a"].shape == item["depth_b"].shape == (32, 3)
    assert item["frame_mask"].sum() == 16 and not item["depth_a"][16:].any() and item["depth_a"][:16].norm(dim=-1).allclose(torch.ones(16))


def test_recovery_target_and_view_consensus_use_the_base_rule():
    pose_a, pose_b, valid, half, depth_a, depth_b = _views()
    target, target_valid = pseudo_target(pose_a, pose_b, valid, valid, consensus_distance=10.0, depth_a=depth_a, depth_b=depth_b, depth_alpha=0.8)
    expected, _ = depth_aware_pose_fusion(pose_a, pose_b, half, half, valid, valid, depth_a, depth_b, alpha=0.8)
    torch.testing.assert_close(target, expected)
    assert target_valid.all()
    plain, _ = pseudo_target(pose_a, pose_b, valid, valid, consensus_distance=10.0)
    torch.testing.assert_close(plain, 0.5 * (pose_a + pose_b))
    config = ViewConsensusConfig(disagreement_threshold=10.0, method="depth_aware", depth_alpha=0.8)
    fused, fused_valid, _ = view_consensus(pose_a[0].numpy(), pose_b[0].numpy(), valid[0].numpy(), valid[0].numpy(), config, depth_a[0].numpy(), depth_b[0].numpy())
    np.testing.assert_allclose(fused, expected[0].numpy(), atol=1e-6)
    assert fused_valid.all()
    with pytest.raises(ValueError):
        ViewConsensusConfig(method="depth_aware", depth_alpha=1.0)


def test_synthetic_depth_noise_is_removed_by_the_rule_and_logged_by_the_module():
    cfg = compose_config(["experiment=smoke", "data.options.depth_noise_ratio=4.0", "data.options.noise=0.02", "corruption=none"])
    dm = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))
    dm.setup("test")
    batch = next(iter(dm.test_dataloader()))
    assert batch["depth_a"].shape[-1] == 3 and bool(batch["depth_a"].any()) and bool(batch["depth_b"].any())
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("name")
    module = CycleAwareFusionModule(model_config=model_config, loss_config=OmegaConf.to_container(cfg.loss, resolve=True))
    with torch.no_grad():
        output = module(batch)
        metrics = module._diagnostics(output, batch)
    assert "ta_mpjpe_rule" in metrics and "pa_mpjpe_rule" in metrics and "pa_mpjpe_base" in metrics
    usable = batch["reference_valid"] & output.valid & batch["frame_mask"][..., None]
    average = 0.5 * (batch["pose_a"] + batch["pose_b"])
    average_error, mask = per_joint_error(average, batch["reference"], usable & batch["valid_a"] & batch["valid_b"], align="translation")
    assert metrics["ta_mpjpe_rule"] < average_error[mask].mean() * 0.9
