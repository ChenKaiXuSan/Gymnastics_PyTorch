from __future__ import annotations

import pytest
import torch

from gymnastics.fusion.cycle_aware.modules import (
    BidirectionalCrossViewAttention,
    FiLMMotionGuidance,
    JointReliabilityHead,
    LongMotionTransformer,
    MaskedTransformerEncoder,
    MotionFusion,
    ResidualRefinement,
    ShortMotionTransformer,
    SpatialTransformer,
    weighted_pose_fusion,
)
from gymnastics.fusion.cycle_aware.modules.temporal_transformer import local_band_mask
from gymnastics.fusion.cycle_aware.modules.transformer import build_blocked_mask

B, T, J, D = 2, 12, 5, 16


def test_blocked_mask_never_blocks_every_key():
    valid = torch.zeros(3, 4, dtype=torch.bool)
    valid[0, 1] = True
    blocked = build_blocked_mask(valid, heads=2, band=None)
    assert blocked.shape == (6, 4, 4)
    assert not blocked.all(dim=-1).any()  # each query keeps a key
    assert not blocked[0, :, 1].any()  # the valid key is attendable by everyone


def test_masked_encoder_is_finite_and_ignores_invalid_tokens():
    torch.manual_seed(0)
    encoder = MaskedTransformerEncoder(D, heads=2, layers=1)
    tokens = torch.randn(B, T, D)
    valid = torch.ones(B, T, dtype=torch.bool)
    valid[0, 3:] = False
    out = encoder(tokens, valid)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0, 3:], torch.zeros(T - 3, D))
    # Changing an invalid token cannot change valid outputs.
    perturbed = tokens.clone()
    perturbed[0, 5] += 10.0
    torch.testing.assert_close(encoder(perturbed, valid)[0, :3], out[0, :3])


def test_spatial_transformer_shapes_and_masking():
    torch.manual_seed(0)
    module = SpatialTransformer(J, D, heads=2, layers=1)
    pose = torch.randn(B, T, J, 3)
    valid = torch.ones(B, T, J, dtype=torch.bool)
    valid[:, :, 0] = False
    out = module(pose, valid)
    assert out.shape == (B, T, J, D)
    assert torch.equal(out[:, :, 0], torch.zeros(B, T, D))
    with pytest.raises(ValueError):
        module(pose[..., :2], valid)


def test_short_transformer_respects_local_band():
    torch.manual_seed(0)
    module = ShortMotionTransformer(J, D, samples_per_cycle=8, cycle_ratio=0.25, heads=2, layers=1)
    assert module.window == 2 and module.half_window == 1
    pose = torch.randn(1, T, J, 3)
    velocity = torch.zeros_like(pose)
    valid = torch.ones(1, T, J, dtype=torch.bool)
    out = module(pose, velocity, valid)
    assert out.shape == (1, T, J, D)
    # A perturbation at t=10 must not affect t=0 (outside the band).
    perturbed = pose.clone()
    perturbed[0, 10] += 5.0
    torch.testing.assert_close(module(perturbed, velocity, valid)[0, 0], out[0, 0])
    assert local_band_mask(4, 1, device=pose.device).tolist() == [
        [True, True, False, False],
        [True, True, True, False],
        [False, True, True, True],
        [False, False, True, True],
    ]


def test_long_transformer_uses_phase_channels():
    torch.manual_seed(0)
    module = LongMotionTransformer(J, D, phase_channels=2, heads=2, layers=1)
    pose = torch.randn(1, T, J, 3)
    velocity = torch.zeros_like(pose)
    valid = torch.ones(1, T, J, dtype=torch.bool)
    phase = torch.randn(1, T, 2)
    out = module(pose, velocity, valid, phase)
    assert out.shape == (1, T, J, D)
    assert not torch.allclose(out, module(pose, velocity, valid, torch.zeros_like(phase)))
    with pytest.raises(ValueError):
        module(pose, velocity, valid, None)


def test_motion_fusion_and_film_identity():
    torch.manual_seed(0)
    fusion = MotionFusion(D)
    short, long = torch.randn(B, T, J, D), torch.randn(B, T, J, D)
    assert fusion(short, long).shape == (B, T, J, D)
    film = FiLMMotionGuidance(D)
    pose_feature = torch.randn(B, T, J, D)
    # Zero-initialised FiLM is the identity: H == F_pose.
    torch.testing.assert_close(film(pose_feature, torch.randn(B, T, J, D)), pose_feature)
    with torch.no_grad():
        film.gamma.weight.fill_(0.1)
        film.beta.bias.fill_(0.5)
    motion = torch.randn(B, T, J, D)
    expected = (1.0 + film.gamma(motion)) * pose_feature + film.beta(motion)
    torch.testing.assert_close(film(pose_feature, motion), expected)
    bounded = FiLMMotionGuidance(D, gamma_bound=0.5)
    with torch.no_grad():
        bounded.gamma.weight.fill_(5.0)
    assert (bounded(pose_feature, motion) - pose_feature).abs().max() < 1.5 * pose_feature.abs().max() + 1.0


def test_cross_view_attention_is_symmetric_and_masked():
    torch.manual_seed(0)
    module = BidirectionalCrossViewAttention(D, heads=2, layers=1)
    a, b = torch.randn(B, T, J, D), torch.randn(B, T, J, D)
    valid_a = torch.ones(B, T, J, dtype=torch.bool)
    valid_b = torch.ones(B, T, J, dtype=torch.bool)
    valid_b[0, :, :] = False  # no valid source for view A at batch 0
    out_a, out_b = module(a, b, valid_a, valid_b)
    swapped_b, swapped_a = module(b, a, valid_b, valid_a)
    torch.testing.assert_close(out_a, swapped_a)
    torch.testing.assert_close(out_b, swapped_b)
    assert torch.equal(out_b[0], torch.zeros(T, J, D))
    assert torch.isfinite(out_a).all()


def test_reliability_weights_sum_to_one_and_respect_validity():
    torch.manual_seed(0)
    head = JointReliabilityHead(D)
    a, b = torch.randn(B, T, J, D), torch.randn(B, T, J, D)
    valid_a = torch.ones(B, T, J, dtype=torch.bool)
    valid_b = torch.ones(B, T, J, dtype=torch.bool)
    valid_a[0, :, 1] = False
    valid_b[1, :, 2] = False
    valid_a[1, 0, 3] = False
    valid_b[1, 0, 3] = False
    logits, w_a, w_b = head(a, b, valid_a, valid_b)
    assert logits.shape == (B, T, J, 2) and w_a.shape == (B, T, J, 1)
    torch.testing.assert_close(w_a + w_b, torch.ones(B, T, J, 1))
    assert torch.allclose(w_a[0, :, 1], torch.zeros(T, 1), atol=1e-6)
    assert torch.allclose(w_b[1, :, 2], torch.zeros(T, 1), atol=1e-6)
    torch.testing.assert_close(w_a[1, 0, 3], torch.tensor([0.5]))
    # Swap equivariance of the shared scorer.
    logits_swapped, w_b2, w_a2 = head(b, a, valid_b, valid_a)
    torch.testing.assert_close(logits_swapped[..., 0], logits[..., 1])
    torch.testing.assert_close(w_a2, w_a)
    disabled = JointReliabilityHead(D, enabled=False)
    _, w_a3, w_b3 = disabled(a, b, torch.ones_like(valid_a), torch.ones_like(valid_b))
    torch.testing.assert_close(w_a3, torch.full((B, T, J, 1), 0.5))


def test_weighted_fusion_anchors_to_inputs():
    pose_a = torch.randn(B, T, J, 3)
    pose_b = torch.randn(B, T, J, 3)
    valid_a = torch.ones(B, T, J, dtype=torch.bool)
    valid_b = torch.ones(B, T, J, dtype=torch.bool)
    valid_b[0, :, 0] = False
    w_a = torch.full((B, T, J, 1), 0.25)
    w_a[0, :, 0] = 1.0
    w_b = 1.0 - w_a
    fused, valid = weighted_pose_fusion(pose_a, pose_b, w_a, w_b, valid_a, valid_b)
    torch.testing.assert_close(fused[1], 0.25 * pose_a[1] + 0.75 * pose_b[1])
    torch.testing.assert_close(fused[0, :, 0], pose_a[0, :, 0])
    assert valid.all()
    with pytest.raises(ValueError):
        weighted_pose_fusion(pose_a, pose_b, w_a[..., 0], w_b, valid_a, valid_b)


def test_residual_is_bounded_zero_initialised_and_maskable():
    torch.manual_seed(0)
    head = ResidualRefinement(D, max_delta=0.1)
    a, b = torch.randn(B, T, J, D), torch.randn(B, T, J, D)
    w_a = torch.full((B, T, J, 1), 0.5)
    base = torch.randn(B, T, J, 3)
    valid = torch.ones(B, T, J, dtype=torch.bool)
    valid[0, :, 4] = False
    delta = head(a, b, w_a, 1 - w_a, base, valid)
    assert torch.equal(delta, torch.zeros_like(delta))  # zero init
    with torch.no_grad():
        head.mlp[-1].weight.normal_(std=5.0)
    delta = head(a, b, w_a, 1 - w_a, base, valid)
    assert delta.abs().max() <= 0.1 + 1e-6
    assert torch.equal(delta[0, :, 4], torch.zeros(T, 3))
    disabled = ResidualRefinement(D, enabled=False)
    assert torch.equal(disabled(a, b, w_a, 1 - w_a, base, valid), torch.zeros_like(base))
