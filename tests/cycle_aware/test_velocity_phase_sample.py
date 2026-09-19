from __future__ import annotations

import numpy as np
import pytest
import torch

from gymnastics.fusion.cycle_aware.phase import (
    PhaseEncoding,
    normalize_sample_to_phase,
    phase_from_cycle_bounds,
)
from gymnastics.fusion.cycle_aware.sample import DualViewSample, collate_fusion_batch
from gymnastics.fusion.cycle_aware.velocity import compute_velocity
from tests.cycle_aware.conftest import make_sample


def test_velocity_uses_physical_intervals():
    pose = torch.zeros(1, 4, 2, 3)
    pose[0, :, 0, 0] = torch.tensor([0.0, 1.0, 3.0, 6.0])
    delta_t = torch.tensor([[0.5, 0.5, 1.0, 1.5]])
    velocity, valid = compute_velocity(pose, delta_t)
    assert velocity.shape == pose.shape
    torch.testing.assert_close(velocity[0, 1:, 0, 0], torch.tensor([2.0, 2.0, 2.0]))
    torch.testing.assert_close(velocity[0, 0], velocity[0, 1])  # replicate boundary
    assert valid.all()
    velocity_zero, _ = compute_velocity(pose, delta_t, boundary="zero")
    assert torch.equal(velocity_zero[0, 0], torch.zeros(2, 3))


def test_velocity_masks_invalid_pairs_and_validates_shapes():
    pose = torch.randn(2, 5, 3, 3)
    valid = torch.ones(2, 5, 3, dtype=torch.bool)
    valid[0, 2, 1] = False
    velocity, velocity_valid = compute_velocity(pose, torch.full((2, 5), 0.1), valid)
    assert not velocity_valid[0, 2, 1] and not velocity_valid[0, 3, 1]
    assert torch.equal(velocity[0, 2, 1], torch.zeros(3))
    with pytest.raises(ValueError):
        compute_velocity(pose[..., :2], torch.full((2, 5), 0.1))
    with pytest.raises(ValueError):
        compute_velocity(pose, torch.full((2, 3), 0.1))
    with pytest.raises(ValueError):
        compute_velocity(pose, torch.full((2, 5), 0.1), boundary="mirror")


def test_phase_from_cycle_bounds():
    phase, valid, index, half = phase_from_cycle_bounds(10, [(2, 6), (6, 10)])
    assert not valid[:2].any() and valid[2:].all()
    np.testing.assert_allclose(phase[2:6], [0.0, 0.25, 0.5, 0.75])
    assert index.tolist() == [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    assert (half == -1).all()
    _, _, _, half = phase_from_cycle_bounds(10, [(2, 6), (6, 10)], cycle_mids=[3, 9])
    assert half.tolist() == [-1, -1, 0, 1, 1, 1, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        phase_from_cycle_bounds(10, [(2, 6)], cycle_mids=[6])
    with pytest.raises(ValueError):
        phase_from_cycle_bounds(10, [(2, 6), (6, 10)], cycle_mids=[3])


def test_phase_encoding_is_periodic_and_masked():
    encoder = PhaseEncoding(harmonics=2)
    phase = torch.tensor([[0.0, 0.5, 0.999999]])
    valid = torch.tensor([[True, True, False]])
    out = encoder(phase, valid)
    assert out.shape == (1, 3, 4)
    torch.testing.assert_close(out[0, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert torch.equal(out[0, 2], torch.zeros(4))


def test_normalize_sample_to_phase_keeps_physical_time(skeleton):
    sample = make_sample(skeleton, frames=48, period=16, fps=30.0)
    normalized = normalize_sample_to_phase(sample, samples_per_cycle=8)
    assert normalized.num_frames == 3 * 8
    assert normalized.cycle_bounds == ((0, 8), (8, 16), (16, 24))
    # Two samples per cycle step correspond to 16/8 = 2 frames = 2/30 s.
    np.testing.assert_allclose(np.diff(normalized.timestamps), 2.0 / 30.0, rtol=1e-6)
    # Interpolated values sit between the bracketing frames.
    assert np.isfinite(normalized.view_a).all()
    assert not normalized.valid_b[:, -1].any()  # missing joint stays missing
    unchanged = normalize_sample_to_phase(make_sample(skeleton, cycles=False), 8)
    assert unchanged.cycle_bounds == ()


def test_normalize_with_middles_puts_mid_at_half_cycle(skeleton):
    sample = make_sample(skeleton, frames=48, period=16, fps=30.0, mids=True)
    assert sample.has_cycle_mids and sample.cycle_mids == (4, 20, 36)
    normalized = normalize_sample_to_phase(sample, samples_per_cycle=8)
    assert normalized.cycle_bounds == ((0, 8), (8, 16), (16, 24))
    assert normalized.cycle_mids == (4, 12, 20)
    # Outward half covers 4 frames in 4 samples (dt = 1/30), the return half
    # covers 12 frames in 4 samples (dt = 3/30); the middle keeps its time.
    np.testing.assert_allclose(normalized.timestamps[4], sample.timestamps[4])
    np.testing.assert_allclose(np.diff(normalized.timestamps[:4]), 1.0 / 30.0)
    np.testing.assert_allclose(np.diff(normalized.timestamps[4:8]), 3.0 / 30.0)
    with pytest.raises(ValueError):
        normalize_sample_to_phase(sample, samples_per_cycle=7)


def test_dual_view_sample_validation(skeleton):
    sample = make_sample(skeleton)
    assert sample.has_cycles and sample.key == "synthetic/s1/seq0"
    with pytest.raises(ValueError):
        DualViewSample(
            dataset="x",
            subject_id="s",
            sequence_id="q",
            view_a=sample.view_a,
            view_b=sample.view_b,
            valid_a=sample.valid_a,
            valid_b=sample.valid_b,
            timestamps=sample.timestamps,
            joint_names=sample.joint_names,
            cycle_bounds=((0, 20), (10, 30)),
        )
    with pytest.raises(ValueError):
        DualViewSample(
            dataset="x",
            subject_id="s",
            sequence_id="q",
            view_a=sample.view_a,
            view_b=sample.view_b,
            valid_a=sample.valid_a,
            valid_b=sample.valid_b,
            timestamps=sample.timestamps[::-1],
            joint_names=sample.joint_names,
        )
    with pytest.raises(ValueError):
        DualViewSample(
            dataset="x",
            subject_id="s",
            sequence_id="q",
            view_a=sample.view_a,
            view_b=sample.view_b,
            valid_a=sample.valid_a,
            valid_b=sample.valid_b,
            timestamps=sample.timestamps,
            joint_names=sample.joint_names,
            cycle_bounds=((0, 16),),
            cycle_mids=(16,),
        )


def test_collate_stacks_tensors_and_lists_strings():
    windows = [{"pose_a": torch.zeros(2, 3), "window_id": "a"}, {"pose_a": torch.ones(2, 3), "window_id": "b"}]
    batch = collate_fusion_batch(windows)
    assert batch["pose_a"].shape == (2, 2, 3)
    assert batch["window_id"] == ["a", "b"]
    with pytest.raises(ValueError):
        collate_fusion_batch([])
