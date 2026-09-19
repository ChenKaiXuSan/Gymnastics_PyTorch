from __future__ import annotations

import pytest
import torch

from gymnastics.fusion.cycle_aware.losses import (
    LossConfig,
    compute_losses,
    periodicity_loss,
    pseudo_target,
    symmetry_loss,
)
from gymnastics.fusion.cycle_aware.metrics import mean_per_joint_position_error, procrustes_align
from gymnastics.fusion.cycle_aware.model import CycleAwareFusionModel


def test_pseudo_target_rules():
    a = torch.zeros(1, 1, 3, 3)
    b = torch.zeros(1, 1, 3, 3)
    b[0, 0, 0] = torch.tensor([0.1, 0.0, 0.0])  # consensus
    b[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])  # disagreement
    valid_a = torch.tensor([[[True, True, False]]])
    valid_b = torch.tensor([[[True, True, True]]])
    target, valid = pseudo_target(a, b, valid_a, valid_b, consensus_distance=0.15)
    assert valid.tolist() == [[[True, False, True]]]
    torch.testing.assert_close(target[0, 0, 0], torch.tensor([0.05, 0.0, 0.0]))
    torch.testing.assert_close(target[0, 0, 2], b[0, 0, 2])


def test_periodicity_uses_consecutive_cycles_only():
    pose = torch.zeros(1, 8, 1, 3)
    pose[0, 4:, 0, 0] = 1.0  # second cycle differs by 1.0
    valid = torch.ones(1, 8, 1, dtype=torch.bool)
    cycle_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    assert periodicity_loss(pose, valid, cycle_index, 4).item() == pytest.approx(1.0)
    assert periodicity_loss(pose, valid, torch.full((1, 8), -1), 4).item() == 0.0
    assert periodicity_loss(pose, valid, cycle_index, 8).item() == 0.0


def test_symmetry_zero_for_mirror_symmetric_pose(skeleton):
    torch.manual_seed(0)
    pose = torch.randn(1, 2, skeleton.num_joints, 3)
    # Enforce mirror symmetry: right joints are x-reflected left joints and
    # mid-line joints (nose, neck) sit on the x = 0 plane.
    paired = {index for pair in skeleton.left_right_pairs for index in pair}
    for index in range(skeleton.num_joints):
        if index not in paired:
            pose[:, :, index, 0] = 0.0
    for left, right in skeleton.left_right_pairs:
        pose[:, :, right] = pose[:, :, left] * torch.tensor([-1.0, 1.0, 1.0])
    valid = torch.ones(1, 2, skeleton.num_joints, dtype=torch.bool)
    assert symmetry_loss(pose, valid, skeleton).item() == pytest.approx(0.0, abs=1e-6)
    pose[:, :, skeleton.index("left-wrist")] += 1.0
    assert symmetry_loss(pose, valid, skeleton).item() > 0.0


def test_compute_losses_end_to_end(tiny_config, tiny_batch, skeleton):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config)
    batch = dict(tiny_batch)
    batch["clean_a"] = batch["pose_a"].clone()
    batch["clean_b"] = batch["pose_b"].clone()
    batch["clean_valid_a"] = batch["valid_a"].clone()
    batch["clean_valid_b"] = batch["valid_b"].clone()
    batch["cycle_index"] = torch.arange(16)[None].repeat(2, 1) // 8
    output = model(**tiny_batch)
    losses = compute_losses(output, batch, skeleton=skeleton, config=LossConfig(), samples_per_cycle=8)
    for name, value in losses.as_dict().items():
        assert value.ndim == 0 and torch.isfinite(value), name
    assert losses.residual.item() == 0.0  # zero-initialised residual
    losses.total.backward()
    with pytest.raises(ValueError):
        LossConfig(recovery_kind="huber")
    assert LossConfig.from_mapping({"symmetry_weight": 0.5}).symmetry_weight == 0.5


def test_procrustes_alignment_recovers_similarity_transform():
    torch.manual_seed(0)
    reference = torch.randn(3, 10, 3)
    angle = torch.tensor(0.7)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    prediction = 2.5 * reference @ rotation.T + torch.tensor([1.0, -2.0, 3.0])
    valid = torch.ones(3, 10, dtype=torch.bool)
    aligned = procrustes_align(prediction, reference, valid)
    torch.testing.assert_close(aligned, reference, atol=1e-4, rtol=1e-4)
    error = mean_per_joint_position_error(prediction[None], reference[None], valid[None])
    assert error.item() < 1e-4
    assert mean_per_joint_position_error(prediction[None], reference[None], valid[None], align="none").item() > 1.0
