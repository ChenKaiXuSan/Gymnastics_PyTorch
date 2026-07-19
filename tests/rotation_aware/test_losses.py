from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fuse.rotation_aware.config import RoleSpec, SkeletonSpec
from fuse.rotation_aware.losses import LossConfig, compute_self_supervised_losses
from fuse.rotation_aware.model import FusionOutput
from fuse.rotation_aware.trunk import extract_trunk_features


def _spec() -> SkeletonSpec:
    names = ("left-hip", "right-hip", "left-acromion", "right-acromion", "neck", "left-wrist", "right-wrist")
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
    return SkeletonSpec("tiny", names, ((0, 1), (2, 3), (0, 2), (1, 3)), roles, tuple(roles))


def _batch_and_output(*, frames: int = 5) -> tuple[dict[str, torch.Tensor], FusionOutput, SkeletonSpec]:
    spec = _spec()
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
    reference = pose[None, None].repeat(1, frames, 1, 1)
    reference[:, :, 4, 0] = torch.linspace(0.0, 0.2, frames)
    valid = torch.ones(reference.shape[:-1], dtype=torch.bool)
    trunk = extract_trunk_features(reference, valid, spec, dt=1.0)
    output = FusionOutput(
        fused_kpts=reference.clone(),
        base_kpts=reference.clone(),
        delta_kpts=torch.zeros_like(reference),
        valid=valid.clone(),
        fused_theta=trunk.angle.clone(),
        fused_theta_valid=trunk.angle_valid.clone(),
        fused_r_pt=trunk.rotation.clone(),
        fused_r_pt_valid=trunk.rotation_valid.clone(),
    )
    batch = {
        "reference_face": reference.clone(),
        "reference_side": reference.clone(),
        "valid_face": valid.clone(),
        "valid_side": valid.clone(),
        "loss_mask": valid.clone(),
        "padding_mask": torch.ones((1, frames), dtype=torch.bool),
        "face_corruption_mask": torch.ones_like(valid),
        "side_corruption_mask": torch.zeros_like(valid),
        "quality_face": torch.ones((1, frames)),
        "quality_side": torch.ones((1, frames)),
        "complete_cycle": torch.ones(1, dtype=torch.bool),
    }
    return batch, output, spec


def test_perfect_prediction_has_zero_for_each_named_masked_loss() -> None:
    batch, output, spec = _batch_and_output()

    losses = compute_self_supervised_losses(output, batch, LossConfig(), spec)

    for name in (
        "corruption_recovery",
        "high_consensus_identity",
        "circular_axial_rotation",
        "so3_rotation",
        "trial_bone_length",
        "local_rigidity",
        "adaptive_temporal_acceleration",
        "minimal_residual",
        "complete_cycle_rom",
        "total",
    ):
        assert getattr(losses, name).item() == pytest.approx(0.0, abs=1e-7)


def test_padding_invalid_and_nonfinite_values_cannot_affect_total() -> None:
    batch, output, spec = _batch_and_output()
    batch["loss_mask"][:, -1] = False
    batch["valid_face"][:, -1] = False
    batch["valid_side"][:, -1] = False
    batch["padding_mask"][:, -1] = False
    baseline = compute_self_supervised_losses(output, batch, LossConfig(), spec)
    corrupted = replace(
        output,
        fused_kpts=output.fused_kpts.clone(),
        delta_kpts=output.delta_kpts.clone(),
        fused_theta=output.fused_theta.clone(),
        fused_r_pt=output.fused_r_pt.clone(),
    )
    corrupted.fused_kpts[:, -1] = float("nan")
    corrupted.delta_kpts[:, -1] = 1e6
    corrupted.fused_theta[:, -1] = 1e6
    corrupted.fused_r_pt[:, -1] = 1e6

    masked = compute_self_supervised_losses(corrupted, batch, LossConfig(), spec)

    torch.testing.assert_close(baseline.total, masked.total, atol=0, rtol=0)
    assert torch.isfinite(masked.total)


def test_corruption_and_identity_targets_obey_quality_and_consensus_boundaries() -> None:
    batch, output, spec = _batch_and_output()
    batch["reference_side"] += 10.0
    batch["quality_face"][:] = 1.0
    batch["quality_side"][:] = 0.0
    batch["face_corruption_mask"][:] = True
    batch["side_corruption_mask"][:] = False

    losses = compute_self_supervised_losses(output, batch, LossConfig(), spec)

    assert losses.corruption_recovery.item() == pytest.approx(0.0, abs=1e-7)
    assert losses.high_consensus_identity.item() == pytest.approx(0.0, abs=1e-7)


def test_zero_quality_high_consensus_uses_the_reference_average() -> None:
    batch, output, spec = _batch_and_output()
    batch["reference_side"] += 0.02
    batch["quality_face"][:] = 0.0
    batch["quality_side"][:] = 0.0
    output = replace(output, fused_kpts=(batch["reference_face"] + batch["reference_side"]) / 2.0)

    losses = compute_self_supervised_losses(output, batch, LossConfig(), spec)

    assert losses.corruption_recovery.item() == pytest.approx(0.0, abs=1e-7)


def test_complete_cycle_rom_excludes_incomplete_windows() -> None:
    batch, output, spec = _batch_and_output()
    shifted = replace(output, fused_theta=output.fused_theta + torch.linspace(0.0, 1.0, output.fused_theta.shape[1]))
    batch["complete_cycle"][:] = False

    losses = compute_self_supervised_losses(shifted, batch, LossConfig(), spec)

    assert losses.complete_cycle_rom.item() == pytest.approx(0.0, abs=1e-7)
