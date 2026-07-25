from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fuse.rotation_aware.config import RoleSpec, SkeletonSpec
from fuse.rotation_aware import losses as losses_module
from fuse.rotation_aware.features import extract_pose_features
from fuse.rotation_aware.losses import (
    LossConfig,
    _complete_cycle_mask,
    _rom_loss,
    _rom_peak_loss,
    compute_self_supervised_losses,
)
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
    pose_features = extract_pose_features(reference, valid, spec, dt=1.0)
    baseline = torch.stack(
        [pose_features.bone_lengths[0, :, index][pose_features.bone_valid[0, :, index]].median() for index in range(len(spec.bones))]
    )[None]
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
        "dt": torch.ones((1, frames)),
        "trial_bone_baseline": baseline,
        "trial_bone_baseline_valid": torch.ones_like(baseline, dtype=torch.bool),
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


def test_complete_cycle_rom_helper_excludes_padded_suffix() -> None:
    batch, output, spec = _batch_and_output()
    batch["padding_mask"][:, -1] = False
    batch["loss_mask"][:, -1] = False
    baseline = losses_module.compute_complete_cycle_rom_loss(
        output, batch, LossConfig(), spec
    )
    shifted = replace(output, fused_kpts=output.fused_kpts.clone())
    shifted.fused_kpts[:, -1] = 1e6

    masked = losses_module.compute_complete_cycle_rom_loss(
        shifted, batch, LossConfig(), spec
    )

    torch.testing.assert_close(masked, baseline, atol=0, rtol=0)


def test_so3_exact_agreement_has_finite_backward_gradient() -> None:
    batch, output, spec = _batch_and_output()
    fused = output.fused_kpts.detach().clone().requires_grad_()
    output = replace(output, fused_kpts=fused)

    losses = compute_self_supervised_losses(output, batch, LossConfig(), spec)
    losses.so3_rotation.backward()

    assert fused.grad is not None
    assert torch.isfinite(fused.grad).all()


def test_twist_overshoot_is_zero_within_envelope_and_positive_beyond_it() -> None:
    # 改法3 is a one-sided bound. Within the wider per-view twist rate it must be
    # exactly zero (so it never suppresses the twist / fights 改法4); beyond it, it
    # penalises the excess and carries gradient.
    from fuse.rotation_aware.losses import _twist_overshoot_loss

    valid = torch.ones((1, 4), dtype=torch.bool)
    omega_face = torch.tensor([[0.0, 0.3, -0.4, 0.5]])
    omega_side = torch.tensor([[0.0, 0.2, -0.6, 0.1]])  # envelope = max|.| = [0,.3,.6,.5]

    within = torch.tensor([[0.0, 0.25, -0.5, 0.4]])  # everywhere <= envelope
    loss_within = _twist_overshoot_loss(within, valid, omega_face, valid, omega_side, valid)
    assert loss_within.item() == pytest.approx(0.0, abs=1e-7)

    over = torch.tensor([[0.0, 0.3, -1.0, 0.5]], requires_grad=True)  # frame2 |1.0|>0.6
    loss_over = _twist_overshoot_loss(over, valid, omega_face, valid, omega_side, valid)
    # only the excess (1.0-0.6)=0.4 on one of four frames contributes
    assert loss_over.item() == pytest.approx((0.4**2) / 4, abs=1e-6)
    loss_over.backward()
    assert over.grad is not None and torch.isfinite(over.grad).all() and over.grad.abs().sum() > 0


def test_rom_peak_anchors_to_the_wider_view_not_the_average() -> None:
    # face twists 1.0 rad, side 0.6 rad, fused only 0.5 -> the target must be the
    # larger view ROM (1.0), NOT the averaged pseudo-target, so the shrunk fused
    # range is penalised toward the view that actually saw the wider twist.
    fused = torch.tensor([[0.0, 0.25, 0.5]], requires_grad=True)
    face = torch.tensor([[0.0, 0.5, 1.0]])
    side = torch.tensor([[0.0, 0.3, 0.6]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    complete = torch.ones_like(valid)

    loss = _rom_peak_loss(fused, face, side, valid, valid, valid, complete)
    loss.backward()

    assert loss.item() == pytest.approx((0.5 - 1.0) ** 2, abs=1e-6)
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


def test_rom_peak_is_zero_when_fused_matches_the_wider_view() -> None:
    fused = torch.tensor([[0.0, 0.5, 1.0]])
    face = torch.tensor([[0.0, 0.5, 1.0]])
    side = torch.tensor([[0.0, 0.3, 0.6]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    complete = torch.ones_like(valid)

    loss = _rom_peak_loss(fused, face, side, valid, valid, valid, complete)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_rom_peak_skips_a_view_with_a_gap_in_the_run() -> None:
    # face has a mid-run gap, so only side (ROM 0.6) may anchor the target.
    fused = torch.tensor([[0.0, 0.25, 0.5]])
    face = torch.tensor([[0.0, 0.5, 1.0]])
    side = torch.tensor([[0.0, 0.3, 0.6]])
    fused_valid = torch.ones((1, 3), dtype=torch.bool)
    face_valid = torch.tensor([[True, False, True]])
    side_valid = torch.ones((1, 3), dtype=torch.bool)
    complete = torch.ones_like(fused_valid)

    loss = _rom_peak_loss(fused, face, side, fused_valid, face_valid, side_valid, complete)

    assert loss.item() == pytest.approx((0.5 - 0.6) ** 2, abs=1e-6)


def test_rom_unwraps_across_the_pi_boundary() -> None:
    wrapped = torch.tensor([[3.0, -3.0]])
    target = torch.tensor([[3.0, 3.1]])
    valid = torch.ones_like(wrapped, dtype=torch.bool)
    complete = torch.ones_like(valid)

    loss = _rom_loss(wrapped, target, valid, complete)

    assert loss.item() < 0.05


def test_rom_never_spans_independently_reset_runs_and_has_finite_backward() -> None:
    prediction = torch.tensor([[3.13, 0.0, 3.14]], requires_grad=True)
    target = torch.tensor([[3.13, 0.0, -3.13]])
    valid = torch.tensor([[True, False, True]])
    complete = torch.ones_like(valid)

    loss = _rom_loss(prediction, target, valid, complete)
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize(
    ("prediction_values", "target_values", "valid", "complete"),
    [
        ([[0.2, -0.3, 0.4]], [[0.1, -0.2, 0.3]], [[False, False, False]], [[True, True, True]]),
        ([[0.2, 0.7, 1.1]], [[0.1, 0.4, 0.9]], [[True, True, True]], [[True, True, True]]),
        ([[3.0, -3.0, 0.0, 3.1, -3.1]], [[3.1, -3.1, 0.0, 3.0, -3.0]], [[True, True, False, True, True]], [[True, True, True, True, True]]),
        ([[0.2, 0.5, 0.7, 1.0]], [[0.1, 0.4, 0.8, 0.9]], [[True, True, False, False]], [[True, True, True, True]]),
        ([[3.0, -3.0, -2.8]], [[3.1, -3.1, -2.9]], [[True, True, True]], [[True, True, True]]),
        (
            [[0.1, 0.4, 0.8, 1.0], [3.0, -3.0, 0.2, 0.6]],
            [[0.0, 0.3, 0.7, 0.9], [3.1, -3.1, 0.1, 0.5]],
            [[True, True, True, True], [True, True, False, True]],
            [[True, True, True, True], [True, True, True, True]],
        ),
    ],
    ids=("no_run", "one_run", "separated_runs", "padding", "wrapped_angles", "two_batch_members"),
)
def test_rom_matches_reference_values_and_gradients(
    prediction_values: list[list[float]],
    target_values: list[list[float]],
    valid: list[list[bool]],
    complete: list[list[bool]],
) -> None:
    prediction = torch.tensor(prediction_values, requires_grad=True)
    prediction_reference = prediction.detach().clone().requires_grad_()
    target = torch.tensor(target_values)
    valid_tensor = torch.tensor(valid)
    complete_tensor = torch.tensor(complete)

    actual = _rom_loss(prediction, target, valid_tensor, complete_tensor)
    expected = losses_module._rom_loss_reference(
        prediction_reference, target, valid_tensor, complete_tensor
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if not actual.requires_grad:
        assert not expected.requires_grad
        assert prediction.grad is None
        assert prediction_reference.grad is None
        return
    actual.backward()
    expected.backward()
    torch.testing.assert_close(
        prediction.grad, prediction_reference.grad, rtol=1e-6, atol=1e-6
    )


def test_rom_multiframe_execution_never_converts_a_tensor_to_bool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.tensor([[3.0, -3.0, -2.8]], requires_grad=True)
    target = torch.tensor([[3.1, -3.1, -2.9]])
    valid = torch.ones_like(prediction, dtype=torch.bool)
    complete = torch.ones_like(valid)

    def fail_on_tensor_bool(_: torch.Tensor) -> bool:
        raise AssertionError("ROM execution must not synchronize through Tensor.__bool__")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "__bool__", fail_on_tensor_bool)
        loss = _rom_loss(prediction, target, valid, complete)

    assert torch.isfinite(loss)


def test_zero_weight_losses_preserve_reporting_and_active_gradients() -> None:
    batch, output, spec = _batch_and_output()
    fused = (output.fused_kpts.detach().clone() + 0.05).requires_grad_()
    output = replace(output, fused_kpts=fused)
    zero_weight_config = LossConfig(
        circular_axial_rotation_weight=0.0,
        so3_rotation_weight=0.0,
        adaptive_temporal_acceleration_weight=0.0,
        complete_cycle_rom_weight=0.0,
    )
    comparison_fused = fused.detach().clone().requires_grad_()
    comparison_output = replace(output, fused_kpts=comparison_fused)

    actual = compute_self_supervised_losses(output, batch, zero_weight_config, spec)
    expected = compute_self_supervised_losses(comparison_output, batch, LossConfig(), spec)
    expected_total = (
        expected.corruption_recovery
        + expected.high_consensus_identity
        + expected.trial_bone_length
        + expected.local_rigidity
        + LossConfig().minimal_residual_weight * expected.minimal_residual
    )

    for name in expected.as_dict():
        if name != "total":
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=1e-7, atol=1e-7
            )
    torch.testing.assert_close(actual.total, expected_total, rtol=1e-7, atol=1e-7)
    actual.total.backward()
    expected_total.backward()
    torch.testing.assert_close(fused.grad, comparison_fused.grad, rtol=1e-7, atol=1e-7)


def test_missing_complete_cycle_disables_rom() -> None:
    batch, output, spec = _batch_and_output()
    batch.pop("complete_cycle")

    mask = _complete_cycle_mask(batch, output.fused_kpts.shape[1], output.fused_kpts.device, batch_size=1)

    assert not mask.any()


def test_trial_bone_length_uses_a_temporal_target_baseline() -> None:
    batch, output, spec = _batch_and_output()
    ramp = torch.linspace(0.0, 0.4, output.fused_kpts.shape[1])
    batch["reference_face"][:, :, 1, 0] += ramp
    batch["reference_side"][:, :, 1, 0] += ramp
    output = replace(output, fused_kpts=batch["reference_face"].clone())

    losses = compute_self_supervised_losses(output, batch, LossConfig(), spec)

    assert losses.trial_bone_length.item() > 0


def test_identity_excludes_quality_dominant_disagreements() -> None:
    batch, output, spec = _batch_and_output()
    batch["reference_side"] += 1.0
    batch["quality_face"][:] = 1.0
    batch["quality_side"][:] = 0.0
    batch["face_corruption_mask"][:] = False
    batch["side_corruption_mask"][:] = False
    output = replace(output, fused_kpts=batch["reference_side"].clone())

    losses = compute_self_supervised_losses(output, batch, LossConfig(), spec)

    assert losses.high_consensus_identity.item() == pytest.approx(0.0, abs=1e-7)


def test_adaptive_acceleration_is_invariant_to_equivalent_sampling_rates() -> None:
    def acceleration_loss(frames: int, interval: float) -> torch.Tensor:
        batch, output, spec = _batch_and_output(frames=frames)
        time = torch.arange(frames, dtype=torch.float32) * interval
        output = replace(output, fused_kpts=output.fused_kpts + (0.2 * time.square())[None, :, None, None])
        batch["dt"] = torch.full((1, frames), interval)
        return compute_self_supervised_losses(output, batch, LossConfig(), spec).adaptive_temporal_acceleration

    coarse = acceleration_loss(5, 0.25)
    fine = acceleration_loss(9, 0.125)

    torch.testing.assert_close(coarse, fine, atol=1e-6, rtol=1e-5)


def test_local_rigidity_compares_physical_bone_rates_across_sampling_rates() -> None:
    def rigidity_loss(frames: int, interval: float) -> torch.Tensor:
        batch, output, spec = _batch_and_output(frames=frames)
        time = torch.arange(frames, dtype=torch.float32) * interval
        batch["reference_face"][:, :, 1, 0] += 0.2 * time
        batch["reference_side"][:, :, 1, 0] += 0.2 * time
        output = replace(output, fused_kpts=output.fused_kpts + (0.1 * time)[None, :, None, None])
        batch["dt"] = torch.full((1, frames), interval)
        return compute_self_supervised_losses(output, batch, LossConfig(), spec).local_rigidity

    coarse = rigidity_loss(5, 0.25)
    fine = rigidity_loss(9, 0.125)

    torch.testing.assert_close(coarse, fine, atol=1e-6, rtol=1e-5)


def test_loss_validates_output_masks_and_residual_shapes() -> None:
    batch, output, spec = _batch_and_output()

    with pytest.raises(ValueError, match="FusionOutput.valid"):
        compute_self_supervised_losses(replace(output, valid=output.valid[:, :-1]), batch, LossConfig(), spec)
    with pytest.raises(ValueError, match="FusionOutput.delta_kpts"):
        compute_self_supervised_losses(replace(output, delta_kpts=output.delta_kpts[..., :2]), batch, LossConfig(), spec)
