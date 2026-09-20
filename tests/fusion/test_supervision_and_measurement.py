"""Reference-supervised recovery, half-cycle symmetry, measurement metrics, diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fusion.data.synthetic import SyntheticDataModule
from fusion.lightning_module import CycleAwareFusionModule, EvaluationConfig
from fusion.losses import LossConfig, compute_losses, half_symmetry_loss, reference_target
from fusion.measurement import retention_ratios, trunk_twist
from fusion.model import CycleAwareFusionModel
from fusion.train import compose_config, run

TINY_MODEL = {"hidden_dim": 16, "num_heads": 2, "samples_per_cycle": 8, "spatial": {"layers": 1}, "short_motion": {"layers": 1}, "long_motion": {"layers": 1}}
TINY_DATA = {"name": "synthetic", "batch_size": 4, "window": {"num_cycles": 2, "samples_per_cycle": 8}, "options": {"subjects": 4, "sequences_per_subject": 1, "frames": 48, "period": 12}}


def test_reference_target_recovers_anchor_frame():
    torch.manual_seed(0)
    anchor = torch.randn(2, 3, 12, 3)
    angle = torch.tensor(0.4)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    reference = 100.0 * anchor @ rotation.T + torch.tensor([5.0, 6.0, 7.0])  # world frame, other units
    valid = torch.ones(2, 3, 12, dtype=torch.bool)
    valid[0, 0, :10] = False  # frame with only two usable joints -> no target
    target, target_valid = reference_target(reference, valid, anchor, torch.ones_like(valid))
    assert not target_valid[0, 0].any() and target_valid[1].all()
    torch.testing.assert_close(target[1], anchor[1], atol=1e-4, rtol=1e-4)


def test_compute_losses_uses_reference_when_requested(tiny_config, tiny_batch, skeleton):
    torch.manual_seed(0)
    model = CycleAwareFusionModel(tiny_config)
    output = model(**tiny_batch)
    batch = dict(tiny_batch)
    batch["clean_a"], batch["clean_b"] = batch["pose_a"], batch["pose_b"]
    batch["clean_valid_a"], batch["clean_valid_b"] = batch["valid_a"], batch["valid_b"]
    batch["cycle_index"] = torch.arange(16)[None].repeat(2, 1) // 8
    batch["reference"] = batch["pose_a"] + 0.3  # a reference that disagrees with the inputs
    batch["reference_valid"] = torch.ones_like(batch["valid_a"])
    pseudo = compute_losses(output, batch, skeleton=skeleton, config=LossConfig(recovery_target="pseudo"), samples_per_cycle=8)
    supervised = compute_losses(output, batch, skeleton=skeleton, config=LossConfig(recovery_target="reference"), samples_per_cycle=8)
    assert supervised.recovery.item() > pseudo.recovery.item()  # translation is removed, but the +0.3 shift changes nothing... shape differs by masking
    supervised.total.backward()
    with pytest.raises(ValueError):
        LossConfig(recovery_target="gt")


def test_half_symmetry_loss_zero_for_time_reversed_cycle():
    S, J = 8, 3
    pose = torch.zeros(1, 16, J, 3)
    # Outward half rises 0..3, return half mirrors it: sample k equals sample S-k.
    values = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    pose[0, :8, :, 0] = values[:, None]
    pose[0, 8:, :, 0] = values[:, None]
    valid = torch.ones(1, 16, J, dtype=torch.bool)
    phase = (torch.arange(16) % S / S)[None].float()
    phase_valid = torch.ones(1, 16, dtype=torch.bool)
    cycle_index = (torch.arange(16) // S)[None]
    half_index = ((torch.arange(16) % S) >= S // 2).long()[None]
    assert half_symmetry_loss(pose, valid, phase, phase_valid, cycle_index, half_index, S).item() == pytest.approx(0.0)
    pose[0, 1, :, 0] += 1.0  # break the mirror
    assert half_symmetry_loss(pose, valid, phase, phase_valid, cycle_index, half_index, S).item() > 0.0
    # No middles known -> no pairs -> zero.
    assert half_symmetry_loss(pose, valid, phase, phase_valid, cycle_index, torch.full_like(half_index, -1), S).item() == 0.0


def _twisting_pose(skeleton, frames: int, amplitude: float):
    J = skeleton.num_joints
    pose = torch.zeros(1, frames, J, 3)
    angle = amplitude * torch.sin(2 * torch.pi * torch.arange(frames) / frames)
    pose[0, :, skeleton.left_hip_index] = torch.tensor([-0.1, 0.0, 0.0])
    pose[0, :, skeleton.right_hip_index] = torch.tensor([0.1, 0.0, 0.0])
    pose[0, :, skeleton.index("left-shoulder")] = torch.stack([-0.2 * torch.cos(angle), torch.full((frames,), 0.5), -0.2 * torch.sin(angle)], -1)
    pose[0, :, skeleton.index("right-shoulder")] = torch.stack([0.2 * torch.cos(angle), torch.full((frames,), 0.5), 0.2 * torch.sin(angle)], -1)
    return pose, torch.ones(1, frames, J, dtype=torch.bool)


def test_trunk_twist_is_frame_invariant_and_retention_detects_shrinkage(skeleton):
    pose, valid = _twisting_pose(skeleton, 32, 0.6)
    theta, ok = trunk_twist(pose, valid, skeleton)
    assert ok.all() and abs(theta.max().item() - 0.6) < 1e-3 and abs(theta.min().item() + 0.6) < 1e-3
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    moved, _ = trunk_twist(pose @ q.T + 2.0, valid, skeleton)
    torch.testing.assert_close(theta, moved, atol=1e-4, rtol=0)
    cycle_index = torch.zeros(1, 32, dtype=torch.long)
    dt = torch.full((1, 32), 1.0 / 30.0)
    shrunk, _ = _twisting_pose(skeleton, 32, 0.3)
    ratios = retention_ratios(shrunk, valid, pose, valid, pose, valid, cycle_index, dt, skeleton, reference=pose, reference_valid=valid)
    assert ratios["rom_retention"].item() == pytest.approx(0.5, abs=1e-3)
    assert ratios["peak_omega_retention"].item() == pytest.approx(0.5, abs=1e-2)
    assert ratios["rom_retention_vs_reference"].item() == pytest.approx(0.5, abs=1e-3)


def test_diagnostics_report_views_failures_and_measurement(skeleton):
    torch.manual_seed(0)
    module = CycleAwareFusionModule(TINY_MODEL, evaluation_config=EvaluationConfig(failure_threshold=0.05))
    frames = 16
    pose, valid = _twisting_pose(skeleton, frames, 0.5)
    batch = {
        "pose_a": pose.clone(), "pose_b": pose.clone(), "valid_a": valid.clone(), "valid_b": valid.clone(),
        "frame_mask": torch.ones(1, frames, dtype=torch.bool), "delta_t": torch.full((1, frames), 1.0 / 30.0),
        "phase": (torch.arange(frames) % 8 / 8.0)[None].float(), "phase_valid": torch.ones(1, frames, dtype=torch.bool),
        "cycle_index": (torch.arange(frames) // 8)[None], "half_index": ((torch.arange(frames) % 8) >= 4).long()[None],
        "reference": pose.clone(), "reference_valid": valid.clone(), "reference_canonical": torch.tensor([True]),
    }
    # View B fails badly on the first half of the frames only.
    batch["pose_b"][0, :8] += 0.5 * torch.randn(8, skeleton.num_joints, 3)
    with torch.no_grad():
        metrics = module._diagnostics(module(batch), batch)
    assert {"pa_mpjpe", "pa_mpjpe_base", "pa_mpjpe_face", "pa_mpjpe_side", "svf_fraction", "both_fail_fraction", "rom_retention", "rom_retention_base", "peak_omega_retention"} <= set(metrics)
    assert metrics["pa_mpjpe_side"] > metrics["pa_mpjpe_face"]
    assert 0.0 < metrics["svf_fraction"].item() <= 0.5 and metrics["both_fail_fraction"].item() == 0.0
    assert metrics["pa_mpjpe_svf_oracle"] <= metrics["pa_mpjpe_svf_side"]
    assert "ta_mpjpe" in metrics  # canonical reference


def test_train_with_reference_and_gymnastics_refusal(tmp_path: Path):
    datamodule = SyntheticDataModule({**TINY_DATA, "train_with_reference": True})
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["reference_valid"].any()  # kept for supervision
    plain = SyntheticDataModule(TINY_DATA)
    plain.setup("fit")
    assert not next(iter(plain.train_dataloader()))["reference_valid"].any()
    from fusion.data.gymnastics import GymnasticsDataModule

    with pytest.raises(ValueError, match="must not supervise"):
        GymnasticsDataModule({"name": "gymnastics", "train_with_reference": True, "options": {"persons": ["1"]}}, trial_loader=lambda p: [], reference_loader=lambda p, c: None).setup("fit")


def test_test_with_corruption_logs_corrupted_reference_error(tmp_path: Path):
    cfg = compose_config(["experiment=smoke", f"output_root={tmp_path}", "data.test_with_corruption=true", "run_name=robust"])
    result = run(cfg)
    assert "test/pa_mpjpe_corrupted" in result["test_metrics"] and "test/pa_mpjpe_corrupted_base" in result["test_metrics"]


def test_checkpoint_transfer_test_only_and_finetune(tmp_path: Path):
    cfg = compose_config(["experiment=smoke", f"output_root={tmp_path}", "run_name=source"])
    source = run(cfg)
    ckpt = Path(source["run_dir"]) / "checkpoints" / "last.ckpt"
    evaluated = run(compose_config(["experiment=smoke", f"output_root={tmp_path}", "run_name=transfer", f"checkpoint={ckpt}", "test_only=true", "seed=3"]))
    assert evaluated["test_only"] and "fit_metrics" not in evaluated and "test/pa_mpjpe" in evaluated["test_metrics"]
    assert (Path(evaluated["run_dir"]) / "result.json").is_file()
    finetuned = run(compose_config(["experiment=smoke", f"output_root={tmp_path}", "run_name=finetune", f"checkpoint={ckpt}"]))
    assert "fit_metrics" in finetuned and "test/pa_mpjpe" in finetuned["test_metrics"]
    with pytest.raises(ValueError):
        run(compose_config(["experiment=smoke", f"output_root={tmp_path}", "run_name=bad", "test_only=true"]))


def test_reference_supervised_smoke_run(tmp_path: Path):
    cfg = compose_config(["experiment=smoke", f"output_root={tmp_path}", "loss.recovery_target=reference", "data.train_with_reference=true", "loss.half_symmetry_weight=0.2", "run_name=supervised"])
    result = run(cfg)
    assert "train/half_symmetry_epoch" in result["fit_metrics"] or "train/half_symmetry" in result["fit_metrics"]
    assert result["test_metrics"]["test/pa_mpjpe"] > 0
