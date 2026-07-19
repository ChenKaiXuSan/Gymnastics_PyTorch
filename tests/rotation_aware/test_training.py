from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fuse.rotation_aware.config import RoleSpec, SkeletonSpec
from fuse.rotation_aware.dataset import collate_pose_pair_windows
from fuse.rotation_aware.losses import LossConfig
from fuse.rotation_aware.model import RotationAwareFusionModel
from fuse.rotation_aware.training import _corrupt_batch, _feature_bundle, _forward_window, _fused_bone_cv, load_checkpoint, save_checkpoint, train_one_epoch, validate


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


def _samples(count: int = 8) -> list[dict[str, object]]:
    pose = torch.tensor(
        [
            [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 2.0, 0.0],
            [0.0, 2.2, 0.0], [-1.5, 1.5, 0.0], [1.5, 1.5, 0.0],
        ]
    )
    sequence = pose[None].repeat(5, 1, 1)
    sequence[:, 4, 0] = torch.linspace(0.0, 0.2, 5)
    valid = torch.ones(sequence.shape[:-1], dtype=torch.bool)
    return [
        {
            "face": sequence.clone(),
            "side": sequence.clone(),
            "valid_face": valid.clone(),
            "valid_side": valid.clone(),
            "padding_mask": torch.ones(5, dtype=torch.bool),
            "loss_mask": valid.clone(),
            "dt": torch.ones(5),
            "complete_cycle": True,
            "window_id": f"window-{index}",
        }
        for index in range(count)
    ]


def _loader() -> DataLoader[dict[str, object]]:
    return DataLoader(_samples(), batch_size=4, shuffle=False, collate_fn=collate_pose_pair_windows)


def test_cpu_tiny_overfit_is_finite_and_reduces_loss() -> None:
    torch.manual_seed(4)
    spec = _spec()
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    config = LossConfig(
        high_consensus_identity_weight=0.0,
        circular_axial_rotation_weight=0.0,
        so3_rotation_weight=0.0,
        trial_bone_length_weight=0.0,
        local_rigidity_weight=0.0,
        adaptive_temporal_acceleration_weight=0.0,
        minimal_residual_weight=0.0,
        complete_cycle_rom_weight=0.0,
    )
    from fuse.rotation_aware.corruptions import CorruptionConfig

    corruption = CorruptionConfig(enabled_families=("spike_noise",), spike_probability=1.0, spike_scale=0.02)
    initial = validate(model, _loader(), spec, loss_config=config, corruption_config=corruption, seed=9)["losses"]["corruption_recovery"]

    for epoch in range(8):
        metrics = train_one_epoch(model, _loader(), optimizer, spec, loss_config=config, corruption_config=corruption, seed=9, epoch=epoch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    final = validate(model, _loader(), spec, loss_config=config, corruption_config=corruption, seed=9)["losses"]["corruption_recovery"]
    assert torch.isfinite(torch.tensor(final))
    assert final < initial


def test_training_combines_temporal_mask_before_feature_extraction() -> None:
    spec = _spec()
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = next(iter(_loader()))
    batch["padding_mask"][:, -1] = False
    batch["face"][:, -1] = 1e6
    batch["side"][:, -1] = -1e6

    metrics = train_one_epoch(model, [batch], optimizer, spec, loss_config=LossConfig(), seed=3)

    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_validation_score_and_checkpoint_metadata_exclude_external_ground_truth(tmp_path: Path) -> None:
    spec = _spec()
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics_a = validate(model, _loader(), spec, loss_config=LossConfig(), seed=17)
    metrics_b = validate(model, _loader(), spec, loss_config=LossConfig(), seed=17)
    assert metrics_a == metrics_b
    assert set(metrics_a["components"]) == {
        "corruption_recovery", "bone_cv", "rotation_consistency", "identity_preservation", "rom_retention"
    }
    assert 0.0 <= metrics_a["score"] <= 1.0

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        loss_config=LossConfig(),
        skeleton=spec,
        provenance={"split_hash": "split", "corruption_manifest_hash": "corrupt", "git_commit": "abc"},
        training_config={"batch_size": 4},
        corruption_config=__import__("fuse.rotation_aware.corruptions", fromlist=["CorruptionConfig"]).CorruptionConfig(),
        score=metrics_a["score"],
    )
    loaded = load_checkpoint(path, model, optimizer)
    assert loaded["score"] == metrics_a["score"]
    assert loaded["provenance"]["split_hash"] == "split"
    assert "model" in loaded and "optimizer" in loaded and "skeleton" in loaded and "loss_config" in loaded
    assert loaded["training_config"] == {"batch_size": 4}
    assert "corruption_config" in loaded


def test_checkpoint_rejects_missing_reproducibility_provenance(tmp_path: Path) -> None:
    spec = _spec()
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    import pytest
    from fuse.rotation_aware.corruptions import CorruptionConfig

    with pytest.raises(ValueError, match="split_hash"):
        save_checkpoint(
            tmp_path / "checkpoint.pt", model, optimizer, loss_config=LossConfig(), skeleton=spec,
            provenance={"git_commit": "abc"}, training_config={"batch_size": 4},
            corruption_config=CorruptionConfig(), score=0.1,
        )


def test_load_checkpoint_rejects_incomplete_reproducibility_provenance(tmp_path: Path) -> None:
    spec = _spec()
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    path = tmp_path / "incomplete.pt"
    torch.save({"model": model.state_dict(), "provenance": {"split_hash": "split"}}, path)

    import pytest

    with pytest.raises(ValueError, match="corruption_manifest_hash"):
        load_checkpoint(path, model)


def test_corruption_is_stable_per_window_when_batch_order_changes() -> None:
    spec = _spec()
    batch = next(iter(_loader()))
    forward = _corrupt_batch(batch, seed=41, skeleton=spec, corruption_config=None)
    reverse_batch = {key: value.flip(0) if isinstance(value, torch.Tensor) else list(reversed(value)) for key, value in batch.items()}
    reverse = _corrupt_batch(reverse_batch, seed=41, skeleton=spec, corruption_config=None)
    forward_by_id = dict(zip(batch["window_id"], forward["face"]))
    reverse_by_id = dict(zip(reverse_batch["window_id"], reverse["face"]))

    for window_id, values in forward_by_id.items():
        torch.testing.assert_close(values, reverse_by_id[window_id])


def test_target_quality_is_measured_on_unmodified_reference_windows() -> None:
    from fuse.rotation_aware.corruptions import CorruptionConfig

    spec = _spec()
    batch = next(iter(_loader()))
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    _, prepared = _forward_window(
        model, batch, spec, seed=3,
        corruption_config=CorruptionConfig(enabled_families=("spike_noise",), spike_probability=1.0, spike_scale=10.0),
        device=torch.device("cpu"),
    )
    reference_valid = batch["valid_face"] & batch["padding_mask"].unsqueeze(-1)
    expected = _feature_bundle(batch["face"], reference_valid, spec, batch["dt"]).quality.loss_weight

    torch.testing.assert_close(prepared["quality_face"], expected)


def test_fused_bone_cv_uses_fused_length_variation() -> None:
    batch = next(iter(_loader()))
    spec = _spec()
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    output, prepared = _forward_window(model, batch, spec, seed=3, corruption_config=None, device=torch.device("cpu"))
    varied = output.fused_kpts.clone()
    varied[:, :, 1, 0] += torch.linspace(0.0, 0.4, varied.shape[1])
    output = output.__class__(varied, output.base_kpts, output.delta_kpts, output.valid, output.fused_theta, output.fused_theta_valid, output.fused_r_pt, output.fused_r_pt_valid)

    assert _fused_bone_cv(output, prepared["dt"], spec) > 0
