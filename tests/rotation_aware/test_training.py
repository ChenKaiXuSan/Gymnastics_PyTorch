from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fuse.rotation_aware.config import RoleSpec, SkeletonSpec
from fuse.rotation_aware.dataset import collate_pose_pair_windows
from fuse.rotation_aware.losses import LossConfig
from fuse.rotation_aware.model import RotationAwareFusionModel
from fuse.rotation_aware.training import load_checkpoint, save_checkpoint, train_one_epoch, validate


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
    config = LossConfig(corruption_recovery_weight=0.0, complete_cycle_rom_weight=0.0)
    initial = validate(model, _loader(), spec, loss_config=config, seed=9)["loss"]

    for epoch in range(8):
        metrics = train_one_epoch(model, _loader(), optimizer, spec, loss_config=config, seed=9, epoch=epoch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    final = validate(model, _loader(), spec, loss_config=config, seed=9)["loss"]
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
        score=metrics_a["score"],
    )
    loaded = load_checkpoint(path, model, optimizer)
    assert loaded["score"] == metrics_a["score"]
    assert loaded["provenance"]["split_hash"] == "split"
    assert "model" in loaded and "optimizer" in loaded and "skeleton" in loaded and "loss_config" in loaded
