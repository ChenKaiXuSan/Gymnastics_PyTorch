from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.supervised import (
    UnityFineTuneConfig,
    discover_completed_runs,
    run_finetuned_inference,
    run_supervised_finetune,
    train_supervised_epoch,
    validate_completed_run,
)
from gymnastics.benchmarks.unity.supervised_data import (
    UNITY_SUPERVISED_FOLDS,
    UnitySupervisedWindowDataset,
    build_supervised_sequence,
    build_supervised_sequences,
)
from gymnastics.benchmarks.unity.supervised_loss import (
    UnitySupervisedLossConfig,
)
from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.rotation_aware.corruptions import CorruptionConfig
from gymnastics.fusion.rotation_aware.dataset import (
    collate_pose_pair_windows,
)
from gymnastics.fusion.rotation_aware.losses import LossConfig
from gymnastics.fusion.rotation_aware.model import RotationAwareFusionModel
from gymnastics.fusion.rotation_aware.training import save_checkpoint


UNITY_ROOT = Path("/home/data/xchen/gymnastics/unity_benchmark")
SAM3D_ROOT = Path("local/runs/unity_benchmark/sam3d")
SKELETON_PATH = Path("configs/fusion/skeleton_mhr70.yaml")


@pytest.fixture(scope="module")
def training_sequence():
    benchmark = load_unity_benchmark(
        UNITY_ROOT,
        sequence_ids=("continuous_left_060_r00",),
    )
    return build_supervised_sequence(
        benchmark,
        SAM3D_ROOT,
        "continuous_left_060_r00",
        skeleton_path=SKELETON_PATH,
        fps=60.0,
    )


def _source_checkpoint(path: Path, ablation: str = "A4") -> Path:
    skeleton = load_skeleton_spec(SKELETON_PATH)
    model = RotationAwareFusionModel(
        skeleton,
        hidden_channels=4,
        twist_residual=ablation in {"A8", "A9"},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    save_checkpoint(
        path,
        model,
        optimizer,
        loss_config=LossConfig(),
        skeleton=skeleton,
        provenance={
            "split_hash": "test-split",
            "corruption_manifest_hash": "test-corruption",
            "git_commit": "test-commit",
            "cache_manifests": {
                "unity": {
                    "layout": "legacy_direct",
                    "person_id": "unity",
                    "trials": ["continuous_left_060_r00"],
                    "generation": None,
                    "source_hash": None,
                    "config_hash": None,
                    "manifest_hash": "test-manifest",
                }
            },
        },
        training_config={"ablation": ablation, "hidden_channels": 4},
        corruption_config=CorruptionConfig(enabled_families=()),
        score=0.0,
    )
    return path


def test_supervised_epoch_updates_model_and_reports_three_losses(
    training_sequence,
) -> None:
    skeleton = load_skeleton_spec(SKELETON_PATH)
    model = RotationAwareFusionModel(skeleton, hidden_channels=4)
    dataset = UnitySupervisedWindowDataset(
        training_sequence,
        skeleton_path=SKELETON_PATH,
        length=32,
        stride=32,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_pose_pair_windows,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    metrics = train_supervised_epoch(
        model,
        loader,
        optimizer,
        skeleton,
        loss_config=UnitySupervisedLossConfig(),
        self_supervised_config=LossConfig(),
        corruption_config=CorruptionConfig(enabled_families=()),
        seed=0,
        epoch=0,
        device="cpu",
    )

    assert set(metrics) >= {
        "unity_3d_loss",
        "self_supervised_loss",
        "total_loss",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_completed_run_has_strict_provenance_and_resume_validation(
    tmp_path: Path,
    training_sequence,
) -> None:
    source = _source_checkpoint(tmp_path / "source_a4.pt")
    fold = UNITY_SUPERVISED_FOLDS["left_to_right"]
    config = UnityFineTuneConfig(
        epochs=2,
        batch_size=16,
        window_length=32,
        train_stride=8,
        device="cpu",
    )
    run = run_supervised_finetune(
        training_sequence,
        ablation="A4",
        fold=fold,
        seed=3,
        source_checkpoint=source,
        skeleton_path=SKELETON_PATH,
        output_root=tmp_path / "runs",
        config=config,
        loss_config=UnitySupervisedLossConfig(),
        self_supervised_config=LossConfig(),
        corruption_config=CorruptionConfig(enabled_families=()),
    )

    provenance = json.loads(run.provenance_path.read_text(encoding="utf-8"))
    history = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    resolved_path = run.run_root / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert provenance["fold"] == "left_to_right"
    assert provenance["train_sequence"] == "continuous_left_060_r00"
    assert provenance["test_sequence"] == "continuous_right_060_r00"
    assert provenance["static_excluded_from_training"] is True
    assert provenance["unity_gt_supervision"] is True
    assert provenance["seed"] == 3
    assert provenance["ablation"] == "A4"
    assert len(provenance["source_checkpoint_sha256"]) == 64
    assert len(provenance["final_checkpoint_sha256"]) == 64
    assert provenance["git_commit"]
    assert len(provenance["unity_manifest_sha256"]) == 64
    assert provenance["sam3d_cache_identity"]
    assert provenance["resolved_config"] == resolved
    assert len(history) == 2
    assert validate_completed_run(
        run,
        source_checkpoint_sha256=provenance["source_checkpoint_sha256"],
        resolved_config=resolved,
        unity_manifest_sha256=provenance["unity_manifest_sha256"],
    )

    provenance["fold"] = "right_to_left"
    run.provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    assert not validate_completed_run(
        run,
        source_checkpoint_sha256=provenance["source_checkpoint_sha256"],
        resolved_config=resolved,
        unity_manifest_sha256=provenance["unity_manifest_sha256"],
    )


def test_run_rejects_source_checkpoint_ablation_mismatch(
    tmp_path: Path,
    training_sequence,
) -> None:
    source = _source_checkpoint(tmp_path / "source_a4.pt", ablation="A4")

    with pytest.raises(ValueError, match="checkpoint ablation mismatch"):
        run_supervised_finetune(
            training_sequence,
            ablation="A5",
            fold=UNITY_SUPERVISED_FOLDS["left_to_right"],
            seed=0,
            source_checkpoint=source,
            skeleton_path=SKELETON_PATH,
            output_root=tmp_path / "runs",
            config=UnityFineTuneConfig(epochs=1, device="cpu"),
            loss_config=UnitySupervisedLossConfig(),
            self_supervised_config=LossConfig(),
            corruption_config=CorruptionConfig(enabled_families=()),
        )


def test_finetuned_inference_writes_only_heldout_and_static_sequences(
    tmp_path: Path,
) -> None:
    benchmark = load_unity_benchmark(UNITY_ROOT)
    sequences = build_supervised_sequences(
        benchmark,
        SAM3D_ROOT,
        skeleton_path=SKELETON_PATH,
        fps=60.0,
    )
    fold = UNITY_SUPERVISED_FOLDS["left_to_right"]
    source = _source_checkpoint(tmp_path / "source_a4.pt")
    run = run_supervised_finetune(
        sequences[fold.train_sequence],
        ablation="A4",
        fold=fold,
        seed=0,
        source_checkpoint=source,
        skeleton_path=SKELETON_PATH,
        output_root=tmp_path / "runs",
        config=UnityFineTuneConfig(
            epochs=2,
            batch_size=16,
            device="cpu",
        ),
        loss_config=UnitySupervisedLossConfig(),
        self_supervised_config=LossConfig(),
        corruption_config=CorruptionConfig(enabled_families=()),
    )

    outputs = run_finetuned_inference(
        run,
        sequences,
        skeleton_path=SKELETON_PATH,
        window_length=32,
        stride=8,
        device="cpu",
    )

    assert {item.sequence_id for item in outputs} == {
        run.test_sequence,
        "static_sweep",
    }
    assert all(
        item.metadata["ranking_group"] == "unity_supervised"
        for item in outputs
    )
    assert all(item.metadata["unity_gt_used_for_training"] for item in outputs)
    assert all(
        item.metadata["evaluation_gt_loaded_after_training"] for item in outputs
    )
    assert all(item.metadata["fold"] == run.fold for item in outputs)
    assert all(item.metadata["seed"] == run.seed for item in outputs)
    assert all(len(item.metadata["source_checkpoint_sha256"]) == 64 for item in outputs)
    assert all(len(item.metadata["final_checkpoint_sha256"]) == 64 for item in outputs)
    assert not (
        run.run_root / "inference" / f"{run.train_sequence}.npz"
    ).exists()


def test_completed_run_discovery_uses_ablation_fold_seed_cell_order(
    tmp_path: Path,
) -> None:
    runs = discover_completed_runs(
        tmp_path,
        expected_cells=(("A4", "left_to_right", 0),),
        resolved_config={},
    )

    assert runs == ()
