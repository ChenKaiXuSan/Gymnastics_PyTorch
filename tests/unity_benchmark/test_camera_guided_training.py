from __future__ import annotations

import json
from pathlib import Path

import torch

from gymnastics.benchmarks.unity.camera_guided_data import (
    build_camera_guided_sequences,
)
from gymnastics.benchmarks.unity.camera_guided_training import (
    CameraGuidedTrainingConfig,
    load_camera_guided_model,
    run_camera_guided_inference,
    train_camera_guided_run,
)
from gymnastics.benchmarks.unity.dataset import load_unity_benchmark
from gymnastics.benchmarks.unity.supervised_data import UNITY_SUPERVISED_FOLDS


UNITY_ROOT = Path("/home/data/xchen/gymnastics/unity_benchmark")
SAM3D_ROOT = Path("local/runs/unity_benchmark/sam3d")
SKELETON = Path("configs/fusion/skeleton_mhr70.yaml")
SOURCE_A6 = Path(
    "local/runs/fuse_rotation_aware/runs/"
    "all137_a6_e100_seed0/checkpoints/best.pt"
)


def _sequences(ablation: str):
    benchmark = load_unity_benchmark(UNITY_ROOT)
    fold = UNITY_SUPERVISED_FOLDS["left_to_right"]
    return fold, build_camera_guided_sequences(
        benchmark,
        SAM3D_ROOT,
        skeleton_path=SKELETON,
        fps=60.0,
        fold=fold,
        ablation=ablation,
    )


def test_one_epoch_camera_guided_run_records_only_self_supervised_losses(
    tmp_path: Path,
) -> None:
    fold, sequences = _sequences("G4")
    config = CameraGuidedTrainingConfig(
        epochs=1,
        learning_rate=1e-4,
        weight_decay=1e-4,
        window_length=128,
        train_stride=32,
        batch_size=1,
        device="cpu",
    )

    run = train_camera_guided_run(
        sequences[fold.train_sequence],
        ablation="G4",
        fold=fold,
        seed=0,
        source_checkpoint=SOURCE_A6,
        skeleton_path=SKELETON,
        output_root=tmp_path,
        config=config,
    )

    history = json.loads(run.history_path.read_text(encoding="utf-8"))
    provenance = json.loads(run.provenance_path.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert "total" in history[0]
    assert not any("unity_3d" in key for key in history[0])
    assert provenance["unity_native_3d_available_to_training"] is False
    assert provenance["triangulated_3d_available_to_training"] is False
    assert provenance["train_sample_ids"] == sequences[
        fold.train_sequence
    ].sample_ids.tolist()
    assert run.final_checkpoint.is_file()

    loaded = load_camera_guided_model(
        run.final_checkpoint,
        SKELETON,
        device="cpu",
    )
    assert loaded.ablation == "G4"
    assert loaded.model.camera_config is not None
    assert loaded.model.camera_config.mode == "film"


def test_g0_checkpoint_has_no_camera_parameters(tmp_path: Path) -> None:
    fold, sequences = _sequences("G0")
    run = train_camera_guided_run(
        sequences[fold.train_sequence],
        ablation="G0",
        fold=fold,
        seed=1,
        source_checkpoint=SOURCE_A6,
        skeleton_path=SKELETON,
        output_root=tmp_path,
        config=CameraGuidedTrainingConfig(epochs=1, device="cpu"),
    )

    loaded = load_camera_guided_model(
        run.final_checkpoint,
        SKELETON,
        device="cpu",
    )

    assert loaded.model.camera_config is None
    assert not any(
        name.startswith("camera_conditioner.")
        for name in loaded.model.state_dict()
    )


def test_camera_guided_inference_writes_only_heldout_and_static(
    tmp_path: Path,
) -> None:
    fold, sequences = _sequences("G4")
    run = train_camera_guided_run(
        sequences[fold.train_sequence],
        ablation="G4",
        fold=fold,
        seed=2,
        source_checkpoint=SOURCE_A6,
        skeleton_path=SKELETON,
        output_root=tmp_path,
        config=CameraGuidedTrainingConfig(epochs=1, device="cpu"),
    )

    outputs = run_camera_guided_inference(
        run,
        sequences,
        skeleton_path=SKELETON,
        window_length=128,
        stride=64,
        device="cpu",
    )

    assert {item.sequence_id for item in outputs} == {
        fold.test_sequence,
        "static_sweep",
    }
    assert all(item.method == "G4" for item in outputs)
    assert not (run.run_root / "inference" / f"{fold.train_sequence}.npz").exists()
    assert torch.load(
        run.final_checkpoint, map_location="cpu", weights_only=False
    )["training_config"]["ablation"] == "G4"
