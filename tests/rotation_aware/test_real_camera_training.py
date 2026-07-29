from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from gymnastics.benchmarks.unity.camera_features import CameraFeatureSequence
from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.rotation_aware.corruptions import CorruptionConfig
from gymnastics.fusion.rotation_aware.inference import canonicalize_trial
from gymnastics.fusion.rotation_aware.losses import LossConfig
from gymnastics.fusion.rotation_aware.model import RotationAwareFusionModel
from gymnastics.fusion.rotation_aware.real_camera_data import RealCameraTrial
from gymnastics.fusion.rotation_aware.real_camera_training import (
    RealCameraTrainingConfig,
    expand_and_freeze_camera_model,
    infer_real_camera_cell,
    train_real_camera_cell,
)
from gymnastics.fusion.rotation_aware.schema import PosePairTrial


SPEC_PATH = Path("configs/fusion/skeleton_mhr70.yaml")
SPEC = load_skeleton_spec(SPEC_PATH)


def _source_checkpoint(path: Path) -> Path:
    torch.manual_seed(7)
    model = RotationAwareFusionModel(SPEC, hidden_channels=16)
    torch.save(
        {
            "model": model.state_dict(),
            "training_config": {"ablation": "A6", "hidden_channels": 16},
            "loss_config": asdict(LossConfig(complete_cycle_rom_weight=0.0)),
            "corruption_config": asdict(CorruptionConfig()),
            "provenance": {"split_hash": "unit-test"},
        },
        path,
    )
    return path


def _real_trial(person_id: str, ablation: str) -> RealCameraTrial:
    frames = 4
    points = np.ones((frames, 70, 3), dtype=np.float32)
    points[..., 0] += np.arange(70, dtype=np.float32)
    points[..., 1] += np.arange(frames, dtype=np.float32)[:, None]
    valid = np.ones((frames, 70), dtype=bool)
    raw = PosePairTrial(
        face=points,
        side=points + np.asarray((0.2, 0.1, 0.3), dtype=np.float32),
        valid_face=valid,
        valid_side=valid,
        timestamps=np.arange(frames, dtype=np.float64) / 60.0,
        face_map=np.arange(frames, dtype=np.int32),
        side_map=np.arange(frames, dtype=np.int32),
        joint_names=tuple(mhr_names),
        person_id=person_id,
        trial_id="cycle_000",
        fps=60.0,
    )
    features = None
    if ablation != "G0":
        features = CameraFeatureSequence(
            global_features=np.zeros(19, dtype=np.float32),
            joint_features=np.ones((frames, 70, 8), dtype=np.float32),
            valid=valid,
            global_schema=tuple(f"g{i}" for i in range(19)),
            joint_schema=tuple(f"j{i}" for i in range(8)),
        )
    return RealCameraTrial(
        canonical_trial=canonicalize_trial(raw, SPEC),
        camera_fit=None if ablation == "G0" else object(),  # contract only
        camera_features=features,
        ablation=ablation,
    )


def test_expansion_freezes_every_non_camera_parameter(tmp_path: Path) -> None:
    source = _source_checkpoint(tmp_path / "source.pt")

    expanded = expand_and_freeze_camera_model(
        source, skeleton_path=SPEC_PATH, ablation="G4", seed=3
    )

    assert expanded.trainable_parameter_prefixes == (
        "camera_conditioner.",
        "camera_delta_head.",
    )
    assert all(
        parameter.requires_grad
        == name.startswith(expanded.trainable_parameter_prefixes)
        for name, parameter in expanded.model.named_parameters()
    )
    source_state = torch.load(source, weights_only=False)["model"]
    for name, value in expanded.model.state_dict().items():
        if not name.startswith(expanded.trainable_parameter_prefixes):
            assert torch.equal(value, source_state[name])


def test_training_writes_isolation_provenance_and_preserves_backbone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_checkpoint(tmp_path / "source.pt")

    def fake_train(model, *args, **kwargs):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.add_(0.01)
        return {"loss": 1.0}

    monkeypatch.setattr(
        "gymnastics.fusion.rotation_aware.real_camera_training.train_one_epoch",
        fake_train,
    )
    monkeypatch.setattr(
        "gymnastics.fusion.rotation_aware.real_camera_training.validate",
        lambda *args, **kwargs: {"loss": 0.5, "score": 0.75},
    )
    run = train_real_camera_cell(
        train_trials=[_real_trial("train", "G4")],
        val_trials=[_real_trial("val", "G4")],
        ablation="G4",
        seed=0,
        source_checkpoint=source,
        skeleton_path=SPEC_PATH,
        output_root=tmp_path / "runs",
        config=RealCameraTrainingConfig(
            epochs=2,
            batch_size=1,
            window_length=8,
            train_stride=4,
            eval_stride=4,
        ),
    )
    payload = torch.load(run.checkpoint, weights_only=False)

    assert payload["provenance"]["triangulated_3d_available_to_training"] is False
    assert payload["provenance"]["test_people_available_to_training"] is False
    assert payload["trainable_parameter_prefixes"] == [
        "camera_conditioner.",
        "camera_delta_head.",
    ]
    source_state = torch.load(source, weights_only=False)["model"]
    for name, value in payload["model"].items():
        if not name.startswith(tuple(payload["trainable_parameter_prefixes"])):
            assert torch.equal(value, source_state[name])


def test_g0_copies_source_without_optimizer_step(tmp_path: Path) -> None:
    source = _source_checkpoint(tmp_path / "source.pt")
    run = train_real_camera_cell(
        train_trials=[_real_trial("train", "G0")],
        val_trials=[_real_trial("val", "G0")],
        ablation="G0",
        seed=0,
        source_checkpoint=source,
        skeleton_path=SPEC_PATH,
        output_root=tmp_path / "runs",
        config=RealCameraTrainingConfig(epochs=2),
    )
    payload = torch.load(run.checkpoint, weights_only=False)

    assert payload["optimizer_steps"] == 0
    assert payload["trainable_parameter_prefixes"] == []
    source_state = torch.load(source, weights_only=False)["model"]
    assert all(torch.equal(value, source_state[name]) for name, value in payload["model"].items())

    outputs = infer_real_camera_cell(
        run,
        test_trials=[_real_trial("test", "G0")],
        skeleton_path=SPEC_PATH,
        window_length=8,
        stride=4,
    )
    with np.load(outputs[0], allow_pickle=False) as result:
        assert result["kpts_world"].shape == (4, 70, 3)
        assert result["joint_valid"].shape == (4, 70)
        assert "triangulated_3d" not in result.files
