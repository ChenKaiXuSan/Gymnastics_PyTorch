from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gymnastics.benchmarks.unity.extrinsic_training import (
    EXTRINSIC_METHODS,
    ExtrinsicRun,
    ExtrinsicSequence,
    ExtrinsicTrainingConfig,
    calibrated_supervised_loss,
    make_extrinsic_model,
    train_extrinsic_run,
    validate_extrinsic_run,
)
from gymnastics.benchmarks.unity.supervised_data import (
    UNITY_SUPERVISED_FOLDS,
)
from gymnastics.benchmarks.unity.supervised_loss import (
    torch_map_mhr70_to_unity16,
)


def _sequence(sequence_id: str, frames: int = 4) -> ExtrinsicSequence:
    rng = np.random.default_rng(7)
    face = rng.normal(0.0, 0.2, size=(frames, 70, 3)).astype(np.float32)
    face[:, 9] = np.asarray((-0.1, 0.0, 0.0))
    face[:, 10] = np.asarray((0.1, 0.0, 0.0))
    side = face.copy()
    valid = np.ones((frames, 70), dtype=bool)
    with torch.no_grad():
        mapped, mapped_valid = torch_map_mhr70_to_unity16(
            torch.from_numpy(face)[None],
            torch.from_numpy(valid)[None],
        )
    pixels = rng.uniform(
        50.0, 150.0, size=(frames, 2, 70, 2)
    ).astype(np.float32)
    projection = np.asarray(
        [
            [[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[100.0, 0.0, 0.0, -100.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return ExtrinsicSequence(
        sequence_id=sequence_id,
        sample_ids=np.arange(frames, dtype=np.int64),
        face_3d=face,
        side_3d=side,
        valid_face_3d=valid,
        valid_side_3d=valid,
        pixels_2d=pixels,
        valid_2d=np.ones((frames, 2, 70), dtype=bool),
        gt_unity16_m=mapped[0].numpy(),
        gt_valid=mapped_valid[0].numpy(),
        relative_rotation=np.eye(3, dtype=np.float32),
        projection=projection,
        image_size=np.asarray(((200.0, 200.0), (200.0, 200.0)), dtype=np.float32),
        source_identity={"manifest_sha256": "a" * 64, "sam3d_sha256": "b" * 64},
    )


def test_extrinsic_sequence_rejects_duplicate_sample_ids() -> None:
    sequence = _sequence("continuous_left_060_r00")
    with pytest.raises(ValueError, match="unique"):
        ExtrinsicSequence(
            **{
                **sequence.as_dict(),
                "sample_ids": np.asarray((1, 1, 2, 3)),
            }
        )


@pytest.mark.parametrize("method", EXTRINSIC_METHODS)
def test_calibrated_supervised_loss_is_finite_and_differentiable(
    method: str,
) -> None:
    sequence = _sequence("continuous_left_060_r00")
    model = make_extrinsic_model(method, hidden_channels=8)
    loss, metrics = calibrated_supervised_loss(model, method, sequence, "cpu")
    assert torch.isfinite(loss)
    assert np.isfinite(metrics["supervised_loss"])
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_train_extrinsic_run_rejects_wrong_training_direction(
    tmp_path: Path,
) -> None:
    fold = UNITY_SUPERVISED_FOLDS["left_to_right"]
    with pytest.raises(ValueError, match="training sequence"):
        train_extrinsic_run(
            _sequence(fold.test_sequence),
            method="extrinsic_gate",
            fold=fold,
            seed=0,
            output_root=tmp_path,
            config=ExtrinsicTrainingConfig(
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                hidden_channels=8,
                device="cpu",
            ),
        )


def test_completed_extrinsic_run_is_hash_validated(tmp_path: Path) -> None:
    fold = UNITY_SUPERVISED_FOLDS["left_to_right"]
    run = train_extrinsic_run(
        _sequence(fold.train_sequence),
        method="extrinsic_gate",
        fold=fold,
        seed=2,
        output_root=tmp_path,
        config=ExtrinsicTrainingConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            hidden_channels=8,
            device="cpu",
        ),
    )
    assert isinstance(run, ExtrinsicRun)
    assert validate_extrinsic_run(run)
    payload = json.loads(run.provenance_path.read_text(encoding="utf-8"))
    assert payload["method"] == "extrinsic_gate"
    assert payload["fold"] == "left_to_right"
    assert payload["seed"] == 2
    assert payload["train_sample_ids"] == [0, 1, 2, 3]
    assert payload["test_sequence"] == fold.test_sequence
    run.history_path.write_text("[]", encoding="utf-8")
    assert not validate_extrinsic_run(run)

