import json

import numpy as np
import torch

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware.dataset import (
    PosePairWindowDataset,
    WindowConfig,
    build_split_manifest,
    collate_pose_pair_windows,
)
from fuse.rotation_aware.schema import PosePairTrial


def _trial(person_id: str, frames: int) -> PosePairTrial:
    points = np.arange(frames * len(mhr_names) * 3, dtype=np.float32).reshape(frames, len(mhr_names), 3) + 1
    valid = np.ones((frames, len(mhr_names)), dtype=bool)
    return PosePairTrial(
        face=points,
        side=points + 10,
        valid_face=valid,
        valid_side=valid,
        timestamps=np.arange(frames, dtype=np.float64) / 60,
        face_map=np.arange(frames, dtype=np.int32),
        side_map=np.arange(frames, dtype=np.int32),
        joint_names=tuple(mhr_names),
        person_id=person_id,
        trial_id="cycle_000",
        fps=60.0,
    )


def test_subjects_do_not_cross_splits(tmp_path):
    fold_json = tmp_path / "fold_00.json"
    fold_json.write_text(
        json.dumps(
            {
                "train": [{"person_id": "1", "label": 8}, {"person_id": "1", "label": 9}],
                "val": [{"person_id": "2", "label": 4}],
                "test": [{"person_id": "3", "label": 1}],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_split_manifest(fold_json)

    assert manifest.train == ("1",)
    assert not (set(manifest.train) & set(manifest.val))
    assert not (set(manifest.train) & set(manifest.test))
    assert not (set(manifest.val) & set(manifest.test))


def test_split_manifest_rejects_person_leakage():
    fold = {"train": [{"person_id": "1"}], "val": [{"person_id": "1"}], "test": []}
    try:
        build_split_manifest(fold)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("expected subject leakage to be rejected")


def test_window_defaults_padding_and_train_stride_are_masked_from_loss():
    dataset = PosePairWindowDataset([_trial("1", 160)], person_ids=("1",), split="train")

    assert dataset.config == WindowConfig()
    assert len(dataset) == 2
    first, last = dataset[0], dataset[1]
    assert first["window_start"] == 0
    assert last["window_start"] == 32
    assert first["padding_mask"].sum().item() == 128
    assert last["padding_mask"].sum().item() == 128
    assert torch.equal(last["loss_mask"], last["padding_mask"].unsqueeze(-1) & last["valid_face"] & last["valid_side"])


def test_eval_windows_use_64_stride_and_short_trial_padding_is_excluded():
    dataset = PosePairWindowDataset([_trial("1", 40)], person_ids=("1",), split="eval")

    sample = dataset[0]
    assert dataset.config.eval_stride == 64
    assert sample["padding_mask"].sum().item() == 40
    assert not sample["loss_mask"][40:].any()
    assert not sample["valid_face"][40:].any()


def test_collate_stacks_window_tensors_without_unmasking_padding():
    dataset = PosePairWindowDataset([_trial("1", 40)], person_ids=("1",), split="eval")
    batch = collate_pose_pair_windows([dataset[0], dataset[0]])

    assert batch["face"].shape == (2, 128, len(mhr_names), 3)
    assert batch["loss_mask"].shape == (2, 128, len(mhr_names))
    assert not batch["loss_mask"][:, 40:].any()
