import json
from pathlib import Path

import numpy as np
import torch

from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.rotation_aware import dataset as dataset_module
from gymnastics.fusion.rotation_aware.dataset import (
    PosePairWindowDataset,
    SplitManifest,
    WindowConfig,
    build_split_manifest,
    collate_pose_pair_windows,
)
from gymnastics.fusion.rotation_aware.config import load_skeleton_spec
from gymnastics.fusion.rotation_aware.schema import PosePairTrial


SPEC = load_skeleton_spec(Path("configs/fusion/skeleton_mhr70.yaml"))


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
    dataset = PosePairWindowDataset(
        [_trial("1", 160)], skeleton=SPEC, manifest=SplitManifest(train=("1",), val=("2",), test=("3",)), split="train"
    )

    assert dataset.config == WindowConfig()
    assert len(dataset) == 2
    first, last = dataset[0], dataset[1]
    assert first["window_start"] == 0
    assert last["window_start"] == 32
    assert first["padding_mask"].sum().item() == 128
    assert last["padding_mask"].sum().item() == 128
    assert torch.equal(last["loss_mask"], last["padding_mask"].unsqueeze(-1) & last["valid_face"] & last["valid_side"])


def test_eval_windows_use_64_stride_and_short_trial_padding_is_excluded():
    dataset = PosePairWindowDataset(
        [_trial("3", 40)], skeleton=SPEC, manifest=SplitManifest(train=("1",), val=("2",), test=("3",)), split="test"
    )

    sample = dataset[0]
    assert dataset.config.eval_stride == 64
    assert sample["padding_mask"].sum().item() == 40
    assert not sample["loss_mask"][40:].any()
    assert not sample["valid_face"][40:].any()
    assert sample["complete_cycle"]
    assert sample["timestamps"].shape == (128,)
    assert sample["dt"].shape == (128,)
    torch.testing.assert_close(sample["dt"][1:40], torch.full((39,), 1 / 60))
    assert not sample["dt"][40:].any()


def test_complete_cycle_is_true_only_for_an_entire_trial_window():
    dataset = PosePairWindowDataset(
        [_trial("1", 160)], skeleton=SPEC, manifest=SplitManifest(train=("1",), val=("2",), test=("3",)), split="train"
    )

    assert not dataset[0]["complete_cycle"]
    assert not dataset[1]["complete_cycle"]


def test_complete_cycle_dataset_emits_long_trial_without_padding() -> None:
    manifest = SplitManifest(train=("1",), val=(), test=())
    windows = PosePairWindowDataset(
        [_trial("1", 257)], skeleton=SPEC, manifest=manifest, split="train"
    )
    cycles = dataset_module.PosePairCompleteCycleDataset(
        [_trial("1", 257)], skeleton=SPEC, manifest=manifest, split="train"
    )

    assert not any(windows[index]["complete_cycle"] for index in range(len(windows)))
    sample = cycles[0]
    assert sample["face"].shape == (257, len(mhr_names), 3)
    assert sample["padding_mask"].all()
    assert sample["complete_cycle"]
    assert sample["window_id"] == "person_1/cycle_000/complete_cycle"


def test_collate_stacks_window_tensors_without_unmasking_padding():
    dataset = PosePairWindowDataset(
        [_trial("3", 40)], skeleton=SPEC, manifest=SplitManifest(train=("1",), val=("2",), test=("3",)), split="test"
    )
    batch = collate_pose_pair_windows([dataset[0], dataset[0]])

    assert batch["face"].shape == (2, 128, len(mhr_names), 3)
    assert batch["loss_mask"].shape == (2, 128, len(mhr_names))
    assert not batch["loss_mask"][:, 40:].any()


def test_dataset_rejects_trials_outside_the_requested_manifest_split():
    manifest = SplitManifest(train=("1",), val=("2",), test=("3",))

    try:
        PosePairWindowDataset([_trial("2", 40)], skeleton=SPEC, manifest=manifest, split="train")
    except ValueError as error:
        assert "not members" in str(error)
    else:
        raise AssertionError("expected wrong-person dataset construction to be rejected")


def test_windows_emit_the_same_full_trial_bone_baseline_and_global_indices():
    dataset = PosePairWindowDataset(
        [_trial("1", 160)], skeleton=SPEC, manifest=SplitManifest(train=("1",), val=("2",), test=("3",)), split="train"
    )
    first, last = dataset[0], dataset[1]

    assert first["trial_bone_baseline"].shape == (len(SPEC.bones),)
    assert first["trial_bone_baseline_valid"].shape == (len(SPEC.bones),)
    assert first["trial_bone_baseline_valid"].all()
    torch.testing.assert_close(first["trial_bone_baseline"], last["trial_bone_baseline"])
    assert first["global_frame_index"][0].item() == 0
    assert last["global_frame_index"][0].item() == 32
