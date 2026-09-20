from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gymnastics.fusion.corruptions import CorruptionConfig
from gymnastics.fusion.data import build_datamodule
from gymnastics.fusion.data.base import DataConfig, SplitSpec
from gymnastics.fusion.data.synthetic import SyntheticDataModule
from gymnastics.fusion.data.windows import CycleWindowDataset, WindowConfig, window_starts
from gymnastics.fusion.sample import collate_fusion_batch
from tests.fusion.conftest import make_sample


def test_window_starts_cover_sequence():
    assert window_starts(10, 4, 3) == (0, 3, 6)
    assert window_starts(3, 4, 2) == (0,)
    assert window_starts(8, 4, 4) == (0, 4)


def test_window_dataset_phase_normalises_and_pads(skeleton):
    samples = [make_sample(skeleton, frames=48, period=16, with_reference=True), make_sample(skeleton, frames=20, period=16, subject="s2", seed=3)]
    window = WindowConfig(num_cycles=2, samples_per_cycle=8, train_stride=4)
    dataset = CycleWindowDataset(samples, skeleton=skeleton, window=window, split="train")
    assert dataset.length == 16
    # First sample: 3 cycles * 8 = 24 samples -> starts 0, 4, 8.
    # Second sample: 1 cycle -> 8 samples -> one padded window.
    assert len(dataset) == 4
    item = dataset[0]
    assert item["pose_a"].shape == (16, skeleton.num_joints, 3)
    assert item["frame_mask"].all() and item["phase_valid"].all()
    torch.testing.assert_close(item["phase"][:8], torch.arange(8) / 8.0)
    assert item["cycle_index"][:8].tolist() == [0] * 8 and item["cycle_index"][8:].tolist() == [1] * 8
    assert (item["half_index"] == -1).all()  # no middles in this sample
    with_mids = CycleWindowDataset([make_sample(skeleton, frames=48, period=16, mids=True)], skeleton=skeleton, window=window, split="train")
    halves = with_mids[0]["half_index"]
    assert halves[:4].tolist() == [0] * 4 and halves[4:8].tolist() == [1] * 4 and halves[8:12].tolist() == [0] * 4
    torch.testing.assert_close(item["delta_t"], torch.full((16,), 2.0 / 30.0))
    assert item["reference_valid"].all() and not bool(item["reference_canonical"])
    padded = dataset[3]
    assert padded["frame_mask"][:8].all() and not padded["frame_mask"][8:].any()
    assert not padded["reference_valid"].any()
    assert padded["window_id"].startswith("synthetic/s2/seq0/")
    batch = collate_fusion_batch([dataset[0], dataset[3]])
    assert batch["pose_a"].shape == (2, 16, skeleton.num_joints, 3)
    assert batch["subject_id"] == ["s1", "s2"]


def test_window_dataset_without_phase_normalisation(skeleton):
    sample = make_sample(skeleton, frames=40, period=16, cycles=False)
    dataset = CycleWindowDataset([sample], skeleton=skeleton, window=WindowConfig(num_cycles=2, samples_per_cycle=8), split="val", phase_normalize=False)
    assert len(dataset) == 3  # eval stride = length = 16 over 40 frames -> 0, 16, 24
    item = dataset[0]
    assert not item["phase_valid"].any() and (item["cycle_index"] == -1).all()
    torch.testing.assert_close(item["delta_t"], torch.full((16,), 1.0 / 30.0))


def test_window_dataset_corruption_is_reproducible_per_epoch(skeleton):
    sample = make_sample(skeleton, frames=48, period=16)
    config = CorruptionConfig(view_probability=1.0)
    dataset = CycleWindowDataset([sample], skeleton=skeleton, window=WindowConfig(num_cycles=2, samples_per_cycle=8), split="train", corruption=config, seed=1)
    first, again = dataset[0], dataset[0]
    assert torch.equal(first["pose_a"], again["pose_a"])
    assert "clean_a" in first and torch.equal(first["clean_valid_a"], torch.ones_like(first["clean_valid_a"]))
    assert (first["corruption_mask_a"] | first["corruption_mask_b"]).any()
    dataset.set_epoch(1)
    later = dataset[0]
    assert not torch.equal(first["pose_a"], later["pose_a"]) or not torch.equal(first["pose_b"], later["pose_b"])


def test_window_dataset_rejects_wrong_skeleton(skeleton):
    from gymnastics.fusion.skeleton import build_common_skeleton

    sample = make_sample(skeleton)
    with pytest.raises(ValueError):
        CycleWindowDataset([sample], skeleton=build_common_skeleton("mhr70"), window=WindowConfig(samples_per_cycle=8), split="train")


def test_split_spec_rejects_overlap():
    with pytest.raises(ValueError):
        SplitSpec(train=("a",), val=("a",))


def test_data_config_from_mapping_collects_options():
    config = DataConfig.from_mapping({"name": "synthetic", "batch_size": 2, "window": {"num_cycles": 1, "samples_per_cycle": 8}, "corruption": {"enabled": False}, "split": {"train": ["a"], "val": [], "test": []}, "custom": 5, "options": {"x": 1}})
    assert config.window.length == 8 and not config.corruption.enabled
    assert config.options == {"x": 1, "custom": 5}
    assert config.split.train == ("a",)
    assert config.to_dict()["name"] == "synthetic"


def test_synthetic_datamodule_end_to_end():
    datamodule = SyntheticDataModule({"name": "synthetic", "batch_size": 3, "window": {"num_cycles": 2, "samples_per_cycle": 8}, "options": {"subjects": 5, "sequences_per_subject": 1, "frames": 48, "period": 12}})
    datamodule.setup("fit")
    datamodule.setup("test")
    summary = datamodule.summary()
    assert summary["subjects"] == {"train": 3, "val": 1, "test": 1}
    assert summary["windows"]["train"] > 0
    train_batch = next(iter(datamodule.train_dataloader()))
    assert "clean_a" in train_batch and not train_batch["reference_valid"].any()  # references stripped in training
    val_batch = next(iter(datamodule.val_dataloader()))
    assert val_batch["reference_valid"].any() and "clean_a" in val_batch
    test_batch = next(iter(datamodule.test_dataloader()))
    assert "clean_a" not in test_batch and test_batch["reference_valid"].any()
    assert set(test_batch["subject_id"]) <= set(datamodule.split.test)
    assert isinstance(build_datamodule({"name": "synthetic"}), SyntheticDataModule)
    with pytest.raises(ValueError):
        build_datamodule({"name": "nope"})


def test_explicit_split_and_cap():
    datamodule = SyntheticDataModule({"name": "synthetic", "options": {"subjects": 4, "sequences_per_subject": 1, "frames": 24, "period": 12}, "split": {"train": ["subject_03"], "val": ["subject_00"], "test": ["subject_01"]}, "window": {"samples_per_cycle": 8, "num_cycles": 1}})
    datamodule.setup()
    assert datamodule.split == SplitSpec(train=("subject_03",), val=("subject_00",), test=("subject_01",))
    with pytest.raises(ValueError):
        SyntheticDataModule({"name": "synthetic", "options": {"subjects": 2}, "split": {"train": ["ghost"]}}).setup()
    # A partial explicit split keeps the other lists empty instead of falling back.
    partial = SyntheticDataModule({"name": "synthetic", "options": {"subjects": 4, "sequences_per_subject": 1, "frames": 24, "period": 12}, "split": {"train": ["subject_00"]}, "window": {"samples_per_cycle": 8, "num_cycles": 1}})
    partial.setup()
    assert partial.split == SplitSpec(train=("subject_00",), val=(), test=())


def test_sample_cache_round_trip(tmp_path, skeleton):
    from gymnastics.fusion.data.sample_cache import load_samples, save_samples

    samples = [make_sample(skeleton, with_reference=True), make_sample(skeleton, subject="s2", cycles=False)]
    directory = save_samples(tmp_path / "cache", samples, config={"name": "synthetic"})
    restored = load_samples(directory)
    assert restored is not None and len(restored) == 2
    for original, loaded in zip(samples, restored):
        assert loaded.key == original.key and loaded.cycle_bounds == original.cycle_bounds
        np.testing.assert_array_equal(loaded.view_a, original.view_a)
        np.testing.assert_array_equal(loaded.valid_b, original.valid_b)
        assert (loaded.reference is None) == (original.reference is None)
        assert dict(loaded.metadata) == dict(original.metadata)
    assert load_samples(tmp_path / "missing") is None


def test_datamodule_uses_sample_cache(tmp_path):
    config = {"name": "synthetic", "cache_dir": str(tmp_path), "window": {"samples_per_cycle": 8, "num_cycles": 1}, "options": {"subjects": 3, "sequences_per_subject": 1, "frames": 24, "period": 12}}
    first = SyntheticDataModule(config)
    first.setup()
    cached_dirs = list(tmp_path.glob("synthetic_*"))
    assert len(cached_dirs) == 1 and (cached_dirs[0] / "index.json").is_file()
    second = SyntheticDataModule(config)
    second.load_samples = lambda: (_ for _ in ()).throw(AssertionError("cache not used"))  # type: ignore[method-assign]
    second.setup()
    assert [s.key for s in second.samples] == [s.key for s in first.samples]


def test_full_context_window_is_one_window_per_sequence(skeleton):
    samples = [make_sample(skeleton, frames=48, period=16), make_sample(skeleton, frames=32, period=16, subject="s2")]
    window = WindowConfig(num_cycles=None, samples_per_cycle=8)
    assert window.full_context and window.length is None
    dataset = CycleWindowDataset(samples, skeleton=skeleton, window=window, split="train")
    assert len(dataset) == 2 and dataset.length == 24  # longest sequence: 3 cycles * 8
    assert dataset[1]["frame_mask"].sum() == 16
    from gymnastics.fusion.model import CycleAwareModelConfig

    assert CycleAwareModelConfig.from_mapping({"samples_per_cycle": 8, "long_motion": {"num_cycles": None}}).window_length is None
