from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from fuse.rotation_aware.config import RoleSpec, SkeletonSpec
from fuse.rotation_aware.dataset import collate_pose_pair_windows
from fuse.rotation_aware.losses import LossConfig
from fuse.rotation_aware.model import RotationAwareFusionModel
from fuse.rotation_aware.prefetch import ordered_prefetch, pin_tensor_batch
from fuse.rotation_aware.profiling import StageProfiler
from fuse.rotation_aware.training import _corrupt_batch, _feature_bundle, _forward_window, _fused_bone_cv, _prepare_window, _tensor_batch, load_checkpoint, save_checkpoint, train_one_epoch, validate


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
    return SkeletonSpec(
        "tiny",
        names,
        ((0, 1), (2, 3), (0, 2), (1, 3)),
        roles,
        tuple(roles),
        {"upper_body": names[2:]},
    )


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
    baseline = torch.tensor([2.0, 2.0, 2.0, 2.0])
    return [
        {
            "face": sequence.clone() + index * 0.01,
            "side": sequence.clone() + index * 0.01,
            "valid_face": valid.clone(),
            "valid_side": valid.clone(),
            "padding_mask": torch.ones(5, dtype=torch.bool),
            "loss_mask": valid.clone(),
            "dt": torch.ones(5),
            "trial_bone_baseline": baseline.clone(),
            "trial_bone_baseline_valid": torch.ones_like(baseline, dtype=torch.bool),
            "global_frame_index": torch.arange(5),
            "person_id": "person-1",
            "trial_id": "trial-1",
            "complete_cycle": True,
            "window_id": f"window-{index}",
        }
        for index in range(count)
    ]


def _loader(*, count: int = 8, batch_size: int = 4, reverse: bool = False) -> DataLoader[dict[str, object]]:
    samples = _samples(count)
    return DataLoader(list(reversed(samples)) if reverse else samples, batch_size=batch_size, shuffle=False, collate_fn=collate_pose_pair_windows)


def _checkpoint_provenance() -> dict[str, object]:
    return {
        "split_hash": "split",
        "corruption_manifest_hash": "corrupt",
        "git_commit": "abc",
        "cache_manifests": {
            "person-1": {
                "layout": "legacy_direct",
                "person_id": "person-1",
                "trials": ["trial-1"],
                "generation": None,
                "source_hash": None,
                "config_hash": None,
                "manifest_hash": "manifest",
            }
        },
    }


def _complete_cycle_batch(frames: int = 129) -> dict[str, object]:
    sample = _samples(1)[0]
    face = sample["face"][:1].repeat(frames, 1, 1)
    side = sample["side"][:1].repeat(frames, 1, 1)
    valid = sample["valid_face"][:1].repeat(frames, 1)
    sample.update(
        {
            "face": face,
            "side": side,
            "valid_face": valid,
            "valid_side": valid.clone(),
            "padding_mask": torch.ones(frames, dtype=torch.bool),
            "loss_mask": valid.clone(),
            "dt": torch.ones(frames),
            "global_frame_index": torch.arange(frames),
            "complete_cycle": True,
            "window_id": "person_person-1/trial-1/complete_cycle",
        }
    )
    return collate_pose_pair_windows([sample])


def test_ordered_prefetch_preserves_source_order() -> None:
    values = list(ordered_prefetch(range(8), lambda value: value * 2, depth=3))

    assert values == [value * 2 for value in range(8)]


def test_ordered_prefetch_propagates_worker_exceptions() -> None:
    def prepare(value: int) -> int:
        if value == 3:
            raise RuntimeError("corruption failed")
        return value

    iterator = ordered_prefetch(range(8), prepare, depth=3)
    assert [next(iterator) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="corruption failed"):
        next(iterator)


def test_prefetched_preparation_matches_direct_corruption_for_stable_epochs() -> None:
    from fuse.rotation_aware.corruptions import CorruptionConfig

    spec = _spec()
    batches = list(_loader(count=8, batch_size=2))
    corruption = CorruptionConfig(
        enabled_families=("spike_noise",), spike_probability=1.0, spike_scale=0.02
    )
    tensor_keys = (
        "face", "side", "corrupted_valid_face", "corrupted_valid_side",
        "reference_face", "reference_side", "reference_valid_face", "reference_valid_side",
        "face_corruption_mask", "side_corruption_mask",
    )

    for epoch in (0, 1):
        direct = [
            _prepare_window(batch, seed=41, skeleton=spec, corruption_config=corruption, epoch=epoch)
            for batch in batches
        ]
        prefetched = list(
            ordered_prefetch(
                batches,
                lambda batch: _prepare_window(
                    batch, seed=41, skeleton=spec, corruption_config=corruption, epoch=epoch
                ),
                depth=2,
            )
        )
        for direct_batch, prefetched_batch in zip(direct, prefetched, strict=True):
            for key in tensor_keys:
                assert torch.equal(direct_batch[key], prefetched_batch[key])


def test_pin_tensor_batch_preserves_input_and_metadata() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    metadata = ["window-1", "window-2"]
    batch = {"face": tensor, "window_id": metadata, "count": 2}

    pinned = pin_tensor_batch(batch)

    assert pinned is not batch
    assert pinned["window_id"] is metadata
    assert pinned["count"] == 2
    assert batch["face"] is tensor
    assert torch.equal(pinned["face"], tensor)
    if torch.cuda.is_available():
        assert pinned["face"].is_pinned()
    else:
        assert pinned["face"] is tensor


def test_tensor_batch_forwards_non_blocking_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    original_to = torch.Tensor.to

    def recording_to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(bool(kwargs.get("non_blocking", False)))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)
    result = _tensor_batch({"face": torch.ones(1), "window_id": ["window-1"]}, torch.device("cpu"), non_blocking=True)

    assert calls == [True]
    assert torch.equal(result["face"], torch.ones(1))


def test_stage_profiler_collects_cpu_wall_time() -> None:
    profiler = StageProfiler(enabled=True, device=torch.device("cpu"))
    with profiler.stage("corruption"):
        torch.arange(1024).sum()
    summary = profiler.summary()
    assert summary["corruption"]["calls"] == 1
    assert summary["corruption"]["wall_seconds"] >= 0
    json.dumps(summary)

    assert StageProfiler(enabled=False, device=torch.device("cpu")).summary() == {}


def test_stage_profiler_uses_profiler_device_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda:1")
    stream = object()
    event_records: list[object] = []
    stream_devices: list[torch.device] = []
    synchronized: list[torch.device] = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self, selected_stream: object) -> None:
            event_records.append(selected_stream)

        def elapsed_time(self, other: object) -> float:
            assert isinstance(other, FakeEvent)
            return 2.0

    def fake_current_stream(selected_device: torch.device) -> object:
        stream_devices.append(selected_device)
        return stream

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", fake_current_stream)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)

    profiler = StageProfiler(enabled=True, device=device)
    with profiler.stage("forward"):
        pass
    summary = profiler.summary()

    assert stream_devices == [device]
    assert event_records == [stream, stream]
    assert synchronized == [device]
    assert summary["forward"]["cuda_seconds"] == pytest.approx(0.002)


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


def test_a6_training_runs_a_separate_complete_cycle_pass() -> None:
    class RecordingModel(RotationAwareFusionModel):
        def __init__(self, spec: SkeletonSpec) -> None:
            super().__init__(spec, hidden_channels=8)
            self.frame_lengths: list[int] = []

        def forward(self, face, *args, **kwargs):
            self.frame_lengths.append(face.shape[1])
            return super().forward(face, *args, **kwargs)

    spec = _spec()
    model = RecordingModel(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = LossConfig(
        corruption_recovery_weight=0.0,
        high_consensus_identity_weight=0.0,
        circular_axial_rotation_weight=0.0,
        so3_rotation_weight=0.0,
        trial_bone_length_weight=0.0,
        local_rigidity_weight=0.0,
        adaptive_temporal_acceleration_weight=0.0,
        minimal_residual_weight=0.0,
    )

    train_one_epoch(
        model,
        _loader(count=1, batch_size=1),
        optimizer,
        spec,
        loss_config=config,
        complete_cycle_loader=[_complete_cycle_batch()],
        seed=3,
    )

    assert model.frame_lengths == [5, 129]


def test_a5_training_does_not_consume_complete_cycle_loader() -> None:
    class UnexpectedLoader:
        def __iter__(self):
            raise AssertionError("A5 must not run the complete-cycle ROM pass")

    spec = _spec()
    model = RotationAwareFusionModel(spec, hidden_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = LossConfig(complete_cycle_rom_weight=0.0)

    train_one_epoch(
        model,
        _loader(count=1, batch_size=1),
        optimizer,
        spec,
        loss_config=config,
        complete_cycle_loader=UnexpectedLoader(),
        seed=3,
    )


def test_validation_runs_complete_cycle_rom_on_long_sequences() -> None:
    class RecordingModel(RotationAwareFusionModel):
        def __init__(self, spec: SkeletonSpec) -> None:
            super().__init__(spec, hidden_channels=8)
            self.frame_lengths: list[int] = []
            self.training_modes: list[bool] = []

        def forward(self, face, *args, **kwargs):
            self.frame_lengths.append(face.shape[1])
            self.training_modes.append(self.training)
            return super().forward(face, *args, **kwargs)

    spec = _spec()
    model = RecordingModel(spec)

    validate(
        model,
        _loader(count=1, batch_size=1),
        spec,
        complete_cycle_loader=[_complete_cycle_batch()],
        seed=3,
    )

    assert model.frame_lengths == [5, 129]
    assert model.training_modes == [False, False]
    assert model.training


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
        provenance=_checkpoint_provenance(),
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


def test_validation_is_batch_size_and_order_invariant() -> None:
    spec = _spec()
    torch.manual_seed(13)
    model = RotationAwareFusionModel(spec, hidden_channels=8)

    one = validate(model, _loader(count=5, batch_size=1), spec, loss_config=LossConfig(), seed=17)
    four = validate(model, _loader(count=5, batch_size=4), spec, loss_config=LossConfig(), seed=17)
    reversed_order = validate(model, _loader(count=5, batch_size=4, reverse=True), spec, loss_config=LossConfig(), seed=17)

    assert one == four == reversed_order


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
