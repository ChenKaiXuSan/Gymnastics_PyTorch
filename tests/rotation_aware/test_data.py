import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gymnastics.common.skeletons.mhr70 import mhr_names
from gymnastics.fusion.rotation_aware import data
from gymnastics.fusion.rotation_aware.config import RoleSpec, SkeletonSpec, load_skeleton_spec
from gymnastics.fusion.rotation_aware.schema import PosePairTrial, valid_from_points


def fake_sam3d_loader(_root, _person_id, view):
    frame_ids = (10, 11, 12) if view == "face" else (7, 8, 9)
    return {
        frame_id: np.full((len(mhr_names), 3), frame_id, dtype=np.float32)
        for frame_id in frame_ids
    }


def _cache_trial(value: float = 1.0) -> PosePairTrial:
    points = np.full((1, len(mhr_names), 3), value, dtype=np.float32)
    valid = valid_from_points(points)
    return PosePairTrial(
        face=points,
        side=points,
        valid_face=valid,
        valid_side=valid,
        timestamps=np.array([0.0], dtype=np.float64),
        face_map=np.array([0], dtype=np.int64),
        side_map=np.array([0], dtype=np.int64),
        joint_names=tuple(mhr_names),
        person_id="1",
        trial_id="cycle_000",
        fps=60.0,
    )


def _cache_source() -> dict[str, object]:
    return {
        "alignment_record": "alignment_record_1.json",
        "offset_side_to_face": 0,
        "fps": 60.0,
        "person_id": "1",
    }


@pytest.fixture
def spec():
    return load_skeleton_spec("configs/fusion/skeleton_mhr70.yaml")


def test_load_skeleton_spec_resolves_mhr70_virtual_roles(spec):
    assert spec.joint_names == tuple(mhr_names)
    assert spec.joint_index("left-hip") == 9
    assert spec.role("pelvis").kind == "midpoint"
    assert spec.role("thorax").fallback == ("left-shoulder", "right-shoulder")
    upper_body = {spec.joint_names[index] for index in spec.joint_group("upper_body")}
    assert {"neck", "left-acromion", "right-acromion", "left-wrist", "right-wrist"} <= upper_body
    assert {"left-hip", "right-hip", "left-knee", "right-knee"}.isdisjoint(upper_body)


def test_load_person_trials_uses_split_cycle_boundaries(tmp_path, monkeypatch, spec):
    sam3d_root = tmp_path / "sam3d_body_results"
    split_root = tmp_path / "split_cycle"
    record_path = split_root / "person_1" / "alignment_record_1.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0, "fps": 60.0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 10, "end": 13},
                        "side_video_frames": {"start": 7, "end": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(data, "load_sam3d_world_by_frame", fake_sam3d_loader)

    trials = data.load_person_trials("1", sam3d_root, split_root, spec)

    assert trials[0].trial_id == "cycle_000"
    assert trials[0].face_map.tolist() == [10, 11, 12]
    assert trials[0].side_map.tolist() == [7, 8, 9]
    assert trials[0].fps == 60.0
    assert trials[0].source_metadata == {
        "alignment_record": str(record_path),
        "offset_side_to_face": 0,
        "fps": 60.0,
        "person_id": "1",
        "cycle_index": 0,
        "face_video_frames": {"start": 10, "end": 13},
        "side_video_frames": {"start": 7, "end": 10},
    }


def test_load_person_trials_rejects_empty_alignment_cycles(tmp_path, spec):
    split_root = tmp_path / "split_cycle"
    record_path = split_root / "person_1" / "alignment_record_1.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps({"metadata": {"offset_side_to_face": 0}, "cycles": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="alignment record has no cycles"):
        data.load_person_trials("1", tmp_path / "sam3d_body_results", split_root, spec)


def test_pose_pair_trial_builds_finite_nonzero_valid_mask():
    points = np.array([[[1, 2, 3], [0, 0, 0], [np.nan, 1, 2]]], dtype=np.float32)

    valid = valid_from_points(points)

    assert valid.tolist() == [[True, False, False]]


def test_pose_pair_trial_rejects_non_monotonic_frame_maps(spec):
    points = np.ones((2, len(spec.joint_names), 3), dtype=np.float32)
    valid = valid_from_points(points)

    with pytest.raises(ValueError, match="face_map"):
        PosePairTrial(
            face=points,
            side=points,
            valid_face=valid,
            valid_side=valid,
            timestamps=np.array([0.0, 1 / 60], dtype=np.float32),
            face_map=np.array([10, 9], dtype=np.int32),
            side_map=np.array([7, 8], dtype=np.int32),
            joint_names=spec.joint_names,
            person_id="1",
            trial_id="cycle_000",
            fps=60.0,
        )


def test_pose_pair_trial_rejects_wrong_mhr70_joint_order(spec):
    points = np.ones((1, len(spec.joint_names), 3), dtype=np.float32)
    swapped_names = list(spec.joint_names)
    swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]

    with pytest.raises(ValueError, match="joint_names"):
        PosePairTrial(
            face=points,
            side=points,
            valid_face=valid_from_points(points),
            valid_side=valid_from_points(points),
            timestamps=np.array([0.0], dtype=np.float32),
            face_map=np.array([10], dtype=np.int32),
            side_map=np.array([7], dtype=np.int32),
            joint_names=tuple(swapped_names),
            person_id="1",
            trial_id="cycle_000",
            fps=60.0,
        )


def test_skeleton_spec_normalizes_direct_constructor_collections():
    joint_names = list(mhr_names)
    bones = [[9, 10]]
    required_roles = ["pelvis"]
    skeleton = SkeletonSpec(
        name="mhr70",
        joint_names=joint_names,
        bones=bones,
        roles={"pelvis": RoleSpec(kind="midpoint", joints=("left-hip", "right-hip"))},
        required_roles=required_roles,
    )
    joint_names[0] = "changed"
    bones[0][0] = 0
    required_roles[0] = "changed"

    assert skeleton.joint_names == tuple(mhr_names)
    assert skeleton.bones == ((9, 10),)
    assert skeleton.required_roles == ("pelvis",)


def test_cache_round_trip_preserves_trial_and_metadata(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(data, "load_sam3d_world_by_frame", fake_sam3d_loader)
    split_root = tmp_path / "split_cycle"
    record_path = split_root / "person_1" / "alignment_record_1.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 10, "end": 13},
                        "side_video_frames": {"start": 7, "end": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trials = data.load_person_trials("1", tmp_path / "sam3d_body_results", split_root, spec)

    person_cache = data.write_person_cache(
        trials,
        tmp_path / "cache",
        source_metadata=trials[0].source_metadata,
        config_metadata={"skeleton": spec.name},
    )
    restored, metadata = data.load_cached_trial(person_cache, "cycle_000")

    np.testing.assert_array_equal(restored.face_map, trials[0].face_map)
    np.testing.assert_array_equal(restored.valid_side, trials[0].valid_side)
    assert metadata["source"]["alignment_record"] == str(record_path)
    assert metadata["config"]["skeleton"] == "mhr70"


def test_write_person_cache_publishes_atomically_and_retains_its_guard(
    tmp_path, monkeypatch
):
    lock_seen_while_writing: list[bool] = []
    replace_calls: list[tuple[str, str]] = []
    original_save = data.np.savez_compressed
    original_replace = os.replace
    lock = tmp_path / "cache" / "person_1" / ".publishing.lock"

    def save_with_lock(path, *args, **kwargs):
        lock_seen_while_writing.append(lock.is_file())
        return original_save(path, *args, **kwargs)

    def record_replace(source, destination):
        replace_calls.append((str(source), str(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(data.np, "savez_compressed", save_with_lock)
    monkeypatch.setattr(data, "os", os, raising=False)
    monkeypatch.setattr(data.os, "replace", record_replace)

    person_cache = data.write_person_cache(
        [_cache_trial()],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )

    assert lock_seen_while_writing == [True]
    assert lock.is_file()
    assert (person_cache / "manifest.json").is_file()
    assert any(
        Path(destination) == person_cache / "manifest.json"
        and Path(source) != Path(destination)
        and ".tmp" in Path(source).name
        for source, destination in replace_calls
    )


def test_write_person_cache_publishes_immutable_generation_and_pointer(tmp_path):
    person_cache = data.write_person_cache(
        [_cache_trial()],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )

    pointer = json.loads((person_cache / "manifest.json").read_text(encoding="utf-8"))
    generation = pointer["generation"]
    generation_dir = person_cache / ".generations" / generation
    generation_manifest = json.loads(
        (generation_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert (generation_dir / "cycle_000.npz").is_file()
    assert not (person_cache / "cycle_000.npz").exists()
    assert generation_manifest["generation"] == generation
    assert generation_manifest["trials"] == ["cycle_000"]


def test_load_cached_trial_from_person_directory_resolves_generation_metadata(tmp_path):
    person_cache = data.write_person_cache(
        [_cache_trial(1.0)],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )
    pointer = json.loads((person_cache / "manifest.json").read_text(encoding="utf-8"))

    trial, metadata = data.load_cached_trial(person_cache, "cycle_000")

    assert metadata["generation"] == pointer["generation"]
    assert trial.face[0, 0, 0] == pytest.approx(1.0)


def test_second_writer_is_rejected_by_exclusive_flock(tmp_path, monkeypatch):
    original_save = data.np.savez_compressed
    attempted_second_writer = False

    def save_while_starting_second(path, *args, **kwargs):
        nonlocal attempted_second_writer
        if not attempted_second_writer:
            attempted_second_writer = True
            with pytest.raises(BlockingIOError, match="publication lock"):
                data.write_person_cache(
                    [_cache_trial(2.0)],
                    tmp_path / "cache",
                    source_metadata=_cache_source(),
                    config_metadata={"skeleton": "mhr70"},
                )
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(data.np, "savez_compressed", save_while_starting_second)

    person_cache = data.write_person_cache(
        [_cache_trial(1.0)],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )

    assert attempted_second_writer
    assert (person_cache / ".publishing.lock").is_file()
    data.write_person_cache(
        [_cache_trial(3.0)],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )


def test_guard_file_remains_after_writer_releases_flock(tmp_path):
    person_cache = data.write_person_cache(
        [_cache_trial()],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )

    lock_path = person_cache / ".publishing.lock"
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert lock_path.is_file()


def test_failed_generation_write_keeps_previous_pointer_usable(tmp_path, monkeypatch):
    person_cache = data.write_person_cache(
        [_cache_trial(1.0)],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )
    previous_pointer = json.loads(
        (person_cache / "manifest.json").read_text(encoding="utf-8")
    )

    monkeypatch.setattr(
        data.np,
        "savez_compressed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(OSError, match="write failed"):
        data.write_person_cache(
            [_cache_trial(2.0)],
            tmp_path / "cache",
            source_metadata=_cache_source(),
            config_metadata={"skeleton": "mhr70"},
        )

    current_pointer = json.loads(
        (person_cache / "manifest.json").read_text(encoding="utf-8")
    )
    restored, metadata = data.load_cached_trial(person_cache, "cycle_000")
    assert current_pointer == previous_pointer
    assert metadata["generation"] == previous_pointer["generation"]
    assert restored.face[0, 0, 0] == pytest.approx(1.0)


def test_failed_pointer_publication_cleans_own_temporary_generation_files(
    tmp_path, monkeypatch
):
    person_cache = data.write_person_cache(
        [_cache_trial(1.0)],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )
    previous_pointer = json.loads(
        (person_cache / "manifest.json").read_text(encoding="utf-8")
    )
    original_replace = os.replace

    def fail_pointer_replace(source, destination):
        if Path(destination) == person_cache / "manifest.json":
            raise OSError("pointer replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(data.os, "replace", fail_pointer_replace)

    with pytest.raises(OSError, match="pointer replace failed"):
        data.write_person_cache(
            [_cache_trial(2.0)],
            tmp_path / "cache",
            source_metadata=_cache_source(),
            config_metadata={"skeleton": "mhr70"},
        )

    assert json.loads(
        (person_cache / "manifest.json").read_text(encoding="utf-8")
    ) == previous_pointer
    assert not list(person_cache.glob(".manifest.json.*.tmp"))
    assert sorted(
        path.name
        for path in (person_cache / ".generations").iterdir()
        if path.is_dir()
    ) == [previous_pointer["generation"]]


def test_captured_generation_path_remains_loadable_after_pointer_flip(tmp_path):
    person_cache = data.write_person_cache(
        [_cache_trial(1.0)],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )
    first_pointer = json.loads(
        (person_cache / "manifest.json").read_text(encoding="utf-8")
    )
    first_path = (
        person_cache
        / ".generations"
        / first_pointer["generation"]
        / "cycle_000.npz"
    )

    data.write_person_cache(
        [_cache_trial(2.0)],
        tmp_path / "cache",
        source_metadata=_cache_source(),
        config_metadata={"skeleton": "mhr70"},
    )

    restored, metadata = data.load_cached_trial(first_path)
    assert metadata["generation"] == first_pointer["generation"]
    assert restored.face[0, 0, 0] == pytest.approx(1.0)


def test_load_cached_trial_keeps_legacy_direct_cache_compatible(tmp_path):
    person_cache = tmp_path / "cache" / "person_1"
    person_cache.mkdir(parents=True)
    trial = _cache_trial(3.0)
    np.savez_compressed(
        person_cache / "cycle_000.npz",
        face=trial.face,
        side=trial.side,
        valid_face=trial.valid_face,
        valid_side=trial.valid_side,
        timestamps=trial.timestamps,
        face_map=trial.face_map,
        side_map=trial.side_map,
        joint_names=np.asarray(trial.joint_names),
        person_id=np.asarray(trial.person_id),
        trial_id=np.asarray(trial.trial_id),
        fps=np.asarray(trial.fps, dtype=np.float64),
    )
    legacy_manifest = {"person_id": "1", "trials": ["cycle_000"]}
    (person_cache / "manifest.json").write_text(
        json.dumps(legacy_manifest), encoding="utf-8"
    )

    restored, metadata = data.load_cached_trial(person_cache, "cycle_000")

    assert restored.face[0, 0, 0] == pytest.approx(3.0)
    assert metadata == legacy_manifest
    identity = data.cache_manifest_identity(metadata)
    assert identity["layout"] == "legacy_direct"
    assert identity["generation"] is None
    assert identity["source_hash"] is None
    assert identity["config_hash"] is None
    assert identity["trials"] == ["cycle_000"]


def test_write_person_cache_requires_nonempty_provenance(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(data, "load_sam3d_world_by_frame", fake_sam3d_loader)
    split_root = tmp_path / "split_cycle"
    record_path = split_root / "person_1" / "alignment_record_1.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 10, "end": 13},
                        "side_video_frames": {"start": 7, "end": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trials = data.load_person_trials("1", tmp_path / "sam3d_body_results", split_root, spec)

    with pytest.raises(ValueError, match="source_metadata"):
        data.write_person_cache(
            trials,
            tmp_path / "cache",
            source_metadata={},
            config_metadata={"skeleton": spec.name},
        )
    with pytest.raises(ValueError, match="config_metadata"):
        data.write_person_cache(
            trials,
            tmp_path / "cache",
            source_metadata=trials[0].source_metadata,
            config_metadata={},
        )
