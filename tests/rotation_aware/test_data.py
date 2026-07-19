import json

import numpy as np
import pytest

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware import data
from fuse.rotation_aware.config import RoleSpec, SkeletonSpec, load_skeleton_spec
from fuse.rotation_aware.schema import PosePairTrial, valid_from_points


def fake_sam3d_loader(_root, _person_id, view):
    frame_ids = (10, 11, 12) if view == "face" else (7, 8, 9)
    return {
        frame_id: np.full((len(mhr_names), 3), frame_id, dtype=np.float32)
        for frame_id in frame_ids
    }


@pytest.fixture
def spec():
    return load_skeleton_spec("configs/fuse/skeleton_mhr70.yaml")


def test_load_skeleton_spec_resolves_mhr70_virtual_roles(spec):
    assert spec.joint_names == tuple(mhr_names)
    assert spec.joint_index("left-hip") == 9
    assert spec.role("pelvis").kind == "midpoint"
    assert spec.role("thorax").fallback == ("left-shoulder", "right-shoulder")


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
