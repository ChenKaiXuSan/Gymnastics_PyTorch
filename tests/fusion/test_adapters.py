"""Adapter tests with injected loaders (no dataset files required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gymnastics.cycle_alignment.cycle_records import cycle_record_path, write_cycle_record
from gymnastics.cycle_alignment.cycles import CycleSpan, DetectionSettings
from gymnastics.common.skeletons.mhr70 import MHR70_INDEX, mhr_names
from gymnastics.fusion.data.freeman import FreeManDataModule, coco17_to_mhr70
from gymnastics.fusion.data.gymnastics import GymnasticsDataModule, concatenate_cycles, reference_from_triangulation
from gymnastics.fusion.data.unity import UnityDataModule, unity22_to_mhr70
from gymnastics.keypoints.schema import PosePairTrial

FPS = 30.0


def _torso_pose(frames: int, *, twist_period: int = 20, seed: int = 0) -> np.ndarray:
    """A plausible MHR70 world pose with hips, thorax, neck, shoulders and limbs."""
    rng = np.random.default_rng(seed)
    pose = np.zeros((frames, 70, 3), dtype=np.float32)
    t = np.arange(frames)
    theta = 0.6 * np.sin(2 * np.pi * t / twist_period)
    anchors = {
        "left-hip": (-0.1, 0.0, 0.0),
        "right-hip": (0.1, 0.0, 0.0),
        "neck": (0.0, 0.55, 0.0),
        "nose": (0.0, 0.65, 0.05),
        "left-knee": (-0.1, -0.45, 0.0),
        "right-knee": (0.1, -0.45, 0.0),
        "left-ankle": (-0.1, -0.9, 0.0),
        "right-ankle": (0.1, -0.9, 0.0),
    }
    for name, value in anchors.items():
        pose[:, MHR70_INDEX[name]] = value
    for side, sign in (("left", -1.0), ("right", 1.0)):
        x = sign * 0.2 * np.cos(theta)
        z = sign * 0.2 * np.sin(theta)
        for joint, height, spread in (("acromion", 0.5, 1.0), ("shoulder", 0.48, 0.95), ("elbow", 0.25, 1.3), ("wrist", 0.0, 1.6)):
            pose[:, MHR70_INDEX[f"{side}-{joint}"], 0] = spread * x
            pose[:, MHR70_INDEX[f"{side}-{joint}"], 1] = height
            pose[:, MHR70_INDEX[f"{side}-{joint}"], 2] = spread * z
    pose += rng.normal(scale=0.002, size=pose.shape).astype(np.float32)
    pose += np.array([1.0, 0.5, 3.0], dtype=np.float32)  # camera offset
    return pose


def _trial(person: str, cycle: int, start_frame: int, frames: int, *, offset: int = -3) -> PosePairTrial:
    face = _torso_pose(frames, seed=cycle)
    side = _torso_pose(frames, seed=cycle + 50) @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    valid = np.ones((frames, 70), dtype=bool)
    face_map = np.arange(start_frame, start_frame + frames, dtype=np.int32)
    return PosePairTrial(
        face=face,
        side=side,
        valid_face=valid,
        valid_side=valid.copy(),
        timestamps=np.arange(frames, dtype=np.float64) / FPS,
        face_map=face_map,
        side_map=face_map + offset,
        joint_names=tuple(mhr_names),
        person_id=person,
        trial_id=f"cycle_{cycle:03d}",
        fps=FPS,
        source_metadata={"alignment_record": "test", "offset_side_to_face": offset, "fps": FPS, "person_id": person},
    )


def test_concatenate_cycles_builds_bounds_and_physical_time():
    trials = [_trial("7", 1, 130, 20), _trial("7", 0, 100, 30)]
    joined, bounds = concatenate_cycles(trials)
    assert bounds == ((0, 30), (30, 50))
    assert joined.face.shape == (50, 70, 3)
    assert joined.timestamps[30] == pytest.approx(30 / FPS)
    assert joined.source_metadata["cycles"] == ("cycle_000", "cycle_001")


def test_reference_from_triangulation_matches_frame_pairs():
    joined, _ = concatenate_cycles([_trial("7", 0, 100, 10)])
    joints = np.full((3, 70, 3), 2.0, dtype=np.float32)
    joints[1, 0] = np.nan

    def loader(person_id, cycle_id):
        assert person_id == "7" and cycle_id == "cycle_000"
        return joints, [(100, 97), (102, 99), (999, 999)]

    reference, valid = reference_from_triangulation(joined, ["cycle_000"], loader)
    # Row 0 -> frame 0, row 1 (NaN at joint 0) -> frame 2, row 2 unmatched.
    assert valid[0].all() and not valid[1].any() and not valid[3].any()
    assert not valid[2, 0] and valid[2, 1:].all()
    assert reference[0, 5, 0] == 2.0 and reference[2, 0, 0] == 0.0


def _write_alignment_records(root: Path, persons, *, cycles: int = 3, length: int = 40, with_mid: bool = True) -> None:
    for person in persons:
        entries = []
        for c in range(cycles):
            fs = 100 + length * c
            face = {"start": fs, "end": fs + length}
            side = {"start": fs - 3, "end": fs - 3 + length}
            if with_mid:
                face["mid"], side["mid"] = fs + 17, fs - 3 + 17
            entries.append({"cycle_index": c, "face_video_frames": face, "side_video_frames": side})
        path = root / f"person_{person}" / f"alignment_record_{person}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": {"person_id": person, "offset_side_to_face": -3, "fps": FPS}, "cycles": entries}))


def test_gymnastics_datamodule_with_injected_loaders(tmp_path: Path):
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": ["1", "2", "3"], "val": ["4"], "test": ["5"]}))
    trials = {p: [_trial(p, c, 100 + 40 * c, 40) for c in range(3)] for p in ("1", "2", "3", "4", "5")}
    _write_alignment_records(tmp_path / "split_cycle", trials)

    def reference_loader(person_id, cycle_id):
        cycle = int(cycle_id.split("_")[1])
        joints = np.ones((40, 70, 3), dtype=np.float32) * (cycle + 1)
        return joints, [(100 + 40 * cycle + i, 97 + 40 * cycle + i) for i in range(40)]

    options = {"fold_json": str(fold), "split_cycle_root": str(tmp_path / "split_cycle")}
    datamodule = GymnasticsDataModule(
        {"name": "gymnastics", "batch_size": 2, "window": {"num_cycles": 2, "samples_per_cycle": 8}, "options": options},
        trial_loader=lambda person: trials[person],
        reference_loader=reference_loader,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    assert datamodule.split.train == ("1", "2", "3") and datamodule.split.test == ("5",)
    sample = datamodule.samples[0]
    assert sample.dataset == "gymnastics" and sample.num_joints == 20 and sample.cycle_bounds == ((0, 40), (40, 80), (80, 120))
    assert sample.cycle_mids == (17, 57, 97)
    assert sample.metadata["canonicalized"] and sample.transform_a is not None
    assert sample.reference_valid.all()
    # Canonical frame: pelvis at the origin.
    pelvis = 0.5 * (sample.view_a[:, sample.joint_names.index("left-hip")] + sample.view_a[:, sample.joint_names.index("right-hip")])
    assert np.abs(pelvis).max() < 1e-4
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["pose_a"].shape[1:] == (16, 20, 3) and batch["phase_valid"].all()
    assert not batch["reference_valid"].any()
    test_batch = next(iter(datamodule.test_dataloader()))
    assert test_batch["reference_valid"].any()
    assert (batch["half_index"] >= 0).all()
    # Records without middles are rejected unless explicitly allowed.
    _write_alignment_records(tmp_path / "no_mid", trials, with_mid=False)
    strict = GymnasticsDataModule({"name": "gymnastics", "options": {**options, "split_cycle_root": str(tmp_path / "no_mid")}}, trial_loader=lambda person: trials[person], reference_loader=reference_loader)
    with pytest.raises(ValueError, match="align cycles"):
        strict.setup("fit")
    lenient = GymnasticsDataModule({"name": "gymnastics", "options": {**options, "split_cycle_root": str(tmp_path / "no_mid"), "require_cycle_mids": False}}, trial_loader=lambda person: trials[person], reference_loader=reference_loader)
    lenient.setup("fit")
    assert lenient.samples[0].cycle_mids == () and lenient.samples[0].cycle_bounds == sample.cycle_bounds


def test_coco17_and_unity22_scatter_into_mhr70():
    coco = np.random.default_rng(0).normal(size=(4, 17, 3)).astype(np.float32)
    coco[2, 0] = np.nan
    points, valid = coco17_to_mhr70(coco)
    assert points.shape == (4, 70, 3) and valid.sum() == 4 * 17 - 1
    assert np.allclose(points[0, MHR70_INDEX["left-wrist"]], coco[0, 9])
    unity = np.random.default_rng(1).normal(size=(3, 22, 3)).astype(np.float32)
    available = np.ones((3, 22), dtype=bool)
    available[1, 9] = False  # LeftHand
    points, valid = unity22_to_mhr70(unity, available)
    assert valid.sum() == 3 * 13 - 1
    assert np.allclose(points[0, MHR70_INDEX["neck"]], unity[0, 4])
    assert not valid[1, MHR70_INDEX["left-wrist"]]


def test_freeman_datamodule_with_injected_loader(tmp_path: Path):
    reference_path = tmp_path / "session.npy"
    np.save(reference_path, np.array([{"keypoints3d_optim": np.ones((60, 17, 3), dtype=np.float32) * 100.0}], dtype=object), allow_pickle=True)

    def session_loader(subject: int):
        trials = []
        for session in range(2):
            trial = _trial(f"{subject:02d}", session, 0, 60, offset=0)
            trials.append((PosePairTrial(**{**trial.__dict__, "trial_id": f"session_{subject}_{session}", "source_metadata": {}}), reference_path))
        return trials

    records_root = tmp_path / "records"
    settings = DetectionSettings(theta_ref=-2.0, theta_ref_mode="manual")
    for subject in (1, 2, 3, 4):
        for session in range(2):
            spans = [CycleSpan(0, 8, 20), CycleSpan(20, 28, 40), CycleSpan(40, 48, 60)] if session == 0 else []
            write_cycle_record(cycle_record_path(records_root, f"{subject:02d}", f"session_{subject}_{session}"), dataset="freeman", subject_id=f"{subject:02d}", sequence_id=f"session_{subject}_{session}", fps=FPS, frames=60, spans=spans, detection=settings, views=("c04", "c07"))
    options = {"subjects": [1, 2, 3, 4], "cycle_records_root": str(records_root)}
    datamodule = FreeManDataModule({"name": "freeman", "batch_size": 2, "window": {"num_cycles": 2, "samples_per_cycle": 8}, "options": options}, session_loader=session_loader)
    datamodule.setup()
    assert len(datamodule.samples) == 8
    sample = datamodule.samples[0]
    assert sample.dataset == "freeman" and sample.subject_id == "01"
    assert sample.cycle_bounds == ((0, 20), (20, 40), (40, 60)) and sample.cycle_mids == (8, 28, 48)
    assert not datamodule.samples[1].has_cycles and datamodule.samples[1].metadata["cycle_record"]
    assert sample.reference is not None and np.allclose(sample.reference[sample.reference_valid], 1.0)  # 100 cm -> 1 m
    assert datamodule.split.train == ("01", "02") and datamodule.split.val == ("03",) and datamodule.split.test == ("04",)
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["phase_valid"].any() and (batch["half_index"][batch["phase_valid"]] >= 0).all()
    # Missing records are an error by default and tolerated when disabled.
    with pytest.raises(FileNotFoundError):
        FreeManDataModule({"name": "freeman", "options": {**options, "cycle_records_root": str(tmp_path / "nowhere")}}, session_loader=session_loader).setup()
    lenient = FreeManDataModule({"name": "freeman", "options": {**options, "cycle_records_root": str(tmp_path / "nowhere"), "require_cycle_records": False}}, session_loader=session_loader)
    lenient.setup()
    assert not lenient.samples[0].has_cycles


def test_unity_datamodule_with_injected_loader(tmp_path: Path):
    def sequence_loader():
        result = []
        for name in ("continuous_left_060_r00", "continuous_right_060_r00", "other_sequence"):
            trial = _trial("unity", 0, 0, 80, offset=0)
            trial = PosePairTrial(**{**trial.__dict__, "trial_id": name, "person_id": "unity", "source_metadata": {}})
            gt = np.ones((80, 22, 3), dtype=np.float32)
            result.append((trial, gt, np.ones((80, 22), dtype=bool)))
        return result

    records_root = tmp_path / "records"
    for name in ("continuous_left_060_r00", "continuous_right_060_r00", "other_sequence"):
        write_cycle_record(cycle_record_path(records_root, name, name), dataset="unity", subject_id=name, sequence_id=name, fps=FPS, frames=80, spans=[CycleSpan(0, 10, 40), CycleSpan(40, 50, 80)], detection=DetectionSettings(), views=("cam0", "cam1"))
    datamodule = UnityDataModule(
        {"name": "unity", "batch_size": 2, "window": {"num_cycles": 2, "samples_per_cycle": 8}, "options": {"cycle_records_root": str(records_root)}},
        sequence_loader=sequence_loader,
    )
    datamodule.setup()
    assert datamodule.split.train == ("continuous_left_060_r00",)
    assert datamodule.split.test == ("continuous_right_060_r00",)
    assert datamodule.split.val == ("other_sequence",)
    sample = datamodule.samples[0]
    assert sample.dataset == "unity" and sample.cycle_bounds == ((0, 40), (40, 80)) and sample.cycle_mids == (10, 50)
    assert sample.reference_valid.sum() == 80 * 13
    batch = next(iter(datamodule.test_dataloader()))
    assert batch["reference_valid"].any() and "clean_a" not in batch


def test_gymnastics_fold_json_restricts_persons(tmp_path: Path):
    fold = tmp_path / "fold_01.json"
    fold.write_text(json.dumps({"train": ["1", "2"], "val": ["3"], "test": ["4"]}))
    trials = {p: [_trial(p, c, 100 + 40 * c, 40) for c in range(2)] for p in ("1", "2", "3", "4", "5")}
    _write_alignment_records(tmp_path / "split_cycle", trials, cycles=2)
    loaded: list[str] = []

    def loader(person):
        loaded.append(person)
        return trials[person]

    datamodule = GymnasticsDataModule(
        {"name": "gymnastics", "fold_json": str(fold), "window": {"num_cycles": 1, "samples_per_cycle": 8}, "options": {"split_cycle_root": str(tmp_path / "split_cycle")}, "attach_reference": False},
        trial_loader=loader,
    )
    datamodule.setup()
    assert sorted(loaded) == ["1", "2", "3", "4"]  # person 5 is not in the fold
    assert datamodule.split.train == ("1", "2") and datamodule.split.val == ("3",) and datamodule.split.test == ("4",)
