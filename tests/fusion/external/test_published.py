"""Strict external baselines: joint mappings, Procrustes averaging, the lifted-trial transform and the protocol evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.skeletons.mhr70 import MHR70_INDEX, mhr_names
from fusion.external.published.fuse import procrustes_average, umeyama
from fusion.external.published.keypoints2d import View2D, load_view, save_view
from fusion.external.published.mapping import COCO17_FROM_MHR70, H36M17_TO_MHR70, coco17_to_h36m17_2d, h36m17_to_mhr70, mhr70_to_coco17_2d
from fusion.external.published.transform import LiftedTrialTransform
from fusion.external.published.videopose3d import normalize_screen_coordinates
from fusion.keypoints.schema import PosePairTrial


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_joint_mappings_are_consistent():
    assert len(COCO17_FROM_MHR70) == 17 and COCO17_FROM_MHR70[0] == MHR70_INDEX["nose"] and COCO17_FROM_MHR70[16] == MHR70_INDEX["right-ankle"]
    points = np.random.default_rng(0).normal(size=(4, 70, 2)).astype(np.float32)
    valid = np.ones((4, 70), dtype=bool)
    valid[1, MHR70_INDEX["left-wrist"]] = False
    coco, ok = mhr70_to_coco17_2d(points, valid)
    assert coco.shape == (4, 17, 2) and np.allclose(coco[:, 9], points[:, MHR70_INDEX["left-wrist"]]) and not ok[1, 9] and ok[0].all()
    h36m = np.random.default_rng(1).normal(size=(3, 17, 3)).astype(np.float32)
    pose, pose_valid = h36m17_to_mhr70(h36m)
    assert pose.shape == (3, 70, 3) and pose_valid.sum() == 3 * len(H36M17_TO_MHR70)
    assert np.allclose(pose[:, MHR70_INDEX["right-knee"]], h36m[:, 2]) and np.allclose(pose[:, MHR70_INDEX["neck"]], h36m[:, 8]) and np.allclose(pose[:, MHR70_INDEX["nose"]], h36m[:, 9])
    assert not pose_valid[:, MHR70_INDEX["left-eye"]].any()  # no H36M counterpart
    h2d, h2d_valid = coco17_to_h36m17_2d(coco, ok)
    assert np.allclose(h2d[:, 0], 0.5 * (coco[:, 11] + coco[:, 12])) and np.allclose(h2d[0, 13], coco[0, 9]) and not h2d_valid[1, 13] and h2d_valid[0].all()
    assert np.allclose(normalize_screen_coordinates(np.array([[[0.0, 0.0], [1920.0, 1080.0]]]), 1920, 1080), [[[-1.0, -0.5625], [1.0, 0.5625]]])


def test_procrustes_average_recovers_a_rotated_scaled_copy():
    rng = np.random.default_rng(2)
    pose_a = rng.normal(size=(5, 12, 3)).astype(np.float32)
    rotation = _rotation(np.array([0.3, 1.0, 0.2]), 0.7)
    pose_b = (1.8 * (rotation @ pose_a.reshape(-1, 3).T).T + np.array([0.5, -0.2, 3.0])).reshape(5, 12, 3).astype(np.float32)
    valid = np.ones((5, 12), dtype=bool)
    scale, rot, trans = umeyama(pose_b[0].astype(np.float64), pose_a[0].astype(np.float64))
    assert abs(scale - 1 / 1.8) < 1e-4
    fused, fused_valid = procrustes_average(pose_a, valid, pose_b, valid)
    assert fused_valid.all() and np.allclose(fused, pose_a, atol=1e-4)
    # A joint valid only in view B keeps view B (aligned); a frame with too few shared joints falls back to A.
    valid_a = valid.copy()
    valid_a[0, 3] = False
    fused, fused_valid = procrustes_average(pose_a, valid_a, pose_b, valid)
    assert fused_valid[0, 3] and np.allclose(fused[0, 3], pose_a[0, 3], atol=1e-3)
    sparse = np.zeros_like(valid)
    sparse[:, :2] = True
    fused, fused_valid = procrustes_average(pose_a, valid, pose_b, sparse)
    assert np.allclose(fused[:, 2:], pose_a[:, 2:]) and fused_valid.all()


def _trial(frames: int = 30, offset: int = 2) -> PosePairTrial:
    rng = np.random.default_rng(3)
    face = rng.normal(size=(frames, 70, 3)).astype(np.float32)
    side = rng.normal(size=(frames, 70, 3)).astype(np.float32)
    valid = np.ones((frames, 70), dtype=bool)
    face_map = np.arange(10, 10 + frames, dtype=np.int32)
    return PosePairTrial(face=face, side=side, valid_face=valid, valid_side=valid.copy(), timestamps=np.arange(frames) / 30.0, face_map=face_map, side_map=face_map + offset, joint_names=tuple(mhr_names), person_id="7", trial_id="cycle_000", fps=30.0, source_metadata={"offset_side_to_face": offset})


class _Source:
    """Two synthetic videos: view A frames 0..99, view B frames 0..119, pixel keypoints from a planted 3D body."""

    def __init__(self) -> None:
        rng = np.random.default_rng(4)
        self.calls = 0
        self.videos = {}
        for name, frames in (("a", 100), ("b", 120)):
            points = rng.uniform(100, 900, size=(frames, 70, 2)).astype(np.float32)
            self.videos[name] = View2D(frame_ids=np.arange(frames, dtype=np.int64), points=points, valid=np.ones((frames, 70), dtype=bool), width=1080, height=1920, name=f"synthetic/{name}")

    def views(self, trial):
        self.calls += 1
        return self.videos["a"], self.videos["b"]


def _fake_lifter(coco: np.ndarray, width: int, height: int) -> np.ndarray:
    """Deterministic 'lifter': H36M joints built from the 2D input so frames are traceable."""
    out = np.zeros((coco.shape[0], 17, 3), dtype=np.float32)
    out[:, :, :2] = np.repeat(coco[:, :1, :], 17, axis=1) / width + np.arange(17)[None, :, None] * 0.01
    out[:, :, 2] = np.arange(coco.shape[0])[:, None] * 1e-3
    return out


def test_lifted_trial_transform_samples_each_view_on_its_own_frames(tmp_path: Path):
    source = _Source()
    transform = LiftedTrialTransform(_fake_lifter, source, mode="per_view", cache_dir=tmp_path, method="fake")
    trial = _trial()
    out = transform(trial)
    assert out.face.shape == trial.face.shape and out.face_map is trial.face_map or np.array_equal(out.face_map, trial.face_map)
    # Frame 10 of view A -> lifted depth 10e-3; frame 12 of view B (offset 2) -> 12e-3.
    assert abs(out.face[0, MHR70_INDEX["right-hip"], 2] - 0.010) < 1e-6 and abs(out.side[0, MHR70_INDEX["right-hip"], 2] - 0.012) < 1e-6
    assert out.valid_face[:, MHR70_INDEX["right-hip"]].all() and not out.valid_face[:, MHR70_INDEX["left-eye"]].any()
    assert out.source_metadata["external_method"] == "fake" and out.source_metadata["external_mode"] == "per_view"
    # Whole videos are lifted once and cached: a second trial of the same person reuses them.
    cached = list((tmp_path / "fake").glob("*.npz"))
    assert len(cached) == 2
    transform(_trial(frames=20))
    assert len(list((tmp_path / "fake").glob("*.npz"))) == 2
    fused = LiftedTrialTransform(_fake_lifter, source, mode="procrustes_average", cache_dir=tmp_path, method="fake")(trial)
    assert np.array_equal(fused.face, fused.side) and fused.valid_face[:, MHR70_INDEX["left-knee"]].all()
    with pytest.raises(ValueError):
        LiftedTrialTransform(_fake_lifter, source, mode="bogus")


def test_view2d_round_trip_and_selection(tmp_path: Path):
    view = View2D(frame_ids=np.array([0, 1, 5]), points=np.ones((3, 70, 2), np.float32) * np.array([1, 2, 3])[:, None, None], valid=np.ones((3, 70), bool), width=10, height=20, name="x")
    save_view(tmp_path / "v.npz", view)
    loaded = load_view(tmp_path / "v.npz", name="x")
    assert loaded.width == 10 and np.array_equal(loaded.frame_ids, view.frame_ids)
    points, valid = loaded.select([1, 3, 5])
    assert np.allclose(points[0], 2.0) and np.allclose(points[2], 3.0) and valid[0].all() and not valid[1].any()


def test_evaluate_fold_runs_the_transform_through_the_freeman_datamodule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The evaluator composes the real data config, injects the transform and scores view A."""
    from cycle_alignment.cycle_records import cycle_record_path, write_cycle_record
    from cycle_alignment.cycles import CycleSpan, DetectionSettings
    from fusion.data import freeman as freeman_module
    from fusion.external.published import evaluate as evaluate_module
    import json

    frames = 60
    # A smooth reference motion (phase resampling interpolates between frames, so a
    # frame-wise random pose would not survive the canonical frame; real motion does).
    rest = np.random.default_rng(5).normal(size=(17, 3)).astype(np.float32) * 30.0
    sway = np.sin(np.linspace(0, 2 * np.pi, frames, dtype=np.float32))[:, None, None] * np.array([3.0, 1.0, 2.0], dtype=np.float32)
    reference = (rest[None] + sway).astype(np.float32)  # cm
    reference_path = tmp_path / "ref.npy"
    np.save(reference_path, np.array([{"keypoints3d_optim": reference}], dtype=object), allow_pickle=True)
    records_root = tmp_path / "records"
    settings = DetectionSettings(theta_ref=-2.0, theta_ref_mode="manual")
    for subject in (1, 2, 3):
        write_cycle_record(cycle_record_path(records_root, f"{subject:02d}", f"20220618_s{subject}_subj{subject:02d}"), dataset="freeman", subject_id=f"{subject:02d}", sequence_id=f"20220618_s{subject}_subj{subject:02d}", fps=30.0, frames=frames, spans=[CycleSpan(0, 8, 20), CycleSpan(20, 28, 40), CycleSpan(40, 48, 60)], detection=settings, views=("c04", "c07"))

    def session_loader(subject: int):
        trial = _trial(frames=frames, offset=0)
        return [(PosePairTrial(**{**trial.__dict__, "person_id": f"{subject:02d}", "trial_id": f"20220618_s{subject}_subj{subject:02d}", "source_metadata": {"subject_id": subject}}), reference_path)]

    original = freeman_module.FreeManDataModule.__init__

    def patched_init(self, config, *, session_loader=None, trial_transform=None):
        original(self, config, session_loader=globals()["_loader"], trial_transform=trial_transform)

    globals()["_loader"] = session_loader
    monkeypatch.setattr(freeman_module.FreeManDataModule, "__init__", patched_init)
    fold = tmp_path / "fold_01.json"
    fold.write_text(json.dumps({"name": "fold_01", "dataset": "freeman", "protocol": "subject_disjoint", "train": ["01"], "val": ["02"], "test": ["03"]}), encoding="utf-8")

    # Transform that plants the reference (converted to metres, MHR70 layout) into view A: error must be ~0.
    from fusion.data.freeman import coco17_to_mhr70

    def perfect(trial):
        from dataclasses import replace

        pose, valid = coco17_to_mhr70(reference * 0.01)
        return replace(trial, face=pose, side=pose, valid_face=valid, valid_side=valid)

    extra = [f"data.options.cycle_records_root={records_root}", "data.options.actions=null", "data.options.min_cycles=0"]
    result = evaluate_module.evaluate_fold("freeman", fold, perfect, extra_overrides=extra)
    assert result["test_subjects"] == ["03"] and result["test_windows"] > 0 and result["pa_mpjpe"] < 2e-3
    assert "left-knee" in result["joint_names"] and "nose" in result["joint_names"]
    noisy = evaluate_module.evaluate_fold("freeman", fold, None, extra_overrides=extra)
    assert noisy["pa_mpjpe"] > 0.05


def test_named_to_h36m17_matches_coco_mapping_and_fill_interpolates():
    from fusion.external.published.mapping import COCO17_NAMES, fill_missing_joints, named_to_h36m17

    rng = np.random.default_rng(1)
    coco = rng.normal(size=(5, 17, 3)).astype(np.float32)
    expected, expected_valid = coco17_to_h36m17_2d(coco)
    # The same joints under their MHR70 names in a different order, plus an unrelated joint.
    names = tuple(reversed(COCO17_NAMES)) + ("left-heel",)
    points = np.concatenate([coco[:, ::-1], np.zeros((5, 1, 3), np.float32)], axis=1)
    got, got_valid = named_to_h36m17(points, None, names)
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert (got_valid == expected_valid).all()
    # Without eyes/ears (major joints) every H36M joint is still defined.
    major = tuple(n for n in COCO17_NAMES if "eye" not in n and "ear" not in n)
    got, got_valid = named_to_h36m17(coco[:, [COCO17_NAMES.index(n) for n in major]], None, major)
    assert got_valid.all() and np.allclose(got, expected)
    # Missing joints are linearly interpolated in time, edges held.
    seq = np.stack([np.full((2, 2), t, np.float32) for t in range(6)])
    valid = np.ones((6, 2), bool)
    valid[2:4, 0] = False
    valid[0, 1] = False
    filled = fill_missing_joints(seq, valid)
    assert np.allclose(filled[2:4, 0, 0], [2.0, 3.0]) and np.allclose(filled[0, 1], 1.0)


def test_canonpose_vectorised_camera_loss_matches_the_released_loop():
    import torch

    from fusion.external.published.canonpose import camera_consistency_loss, loss_weighted_rep_no_scale, within_subject_permutation

    torch.manual_seed(0)
    b, n_cam = 12, 2
    subjects = torch.tensor([0, 1, 0, 2, 1, 0, 3, 1, 2, 0, 4, 1])
    inp = torch.randn(b, n_cam, 32)
    conf = (torch.rand(b, n_cam, 16) > 0.2).float()
    rot = torch.linalg.qr(torch.randn(b, n_cam, 3, 3))[0]
    rot_poses = torch.randn(b, n_cam, 48)
    generator = torch.Generator().manual_seed(3)
    perm, multiple = within_subject_permutation(subjects, generator)
    assert (subjects[perm] == subjects).all() and sorted(perm.tolist()) == list(range(b))
    assert multiple.tolist() == [s in (0, 1, 2) for s in subjects.tolist()]
    for c_cnt in range(n_cam):
        coi = np.delete(np.arange(n_cam), c_cnt)
        # The released loop with the same within-subject permutation.
        relative = rot[:, coi].matmul(rot[:, [c_cnt]].permute(0, 1, 3, 2))
        expected = 0.0
        for subject in subjects.unique():
            mask = subjects == subject
            if int(mask.sum()) > 1:
                local = perm[mask]
                local_index = torch.tensor([torch.nonzero(mask).flatten().tolist().index(int(i)) for i in local])
                shuffled = relative[mask][local_index].matmul(rot_poses[mask].reshape(-1, n_cam, 3, 16)[:, c_cnt : c_cnt + 1].repeat(1, n_cam - 1, 1, 1)).reshape(-1, n_cam - 1, 48)
                expected = expected + loss_weighted_rep_no_scale(inp[mask][:, coi].reshape(-1, 32), shuffled.reshape(-1, 48), conf[mask][:, coi].reshape(-1, 16))
        got = camera_consistency_loss(inp, conf, rot, rot_poses, subjects, c_cnt, coi, perm, multiple)
        assert torch.isclose(got, torch.as_tensor(expected), rtol=1e-5, atol=1e-6)
