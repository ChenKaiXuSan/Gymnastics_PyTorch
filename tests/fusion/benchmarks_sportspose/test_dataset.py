"""SportsPose release access, view selection and the SAM3D cache on a synthetic release."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from fusion.benchmarks.sportspose.cli import read_selected_views
from fusion.benchmarks.sportspose.dataset import camera_azimuths, discover_clips, group_clips, load_calibration, load_reference, load_timing, select_views
from fusion.benchmarks.sportspose.sam3d import cache_path, derived_path, infer_clips, load_derived_prediction, load_prediction, upright
from fusion.benchmarks.sportspose.schema import NUM_CAMERAS, SelectedViews
from fusion.benchmarks.sportspose.trials import build_sequence_trial, load_clip_predictions

FRAMES = 27


def _look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV world-to-camera (R, T) for a camera at ``position`` looking at ``target``."""
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z])
    return rotation, -rotation @ position


def _reference(frames: int = FRAMES) -> np.ndarray:
    """COCO17 subject standing at the origin, facing +x, z up, swinging the right wrist."""
    points = np.zeros((frames, 17, 3), dtype=np.float32)
    points[:, 5] = (0.0, 0.2, 1.4)  # left shoulder (subject's left = +y when facing +x with z up)
    points[:, 6] = (0.0, -0.2, 1.4)
    points[:, 11] = (0.0, 0.1, 0.9)
    points[:, 12] = (0.0, -0.1, 0.9)
    angle = np.linspace(-1.0, 1.0, frames)
    points[:, 10] = np.stack([0.4 * np.cos(angle), -0.3 + 0.4 * np.sin(angle), np.full(frames, 1.2)], axis=1)
    points[:, 9] = (0.1, 0.4, 1.0)
    return points


def make_release(root: Path, *, subjects=("S00", "S01"), days=("indoors",), activities=("tennis",), clips_per=2) -> Path:
    up = np.array([0.0, 0.0, 1.0])
    for day in days:
        for subject in subjects:
            subject_dir = root / "data" / day / subject
            subject_dir.mkdir(parents=True)
            cameras = []
            azimuths = np.radians([0, 45, 90, 135, 180, 225, 270])  # cam0 in front of the subject
            for k, azimuth in enumerate(azimuths):
                position = np.array([3.0 * np.cos(azimuth), 3.0 * np.sin(azimuth), 1.3])
                rotation, translation = _look_at(position, np.array([0.0, 0.0, 1.0]), up)
                cameras.append({"R": rotation, "T": translation, "f": np.array([800.0, 800.0]), "c": np.array([320.0, 240.0]), "k": np.zeros(8), "tpr": None, "P": None, "A": None})
            (subject_dir / "calib.pkl").write_bytes(pickle.dumps({"calibration": cameras, "numtimesrot90clockwise": [1] * NUM_CAMERAS}))
            for activity in activities:
                activity_dir = subject_dir / activity
                activity_dir.mkdir()
                for index in range(clips_per):
                    clip = f"{activity}{index + 21:04d}"
                    np.save(activity_dir / f"{clip}.npy", _reference())
                    video_dir = root / "videos" / day / subject / f"Video_{subject}_{clip}"
                    video_dir.mkdir(parents=True)
                    (activity_dir / f"{clip}_timing.pkl").write_bytes(pickle.dumps({"video_index": np.tile(np.arange(FRAMES), (NUM_CAMERAS, 1)), "times_ms": np.tile(np.arange(FRAMES) * 11, (NUM_CAMERAS, 1)), "video_path": np.array([day, subject, f"Video_{subject}_{clip}"])}))
                    for k in range(NUM_CAMERAS):
                        writer = cv2.VideoWriter(str(video_dir / f"CAM{k}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 90.0, (64, 48))
                        for frame in range(FRAMES):
                            image = np.full((48, 64, 3), frame * 9 % 255, dtype=np.uint8)
                            writer.write(image)
                        writer.release()
    return root


class FakeEstimator:
    def __init__(self) -> None:
        self.calls = 0

    def process_one_image(self, *, img, bboxes=None):
        del bboxes
        self.calls += 1
        assert img.shape[:2] == (64, 48)  # upright: rotated once clockwise
        pose = np.ones((70, 3), dtype=np.float32) * (1.0 + img[0, 0, 0] / 255.0)
        return [{"bbox": np.array([0, 0, 10, 10], dtype=np.float32), "pred_keypoints_3d": pose, "pred_keypoints_2d": pose[:, :2]}]


def test_discover_and_select_views(tmp_path: Path):
    root = make_release(tmp_path / "SportsPose")
    clips = discover_clips(root)
    assert [(c.subject, c.day, c.activity, c.clip_id) for c in clips] == [("S00", "indoors", "tennis", "tennis0021"), ("S00", "indoors", "tennis", "tennis0022"), ("S01", "indoors", "tennis", "tennis0021"), ("S01", "indoors", "tennis", "tennis0022")]
    assert clips[0].subject_key == "S00" and clips[0].sequence_key == "indoors_tennis" and clips[0].frames == FRAMES
    assert discover_clips(root, subjects=["S01"])[0].subject == "S01"
    groups = group_clips(clips)
    assert list(groups) == [("S00", "indoors_tennis"), ("S01", "indoors_tennis")] and len(groups[("S00", "indoors_tennis")]) == 2
    cameras = load_calibration(root / "data" / "indoors" / "S00")
    azimuths = camera_azimuths(cameras, [load_reference(c) for c in groups[("S00", "indoors_tennis")]])
    assert abs(azimuths["cam0"]) < 1.0 and abs(abs(azimuths["cam2"]) - 90.0) < 1.0 and abs(abs(azimuths["cam4"]) - 180.0) < 1.0
    views = select_views("S00", "indoors_tennis", groups[("S00", "indoors_tennis")], cameras)
    assert views.view_a == "cam0" and views.view_b in {"cam2", "cam6"} and abs(views.separation_deg - 90.0) < 1.0
    assert SelectedViews.from_dict(views.to_dict()) == views
    with pytest.raises(ValueError):
        select_views("S00", "indoors_tennis", groups[("S00", "indoors_tennis")], {k: v for k, v in cameras.items() if k in {"cam3", "cam4", "cam5"}})
    timing = load_timing(clips[0])
    assert timing["video_index"].shape == (NUM_CAMERAS, FRAMES)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert upright(frame, cameras["cam0"]).shape == (64, 48, 3)


def test_infer_cache_and_sequence_trial(tmp_path: Path):
    root = make_release(tmp_path / "SportsPose")
    config = tmp_path / "sam3d.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    clips = discover_clips(root, subjects=["S00"])
    cameras = load_calibration(root / "data" / "indoors" / "S00")
    estimator = FakeEstimator()
    cache_root = tmp_path / "cache"
    summaries = infer_clips(clips, {("S00", "indoors_tennis"): ("cam0", "cam2")}, {("indoors", "S00"): cameras}, cache_root=cache_root, config_path=config, device=0, frame_stride=3, estimator_factory=lambda *_: estimator)
    assert len(summaries) == 4 and not any(s.reused for s in summaries) and all(s.frames == 9 for s in summaries)
    assert estimator.calls == 4 * 9
    prediction = load_prediction(cache_path(cache_root, clips[0], "cam0"))
    assert prediction.frame_ids.tolist() == list(range(0, FRAMES, 3)) and prediction.points_3d.shape == (9, 70, 3) and prediction.valid_3d.all()
    # Second run reuses every cache entry; a changed config invalidates it.
    again = infer_clips(clips, {("S00", "indoors_tennis"): ("cam0", "cam2")}, {("indoors", "S00"): cameras}, cache_root=cache_root, config_path=config, device=0, frame_stride=3, estimator_factory=lambda *_: estimator)
    assert all(s.reused for s in again) and estimator.calls == 4 * 9
    config.write_text("model: {other: 1}\n", encoding="utf-8")
    assert not any(s.reused for s in infer_clips(clips[:1], {("S00", "indoors_tennis"): ("cam0", "cam2")}, {("indoors", "S00"): cameras}, cache_root=cache_root, config_path=config, device=0, frame_stride=3, estimator_factory=lambda *_: estimator))

    views = SelectedViews(subject_key="S00", sequence_key="indoors_tennis", view_a="cam0", view_b="cam2", azimuth_a_deg=0.0, azimuth_b_deg=90.0, separation_deg=90.0)
    predictions = [load_clip_predictions(clip, views, cache_root=cache_root) for clip in clips]
    trial, bounds, reference = build_sequence_trial(clips, predictions, views, reference=True)
    assert bounds == ((0, 9), (9, 18)) and trial.face.shape == (18, 70, 3) and trial.fps == 30.0
    assert trial.person_id == "S00" and trial.trial_id == "indoors_tennis" and list(trial.source_metadata["clips"]) == ["tennis0021", "tennis0022"]
    assert reference.shape == (18, 17, 3) and np.allclose(reference[:9], _reference()[::3])
    # selected_views.json round trip
    payload = {"groups": [views.to_dict()], "failures": []}
    path = tmp_path / "selected_views.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_selected_views(path)[("S00", "indoors_tennis")] == views


def test_derived_per_video_cache_is_read_at_the_requested_frames(tmp_path: Path):
    root = make_release(tmp_path / "SportsPose", subjects=("S00",), clips_per=1)
    clip = discover_clips(root)[0]
    cameras = load_calibration(root / "data" / "indoors" / "S00")
    derived_root = tmp_path / "derived"
    # Two persons on every frame; rank 0 is the one we keep. Frame 6 has no detection.
    frames = [f for f in range(FRAMES) if f != 6]
    rows = [(f, rank) for f in frames for rank in (0, 1)]
    points = np.stack([np.full((70, 3), f + 100 * rank, dtype=np.float32) for f, rank in rows])
    path = derived_path(derived_root, clip, cameras["cam0"].index)
    path.parent.mkdir(parents=True)
    np.savez(path, frame_ids=np.array([r[0] for r in rows]), person_rank=np.array([r[1] for r in rows]), points3d=points, points2d=points[:, :, :2], n_frames=FRAMES)
    frame_ids = np.arange(0, FRAMES, 3)
    prediction = load_derived_prediction(derived_root, clip, cameras["cam0"], frame_ids, frame_ids)
    assert prediction.points_3d.shape == (9, 70, 3) and prediction.failed_frames == (6,)
    assert np.allclose(prediction.points_3d[1], 3.0) and not prediction.valid_3d[2].any() and prediction.valid_3d[1].all()
    with pytest.raises(FileNotFoundError):
        load_derived_prediction(derived_root, clip, cameras["cam1"], frame_ids, frame_ids)
    views = SelectedViews(subject_key="S00", sequence_key="indoors_tennis", view_a="cam0", view_b="cam1", azimuth_a_deg=0.0, azimuth_b_deg=45.0, separation_deg=45.0)
    with pytest.raises(FileNotFoundError):
        load_clip_predictions(clip, views, derived_root=derived_root)
    np.savez(derived_path(derived_root, clip, cameras["cam1"].index), frame_ids=np.array([r[0] for r in rows]), person_rank=np.array([r[1] for r in rows]), points3d=points, points2d=points[:, :, :2], n_frames=FRAMES)
    a, b = load_clip_predictions(clip, views, derived_root=derived_root, frame_stride=3)
    assert a.frame_ids.tolist() == b.frame_ids.tolist() == list(range(0, FRAMES, 3))

