"""Fit3D release scanning, view selection, repetitions and the SAM3D cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fusion.benchmarks.fit3d.dataset import (
    body_azimuth,
    discover_sequences,
    facing_azimuth,
    load_camera,
    load_reference,
    load_repetitions,
    repetition_bounds,
    select_views,
)
from fusion.benchmarks.fit3d.sam3d import cache_path, cached_frames, load_prediction
from fusion.benchmarks.fit3d.schema import CAMERAS, JOINTS3D_25_TO_MHR70, Fit3DCamera
from fusion.benchmarks.fit3d.trials import build_trial, common_frames, load_sequence, scale_bounds

FRAMES = 40
# Four cameras around the subject, as in the release (about +-25 and +-157 degrees).
AZIMUTHS = {"50591643": 157.0, "58860488": -155.0, "60457274": -22.0, "65906101": 24.0}


def _camera_json(azimuth_deg: float, radius: float = 4.0) -> dict:
    """Extrinsics of a camera at ``azimuth_deg`` looking at the origin (z up)."""
    angle = np.radians(azimuth_deg)
    center = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.2])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])  # world -> camera rows
    return {
        "extrinsics": {"R": rotation.tolist(), "T": center.tolist()},
        "intrinsics_wo_distortion": {"f": [1000.0, 1000.0], "c": [450.0, 450.0]},
    }


def _reference(frames: int = FRAMES) -> np.ndarray:
    """A standing subject facing +x, with the hips on the y axis."""
    points = np.zeros((frames, 25, 3), dtype=np.float32)
    t = np.arange(frames)
    points[:, 0] = (0.0, 0.0, 0.95)                      # pelvis
    points[:, 1] = (0.0, 0.1, 0.9)                       # left hip
    points[:, 4] = (0.0, -0.1, 0.9)                      # right hip
    points[:, 8] = (0.0, 0.0, 1.4)                       # neck
    points[:, 9] = (0.05, 0.0, 1.5)                      # nose
    points[:, 11] = (0.0, 0.2, 1.35)                     # left shoulder
    points[:, 14] = (0.0, -0.2, 1.35)                    # right shoulder
    points[:, 13] = np.stack([0.3 + 0.0 * t, 0.2 + 0.0 * t, 1.0 + 0.2 * np.sin(2 * np.pi * t / 20)], axis=-1)  # left wrist
    points[:, 16] = np.stack([0.3 + 0.0 * t, -0.2 + 0.0 * t, 1.0 + 0.2 * np.sin(2 * np.pi * t / 20)], axis=-1)
    for column, height in ((2, 0.5), (5, 0.5), (3, 0.1), (6, 0.1)):
        points[:, column] = (0.0, 0.1 if column in (2, 3) else -0.1, height)
    return points


def make_release(root: Path, *, subjects=("s03",), actions=("squat", "pushup"), frames: int = FRAMES, reps=(0, 12, 24, 36)) -> Path:
    """A miniature Fit3D release with four cameras, a reference and repetitions."""
    root = Path(root)
    for subject in subjects:
        subject_dir = root / "train" / subject
        (subject_dir / "joints3d_25").mkdir(parents=True, exist_ok=True)
        for action in actions:
            (subject_dir / "joints3d_25" / f"{action}.json").write_text(json.dumps({"joints3d_25": _reference(frames).tolist()}), encoding="utf-8")
            for camera in CAMERAS:
                path = subject_dir / "camera_parameters" / camera / f"{action}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(_camera_json(AZIMUTHS[camera])), encoding="utf-8")
        (subject_dir / "rep_ann.json").write_text(json.dumps({actions[0]: list(reps)}), encoding="utf-8")
    return root


def make_cache(root: Path, release: Path, *, subjects=("s03",), actions=("squat", "pushup"), frames: int = FRAMES) -> Path:
    """SAM3D predictions for every camera of the miniature release."""
    root = Path(root)
    rng = np.random.default_rng(0)
    for subject in subjects:
        for action in actions:
            for camera in CAMERAS:
                path = root / "train" / subject / camera / f"{action}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                points_3d = rng.normal(scale=0.3, size=(frames, 70, 3)).astype(np.float32)
                np.savez(
                    path,
                    frame_ids=np.arange(frames, dtype=np.int64),
                    person_rank=np.zeros(frames, dtype=np.int32),
                    points3d=points_3d,
                    points2d=points_3d[..., :2] * 100.0 + 450.0,
                    valid3d=np.ones((frames, 70), dtype=bool),
                    valid2d=np.ones((frames, 70), dtype=bool),
                    n_frames=np.int64(frames),
                    rot90=np.int64(0),
                    batched=np.int64(1),
                    source=str(path),
                )
    return root


@pytest.fixture
def release(tmp_path: Path) -> tuple[Path, Path]:
    return make_release(tmp_path / "Fit3D"), make_cache(tmp_path / "cache", tmp_path / "Fit3D")


def test_discovery_skips_subjects_without_a_reference(release, tmp_path: Path):
    root, _ = release
    (root / "train" / "s02" / "camera_parameters").mkdir(parents=True)  # a test subject: no joints3d_25
    sequences = discover_sequences(root)
    assert [(s.subject, s.action) for s in sequences] == [("s03", "pushup"), ("s03", "squat")]
    assert sequences[0].frames == FRAMES and sequences[0].cameras == CAMERAS
    assert discover_sequences(root, actions=["squat"])[0].action == "squat"
    with pytest.raises(FileNotFoundError):
        discover_sequences(root, split="validation")


def test_camera_projection_and_azimuth(release):
    root, _ = release
    sequence = discover_sequences(root, actions=["squat"])[0]
    reference = load_reference(sequence.reference_path)
    for name, azimuth in AZIMUTHS.items():
        camera = load_camera(sequence, name)
        assert body_azimuth(camera, reference) == pytest.approx(azimuth, abs=1.0)
        uv, depth = camera.project(reference[0])
        assert depth.min() > 0 and np.isfinite(uv).all()
    # The subject faces +x (hips along y), so the frontal cameras are the +-25 degree ones.
    assert facing_azimuth(reference) == pytest.approx(0.0, abs=1.0)
    with pytest.raises(ValueError):
        Fit3DCamera(camera_id="x", rotation=np.zeros((3, 3)), center=np.zeros(3), focal=np.ones(2), principal_point=np.ones(2))


def test_view_selection_prefers_the_frontal_camera_and_the_target_separation(release):
    root, _ = release
    sequence = discover_sequences(root, actions=["squat"])[0]
    reference = load_reference(sequence.reference_path)
    views = select_views(sequence, reference, target_separation_deg=90.0)
    assert views.view_a in {"60457274", "65906101"}                     # the frontal pair
    assert views.separation_deg == pytest.approx(132.0, abs=3.0)        # the reachable optimum
    assert views.subject_key == "s03" and views.sequence_key == "squat"
    # With a target of 180 degrees the opposite camera wins instead.
    opposite = select_views(sequence, reference, target_separation_deg=180.0)
    assert opposite.separation_deg == pytest.approx(179.0, abs=3.0)


def test_repetition_bounds_are_the_annotated_marks(release):
    root, _ = release
    marks = load_repetitions(root / "train" / "s03")
    assert marks["squat"] == (0, 12, 24, 36)
    assert repetition_bounds(marks["squat"], FRAMES) == ((0, 12), (12, 24), (24, 36))
    assert repetition_bounds((5, 5), FRAMES) == ()                      # one distinct mark = no cycle
    assert repetition_bounds((5, 3), FRAMES) == ((3, 5),)               # marks are sorted before pairing
    assert repetition_bounds((0, 10, 999), FRAMES) == ((0, 10),)        # marks outside the clip are dropped
    assert load_repetitions(root / "train" / "s03").get("pushup") is None


def test_cache_and_trial_share_one_frame_grid(release):
    root, cache = release
    sequence = discover_sequences(root, actions=["squat"])[0]
    reference = load_reference(sequence.reference_path)
    views = select_views(sequence, reference)
    assert cached_frames(cache, sequence, views.view_a) == FRAMES
    assert cache_path(cache, sequence, views.view_a).is_file()
    frames = common_frames(cache, sequence, views, frame_stride=2)
    assert frames.tolist() == list(range(0, FRAMES, 2))
    predictions = tuple(load_prediction(cache, sequence, camera, frames) for camera in (views.view_a, views.view_b))
    trial = build_trial(sequence, views, predictions, frame_ids=frames, frame_stride=2)
    assert trial.face.shape == (len(frames), 70, 3) and trial.fps == pytest.approx(25.0)
    assert trial.person_id == "s03" and trial.trial_id == "squat"
    assert trial.source_metadata["view_a"] == views.view_a and trial.source_metadata["offset_side_to_face"] == 0
    assert np.allclose(trial.timestamps, frames / 50.0)
    # Cycle bounds follow the thinned grid.
    assert scale_bounds(((0, 12), (12, 24), (24, 36)), frames) == ((0, 6), (6, 12), (12, 18))
    trial_full, reference_points = load_sequence(sequence, views, cache, reference=True)
    assert trial_full.face.shape == (FRAMES, 70, 3) and reference_points.shape == (FRAMES, 25, 3)
    with pytest.raises(FileNotFoundError):
        load_prediction(cache, sequence, "missing-camera", frames)


def test_missing_detections_stay_invalid(release, tmp_path: Path):
    root, cache = release
    sequence = discover_sequences(root, actions=["squat"])[0]
    path = cache_path(cache, sequence, CAMERAS[0])
    with np.load(path, allow_pickle=True) as payload:
        arrays = {key: payload[key] for key in payload.files}
    keep = np.arange(FRAMES) != 5                       # frame 5 was not detected
    arrays.update({key: arrays[key][keep] for key in ("frame_ids", "person_rank", "points3d", "points2d", "valid3d", "valid2d")})
    np.savez(path, **arrays)
    prediction = load_prediction(cache, sequence, CAMERAS[0], np.arange(FRAMES))
    assert prediction.failed_frames == (5,)
    assert not prediction.valid_3d[5].any() and prediction.valid_3d[4].all()


def test_reference_mapping_covers_the_comparison_joints():
    from common.skeletons.mhr70 import MHR70_INDEX

    mapped = {name for name in JOINTS3D_25_TO_MHR70 if name in MHR70_INDEX}
    assert {"left-shoulder", "right-shoulder", "left-elbow", "right-elbow", "left-wrist", "right-wrist",
            "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle", "right-ankle"} <= mapped
    assert "left-heel" not in mapped and "left-eye" not in mapped
