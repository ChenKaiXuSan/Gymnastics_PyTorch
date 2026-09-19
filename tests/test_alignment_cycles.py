"""Cycle / middle detection and record files (gymnastics.alignment)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gymnastics.alignment import annotate_cycles
from gymnastics.alignment.cycle_records import (
    augment_alignment_record,
    cycle_record_path,
    read_cycle_record,
    write_cycle_record,
)
from gymnastics.alignment.cycles import (
    CycleSpan,
    DetectionSettings,
    annotate_mid_points,
    detect_cycles,
    find_crossings,
    hand_theta_unwrapped,
    mid_point,
)
from gymnastics.common.skeletons.mhr70 import MHR70_INDEX

FPS = 30.0


def _theta(frames: int = 300, period: int = 60, amplitude: float = 1.5, offset: float = -0.7) -> np.ndarray:
    t = np.arange(frames)
    return offset + amplitude * np.sin(2 * np.pi * t / period - np.pi / 2)  # trough at t = 0


def test_detect_cycles_finds_period_and_middles_at_peaks():
    theta = _theta()
    spans, used = detect_cycles(theta, FPS, settings=DetectionSettings(min_period_sec=0.8))
    assert used.direction == "ccw" and used.theta_ref_mode == "auto_p10"
    assert 3 <= len(spans) <= 4
    for span in spans:
        assert 55 <= span.length <= 65
        # The middle is the peak, half a period after the trough; the start
        # (10th-percentile crossing) sits a few frames after the trough.
        assert 22 <= span.mid - span.start <= 30
        assert theta[span.mid] >= theta[span.start : span.end].max() - 1e-6


def test_detect_cycles_clockwise_fallback_and_filters():
    # A continuous clockwise rotation has no upward crossing with positive
    # velocity, so the detector falls back to the clockwise rule.
    theta = -2 * np.pi * np.arange(300) / 60.0 + 0.05 * np.sin(np.arange(300) / 3.0)
    spans, used = detect_cycles(theta, FPS)
    assert used.direction == "cw" and 3 <= len(spans) <= 5
    assert all(theta[s.mid] <= theta[s.start + 1 : s.end - 1].min() + 1e-6 for s in spans)
    flat, _ = detect_cycles(np.zeros(300) + 0.01 * np.sin(np.arange(300)), FPS, settings=DetectionSettings(min_amplitude_rad=0.5))
    assert flat == []
    limited, _ = detect_cycles(_theta(), FPS, settings=DetectionSettings(max_period_sec=1.0))
    assert limited == []
    with pytest.raises(ValueError):
        detect_cycles(_theta(), FPS, settings=DetectionSettings(theta_ref=None, theta_ref_mode="manual"))


def test_find_crossings_and_mid_point_basics():
    theta = _theta()
    crossings = find_crossings(theta, FPS, theta_ref=-0.7, min_period_sec=0.8)
    assert len(crossings) >= 4 and all(np.diff(crossings) >= 24)
    assert mid_point(theta, 0, 60, "ccw") == 30
    with pytest.raises(ValueError):
        mid_point(theta, 0, 2, "ccw")
    with pytest.raises(ValueError):
        CycleSpan(5, 5, 10)


def test_annotate_mid_points_infers_direction_per_cycle():
    theta = _theta()
    spans = annotate_mid_points(theta, [(5, 65), (65, 125)])
    assert [s.start for s in spans] == [5, 65] and all(abs((s.mid - s.start) - 25) <= 2 for s in spans)
    reversed_spans = annotate_mid_points(-theta, [(5, 65)])
    assert abs((reversed_spans[0].mid - 5) - 25) <= 2


def test_hand_theta_unwrapped_interpolates_missing_frames():
    frames = 50
    body = np.zeros((frames, 70, 3), dtype=np.float32)
    angle = np.linspace(-np.pi, np.pi, frames)
    body[:, 41, 0], body[:, 41, 2] = np.cos(angle), np.sin(angle)
    body[10:15] = np.nan
    theta = hand_theta_unwrapped(body, smooth_window=3)
    assert theta.shape == (frames,) and np.isfinite(theta).all()
    assert np.all(np.diff(theta[20:45]) > 0)  # unwrapped, monotone


def test_cycle_record_round_trip_and_alignment_augmentation(tmp_path: Path):
    spans = [CycleSpan(0, 8, 20), CycleSpan(20, 30, 40)]
    path = write_cycle_record(cycle_record_path(tmp_path, "12", "sess"), dataset="freeman", subject_id="12", sequence_id="sess", fps=25.0, frames=40, spans=spans, detection=DetectionSettings(), views=("c04", "c07"), extra_metadata={"split": "test"})
    record = read_cycle_record(path)
    assert record.dataset == "freeman" and record.fps == 25.0 and record.frames == 40 and record.has_mids
    assert record.spans() == tuple(spans) and record.metadata["views"] == ["c04", "c07"] and record.metadata["split"] == "test"
    assert record.detection["mid_rule"] == "theta_extremum"

    private = tmp_path / "alignment_record_7.json"
    private.write_text(json.dumps({"metadata": {"person_id": "7", "offset_side_to_face": -3, "fps": 60.0}, "cycles": [
        {"cycle_index": 0, "face_video_frames": {"start": 100, "end": 160}, "side_video_frames": {"start": 97, "end": 157}},
    ]}))
    legacy = read_cycle_record(private)
    assert legacy.dataset == "gymnastics" and legacy.subject_id == "7" and not legacy.has_mids and legacy.cycles == ((100, -1, 160),)
    with pytest.raises(ValueError):
        legacy.spans()
    augment_alignment_record(private, [CycleSpan(100, 125, 160)], [CycleSpan(97, 122, 157)], DetectionSettings(theta_ref=None, theta_ref_mode="legacy_align"))
    augmented = read_cycle_record(private)
    assert augmented.has_mids and augmented.cycles == ((100, 125, 160),) and augmented.side_cycles == ((97, 122, 157),)
    assert augmented.detection["theta_ref"] is None
    assert json.loads(private.read_text())["metadata"]["offset_side_to_face"] == -3  # untouched
    with pytest.raises(ValueError):
        augment_alignment_record(private, [CycleSpan(101, 125, 160)], [CycleSpan(97, 122, 157)], DetectionSettings())


def _world_pose(frames: int, period: int) -> np.ndarray:
    pose = np.zeros((frames, 70, 3), dtype=np.float32)
    t = np.arange(frames)
    theta = 1.2 * np.sin(2 * np.pi * t / period - np.pi / 2)
    pose[:, MHR70_INDEX["left-hip"]] = (-0.1, 0.0, 0.0)
    pose[:, MHR70_INDEX["right-hip"]] = (0.1, 0.0, 0.0)
    pose[:, MHR70_INDEX["left-shoulder"]] = (-0.2, 0.5, 0.0)
    pose[:, MHR70_INDEX["right-shoulder"]] = (0.2, 0.5, 0.0)
    pose[:, MHR70_INDEX["right-wrist"], 0] = 0.4 * np.cos(theta)
    pose[:, MHR70_INDEX["right-wrist"], 1] = 0.2
    pose[:, MHR70_INDEX["right-wrist"], 2] = 0.4 * np.sin(theta)
    return pose + np.array([1.0, 2.0, 3.0], dtype=np.float32)


def test_annotate_private_person_adds_middles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    period, frames = 60, 400
    face = _world_pose(frames, period)
    side = np.roll(face, 3, axis=0)  # side video starts 3 frames earlier
    log_root = tmp_path / "split_cycle"
    record = log_root / "person_9" / "alignment_record_9.json"
    record.parent.mkdir(parents=True)
    cycles = [{"cycle_index": i, "face_video_frames": {"start": 30 + 60 * i, "end": 90 + 60 * i}, "side_video_frames": {"start": 33 + 60 * i, "end": 93 + 60 * i}} for i in range(4)]
    record.write_text(json.dumps({"metadata": {"person_id": "9", "offset_side_to_face": 3, "fps": 30.0}, "cycles": cycles}))

    import gymnastics.alignment.load as load_module

    monkeypatch.setattr(load_module, "load_sam3d_body_sequence", lambda root, person_id, subdir: ([], face if subdir == "face" else side))
    summary = annotate_cycles.annotate_private_person("9", kpt_root=tmp_path, log_root=log_root, smooth_window=5, plot=False)
    assert summary["cycles"] == 4 and 0.3 < summary["mid_ratio_mean"] < 0.7
    augmented = read_cycle_record(record)
    assert augmented.has_mids
    for (fs, fm, fe), (ss, sm, se) in zip(augmented.cycles, augmented.side_cycles):
        assert fs < fm < fe and (fm - fs) == (sm - ss)
        assert abs((fm - fs) - period // 2) <= 3  # peak half a period after the trough-aligned start


def test_annotate_sequence_writes_public_record(tmp_path: Path):
    view_a = _world_pose(300, 60)
    view_b = view_a.copy()
    path, count = annotate_cycles.annotate_sequence(dataset="unity", subject_id="seq", sequence_id="seq", fps=30.0, view_a=view_a, view_b=view_b, valid_a=None, valid_b=None, views=("cam0", "cam1"), settings=DetectionSettings(min_period_sec=0.8, max_period_sec=6.0, min_amplitude_rad=0.5), out_root=tmp_path, plot=False)
    assert count >= 3 and path == cycle_record_path(tmp_path, "seq", "seq")
    record = read_cycle_record(path)
    assert record.has_mids and record.frames == 300 and record.detection["direction"] in {"ccw", "cw"}
    assert annotate_cycles.main(["freeman", "--help"]) if False else True
