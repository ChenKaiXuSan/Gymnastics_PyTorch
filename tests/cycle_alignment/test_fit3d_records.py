"""``python -m cycle_alignment cycles fit3d``: bounds from rep_ann, middles detected."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cycle_alignment import annotate_cycles
from cycle_alignment.cycle_records import cycle_record_path, read_cycle_record

from tests.fusion.benchmarks_fit3d.test_dataset import FRAMES, make_cache, make_release


def test_fit3d_records_use_the_annotated_repetitions(tmp_path: Path):
    release = make_release(tmp_path / "Fit3D")
    cache = make_cache(tmp_path / "cache", release)
    views_path = tmp_path / "selected_views.json"
    views_path.write_text(
        json.dumps(
            {
                "target_separation_deg": 90.0,
                "sequences": [
                    {"subject_key": "s03", "sequence_key": action, "view_a": "65906101", "view_b": "50591643",
                     "azimuth_a_deg": 24.0, "azimuth_b_deg": 157.0, "separation_deg": 133.0, "frames": FRAMES, "reps": 3}
                    for action in ("squat", "pushup")
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "fit3d.yaml"
    config.write_text(
        f"paths:\n  dataset_root: {release}\n  sam3d_derived_root: {cache}\n  views_path: {views_path}\n"
        "dataset:\n  split: train\n  frame_stride: 1\n",
        encoding="utf-8",
    )
    records_root = tmp_path / "records"
    assert annotate_cycles.main(["fit3d", "--config", str(config), "--out-root", str(records_root), "--no-plot", "--smooth-window", "5"]) == 0

    record = read_cycle_record(cycle_record_path(records_root, "s03", "squat"))
    assert record.dataset == "fit3d" and record.frames == FRAMES and record.has_mids
    assert [(start, end) for start, _, end in record.cycles] == [(0, 12), (12, 24), (24, 36)]
    assert all(start < mid < end for start, mid, end in record.cycles)
    assert record.detection["theta_ref_mode"] == "rep_ann"
    assert record.metadata["cycle_source"] == "rep_ann" and record.metadata["repetition_marks"] == [0, 12, 24, 36]
    assert record.metadata["action"] == "squat" and list(record.metadata["views"]) == ["65906101", "50591643"]

    # An exercise without repetition annotations yields a record with no cycles.
    empty = read_cycle_record(cycle_record_path(records_root, "s03", "pushup"))
    assert empty.cycles == () and empty.frames == FRAMES

    summary = json.loads((records_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["sequences"] == 2 and summary["cycles"] == 3 and summary["with_cycles"] == 1


def test_fit3d_records_follow_the_frame_stride(tmp_path: Path):
    release = make_release(tmp_path / "Fit3D", actions=("squat",))
    cache = make_cache(tmp_path / "cache", release, actions=("squat",))
    views_path = tmp_path / "views.json"
    views_path.write_text(json.dumps({"sequences": [{"subject_key": "s03", "sequence_key": "squat", "view_a": "65906101", "view_b": "50591643", "azimuth_a_deg": 24.0, "azimuth_b_deg": 157.0, "separation_deg": 133.0}]}), encoding="utf-8")
    config = tmp_path / "fit3d.yaml"
    config.write_text(f"paths:\n  dataset_root: {release}\n  sam3d_derived_root: {cache}\n  views_path: {views_path}\ndataset:\n  split: train\n  frame_stride: 2\n", encoding="utf-8")
    records_root = tmp_path / "records"
    assert annotate_cycles.main(["fit3d", "--config", str(config), "--out-root", str(records_root), "--no-plot", "--smooth-window", "5"]) == 0
    record = read_cycle_record(cycle_record_path(records_root, "s03", "squat"))
    assert record.frames == FRAMES // 2
    assert [(start, end) for start, _, end in record.cycles] == [(0, 6), (6, 12), (12, 18)]
    assert record.metadata["frame_stride"] == 2 and np.isclose(record.fps, 25.0)
