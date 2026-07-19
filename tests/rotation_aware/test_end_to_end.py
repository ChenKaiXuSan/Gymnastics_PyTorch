"""Synthetic black-box coverage for the rotation-aware command sequence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware.cli import main


def _pose(frame: int, side: bool) -> np.ndarray:
    """Make a non-degenerate MHR70 pose with a small view-specific motion."""
    points = np.zeros((len(mhr_names), 3), dtype=np.float32)
    phase = np.float32(frame / 7.0)
    points[:, 2] = np.linspace(0.0, 0.2, len(mhr_names), dtype=np.float32)
    points[9] = (-0.5, 0.0, 0.0)
    points[10] = (0.5, 0.0, 0.0)
    points[5] = (-0.6, 1.0, 0.05 * np.sin(phase))
    points[6] = (0.6, 1.0, -0.05 * np.sin(phase))
    points[66] = (-0.62, 1.02, 0.05 * np.sin(phase))
    points[67] = (0.62, 1.02, -0.05 * np.sin(phase))
    points[69] = (0.0, 1.5, 0.15 * np.cos(phase))
    if side:
        points = points + np.array((0.03, -0.02, 0.01), dtype=np.float32)
    return points


def _write_sam3d(root: Path, view: str, frames: int) -> None:
    directory = root / "person" / "1" / view
    directory.mkdir(parents=True)
    for frame in range(frames):
        np.savez_compressed(
            directory / f"{frame:06d}_sam3d_body.npz",
            output={"frame_idx": frame, "pred_keypoints_3d": _pose(frame, view == "side")},
        )


def test_end_to_end_overlap_inference_exports_person_metrics(tmp_path: Path) -> None:
    frames = 48
    sam3d = tmp_path / "sam3d_body_results"
    _write_sam3d(sam3d, "face", frames)
    _write_sam3d(sam3d, "side", frames)
    split_root = tmp_path / "split_cycle"
    record = split_root / "person_1" / "alignment_record_1.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0, "fps": 60.0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 0, "end": frames},
                        "side_video_frames": {"start": 0, "end": frames},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fold = tmp_path / "fold_00.json"
    fold.write_text(
        json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}),
        encoding="utf-8",
    )
    output = tmp_path / "outputs"
    old_fuse_root = tmp_path / "legacy_fuse_outputs"
    old_fuse_root.mkdir()
    config = tmp_path / "rotation_aware.yaml"
    config.write_text(
        "\n".join(
            (
                "paths:",
                f"  sam3d_root: {sam3d}",
                f"  split_cycle_root: {split_root}",
                f"  output_root: {output}",
                "  skeleton: configs/fuse/skeleton_mhr70.yaml",
                f"  fold_json: {fold}",
                f"  old_fuse_root: {old_fuse_root}",
                "window:",
                "  length: 32",
                "  train_stride: 16",
                "  eval_stride: 16",
                "training:",
                "  epochs: 1",
                "  batch_size: 2",
                "  hidden_channels: 8",
                "  seed: 0",
            )
        ),
        encoding="utf-8",
    )

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    assert main(["train", "--config", str(config), "--run-id", "tiny", "--ablation", "A6"]) == 0
    assert main(["infer", "--config", str(config), "--run-id", "tiny", "--person", "1"]) == 0
    assert main(["evaluate", "--config", str(config), "--run-id", "tiny", "--person", "1"]) == 0

    sequence_path = output / "inference" / "tiny" / "person_1" / "cycle_000" / "fused_sequence.npz"
    with np.load(sequence_path, allow_pickle=False) as sequence:
        assert sequence["kpts_world"].shape == (frames, 70, 3)
        assert np.isfinite(sequence["kpts_world"]).all()
        assert np.array_equal(sequence["face_map"], np.arange(frames))
        assert np.array_equal(sequence["side_map"], np.arange(frames))
        metadata = json.loads(str(sequence["metadata"].item()))
    assert metadata["coordinate_system"] == "face_reference_uncalibrated"
    assert metadata["ablation"] == "A6"

    with (output / "evaluation" / "tiny" / "metrics_by_person.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["person_id"] for row in rows} == {"1"}
    assert {row["method"] for row in rows} >= {"A6"}
