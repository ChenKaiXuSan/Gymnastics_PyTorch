"""Synthetic black-box coverage for the rotation-aware command sequence."""

from __future__ import annotations

import csv
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fuse.metadata.mhr70 import mhr_names
import fuse.rotation_aware.cli as cli
from fuse.rotation_aware.cli import _training_config_for_ablation, load_config, main


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


def test_batch64_schedule_override_records_resolved_checkpoint_settings(
    tmp_path: Path, monkeypatch
) -> None:
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
    production_config = load_config("configs/fuse/rotation_aware_batch64.yaml")
    assert production_config["paths"]["output_root"] == "logs/fuse_rotation_aware/batch64"
    assert production_config["data"]["cache_dir"] == "logs/fuse_rotation_aware/cache"
    assert production_config["training"]["protocol"] == {
        "run_id_token_template": "{ablation_lower}_b{batch_size}_e{epochs}"
    }
    assert _training_config_for_ablation(production_config, "A4")["epochs"] == 200
    assert _training_config_for_ablation(production_config, "A5")["epochs"] == 200
    assert _training_config_for_ablation(production_config, "A6")["epochs"] == 100
    assert _training_config_for_ablation(production_config, "A6")["batch_size"] == 64

    tiny_config = deepcopy(production_config)
    tiny_config["paths"].update(
        {
            "sam3d_root": str(sam3d),
            "split_cycle_root": str(split_root),
            "output_root": str(output),
            "fold_json": str(fold),
            "old_fuse_root": str(old_fuse_root),
        }
    )
    tiny_config["data"]["cache_dir"] = str(output / "cache")
    tiny_config["window"].update(
        {"length": 32, "train_stride": 16, "eval_stride": 16}
    )
    tiny_training = tiny_config["training"]
    tiny_training.pop("epochs_by_ablation")
    tiny_training.update({"epochs": 1, "hidden_channels": 8, "device": "cpu"})
    config = tmp_path / "rotation_aware_batch64_override.yaml"
    config.write_text(yaml.safe_dump(tiny_config, sort_keys=False), encoding="utf-8")

    step_count = 0
    original_step = torch.optim.Adam.step

    def count_steps(optimizer, *args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(cli.torch.optim.Adam, "step", count_steps)

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    assert main(["train", "--config", str(config), "--run-id", "batch64_a6_b64_e1", "--ablation", "A6"]) == 0

    assert step_count == 2
    checkpoint = torch.load(
        output / "runs" / "batch64_a6_b64_e1" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    training_config = checkpoint["training_config"]
    assert training_config["epochs"] == 1
    assert training_config["batch_size"] == 64
    assert training_config["ablation"] == "A6"
    assert checkpoint["provenance"]["training_config_hash"]

    tiny_config["training"]["device"] = "cuda:0"
    config.write_text(yaml.safe_dump(tiny_config, sort_keys=False), encoding="utf-8")
    benchmark = importlib.import_module("analysis.benchmark_rotation_aware_training")
    profiler_enabled: list[bool] = []
    original_profiler = benchmark.StageProfiler

    class RecordingProfiler(original_profiler):
        def __init__(self, enabled: bool, device: torch.device) -> None:
            profiler_enabled.append(enabled)
            super().__init__(enabled=enabled, device=device)

    monkeypatch.setattr(benchmark, "StageProfiler", RecordingProfiler)
    benchmark_output = tmp_path / "benchmark.json"
    assert benchmark.main(
        [
            "--config", str(config), "--ablation", "A4", "--device", "cpu",
            "--warmup-epochs", "1", "--measured-epochs", "2",
            "--output", str(benchmark_output),
        ]
    ) == 0
    benchmark_report = json.loads(benchmark_output.read_text(encoding="utf-8"))
    assert benchmark_report["measured_epochs"] == 2
    assert len(benchmark_report["epoch_timings_seconds"]) == 2
    assert benchmark_report["median_epoch_seconds"] > 0
    assert benchmark_report["effective_train_window_rate"]["median_windows_per_second"] > 0
    assert benchmark_report["peak_cuda_memory_bytes"] == 0
    assert benchmark_report["measured_peak_cuda_memory_bytes"] == [0, 0]
    assert benchmark_report["workload_counts"] == {
        "train_windows": 2,
        "train_complete_cycles": 1,
        "validation_windows": 2,
        "validation_complete_cycles": 1,
    }
    assert benchmark_report["config"]["source_training"]["device"] == "cuda:0"
    assert benchmark_report["config"]["effective_training"]["device"] == "cpu"
    assert profiler_enabled == [False, False, False, True]
    assert benchmark_report["stage_timings"]["mode"] == "untimed_diagnostic_epoch"
    assert benchmark_report["stage_timings"]["diagnostic"]["timed"] is False
    history = benchmark_report["validation_history"]
    assert [entry["epoch"] for entry in history["epochs"]] == [0, 1, 2]
    assert [entry["phase"] for entry in history["epochs"]] == ["warmup", "measured", "measured"]
    assert all(entry["equivalent"] for entry in history["epochs"])
    assert all(entry["checkpoint_selection"]["agreement"] for entry in history["epochs"])
    assert history["final_selected_epoch_agreement"]
    assert history["accepted"]
    assert history["epochs"][-1]["losses"]["total"]["absolute_delta"] >= 0


def test_benchmark_rejects_single_measured_epoch(tmp_path: Path) -> None:
    benchmark = importlib.import_module("analysis.benchmark_rotation_aware_training")

    with pytest.raises(ValueError, match="at least two"):
        benchmark.main(
            [
                "--config",
                "configs/fuse/rotation_aware_batch64.yaml",
                "--ablation",
                "A6",
                "--device",
                "cpu",
                "--measured-epochs",
                "1",
                "--output",
                str(tmp_path / "benchmark.json"),
            ]
        )


def test_benchmark_rejects_nested_non_finite_report_values() -> None:
    benchmark = importlib.import_module("analysis.benchmark_rotation_aware_training")

    with pytest.raises(FloatingPointError, match="report.epochs.0.loss"):
        benchmark._require_finite(
            {"epochs": [{"loss": float("nan")}]}, "report"
        )


def test_benchmark_validation_history_rejects_scores_straddling_prior_best() -> None:
    benchmark = importlib.import_module("analysis.benchmark_rotation_aware_training")
    reference = {
        "losses": {"total": 1.0},
        "components": {"bone_cv": 0.5},
        "score": 0.75000000,
    }
    optimized = {
        "losses": {"total": 1.0},
        "components": {"bone_cv": 0.5},
        "score": 0.75000005,
    }

    entry = benchmark._validation_history_entry(
        reference,
        optimized,
        epoch=2,
        phase="measured",
        scalar_prior_best_score=0.75000002,
        optimized_prior_best_score=0.75000002,
    )

    assert entry["equivalent"]
    assert not entry["checkpoint_selection"]["scalar_reference"]["selected"]
    assert entry["checkpoint_selection"]["optimized"]["selected"]
    assert not entry["checkpoint_selection"]["agreement"]
    assert not entry["accepted"]
    with pytest.raises(AssertionError, match="validation acceptance failed"):
        benchmark._require_validation_acceptance(entry)
