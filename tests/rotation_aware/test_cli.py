import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware import cli
from fuse.rotation_aware.config import load_skeleton_spec
from fuse.rotation_aware.cli import (
    _cache_trial_paths,
    _cached_trials,
    loss_config_for_ablation,
    main,
    make_parser,
    resolve_fold,
)


def _write_sam3d(root: Path, view: str, person: str = "1") -> None:
    directory = root / "person" / person / view
    directory.mkdir(parents=True)
    for frame in range(4):
        points = np.ones((len(mhr_names), 3), dtype=np.float32)
        points[9, 0], points[10, 0] = -1, 1
        points[5, 1], points[6, 1], points[2, 1] = 2, 2, 3
        np.savez_compressed(
            directory / f"{frame:06d}_sam3d_body.npz",
            output={"frame_idx": frame, "pred_keypoints_3d": points},
        )


def test_cli_parser_lists_all_task_seven_subcommands(capsys) -> None:
    parser = make_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0
    assert "{prepare,train,infer,evaluate}" in capsys.readouterr().out


def test_prepare_smoke_uses_real_tiny_files_and_writes_manifest(tmp_path: Path) -> None:
    sam3d = tmp_path / "sam3d" / "sam3d_body_results"
    _write_sam3d(sam3d, "face")
    _write_sam3d(sam3d, "side")
    split = tmp_path / "split"
    record = split / "person_1" / "alignment_record_1.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0, "fps": 60.0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 0, "end": 4},
                        "side_video_frames": {"start": 0, "end": 4},
                    }
                ],
            }
        )
    )
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}))
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n  output_root: {tmp_path / 'out'}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}",
                "training:\n  epochs: 1\n  batch_size: 1\n  hidden_channels: 8",
            ]
        )
    )

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    assert (tmp_path / "out" / "cache" / "person_1" / "cycle_000.npz").exists()
    assert (tmp_path / "out" / "split_manifest.json").exists()


def test_train_infer_evaluate_smoke_uses_canonical_cached_trials(
    tmp_path: Path,
) -> None:
    sam3d = tmp_path / "sam3d" / "sam3d_body_results"
    _write_sam3d(sam3d, "face")
    _write_sam3d(sam3d, "side")
    split = tmp_path / "split"
    record = split / "person_1" / "alignment_record_1.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0, "fps": 60.0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 0, "end": 4},
                        "side_video_frames": {"start": 0, "end": 4},
                    }
                ],
            }
        )
    )
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}))
    config = tmp_path / "config.yaml"
    out = tmp_path / "out"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}\ntraining:\n  epochs: 1\n  batch_size: 1\n  hidden_channels: 8\n  seed: 3"
    )

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    assert main(["train", "--config", str(config), "--run-id", "tiny"]) == 0
    config.write_text(
        config.read_text().replace("hidden_channels: 8", "hidden_channels: 16")
    )
    assert (
        main(["infer", "--config", str(config), "--run-id", "tiny", "--person", "1"])
        == 0
    )
    assert (
        main(["evaluate", "--config", str(config), "--run-id", "tiny", "--person", "1"])
        == 0
    )
    assert (
        out / "inference" / "tiny" / "person_1" / "cycle_000" / "fused_sequence.npz"
    ).exists()
    assert (out / "evaluation" / "tiny" / "metrics_by_person.csv").exists()


def test_fold_resolver_uses_active_default_and_accepts_index_or_json_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "folds"
    root.mkdir()
    default = root / "fold_00.json"
    indexed = root / "fold_01.json"
    default.write_text("{}")
    indexed.write_text("{}")
    config = {"paths": {"fold_root": str(root), "default_fold": "fold_00.json"}}

    assert resolve_fold(config, None) == default
    assert resolve_fold(config, "1") == indexed
    assert resolve_fold(config, str(indexed)) == indexed
    with pytest.raises(ValueError, match="fold"):
        resolve_fold({"paths": {}}, None)


def test_prepare_filters_people_without_alignment_and_reports_them(
    tmp_path: Path,
) -> None:
    sam3d = tmp_path / "sam3d" / "sam3d_body_results"
    for person in ("1", "2"):
        _write_sam3d(sam3d, "face", person)
        _write_sam3d(sam3d, "side", person)
    split = tmp_path / "split"
    record = split / "person_1" / "alignment_record_1.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0, "fps": 60.0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 0, "end": 4},
                        "side_video_frames": {"start": 0, "end": 4},
                    }
                ],
            }
        )
    )
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}))
    out = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}"
    )

    assert main(["prepare", "--config", str(config)]) == 0

    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["prepared_people"] == ["1"]
    assert manifest["selected_people"] == ["1"]
    assert "2" not in manifest["failures"]
    assert "2" in manifest["excluded_sam3d_people"]


def test_ablation_loss_configs_change_the_actual_training_objectives() -> None:
    spatial = loss_config_for_ablation("A4")
    rotation_temporal = loss_config_for_ablation("A5")
    full = loss_config_for_ablation("A6")

    assert spatial.circular_axial_rotation_weight == 0
    assert spatial.so3_rotation_weight == 0
    assert spatial.adaptive_temporal_acceleration_weight == 0
    assert spatial.complete_cycle_rom_weight == 0
    assert rotation_temporal.complete_cycle_rom_weight == 0
    assert rotation_temporal.circular_axial_rotation_weight > 0
    assert full.complete_cycle_rom_weight > 0


def test_train_person_subset_without_train_trials_fails_clearly(
    tmp_path: Path, monkeypatch
) -> None:
    fold = tmp_path / "fold.json"
    fold.write_text(
        json.dumps(
            {
                "train": [{"person_id": "1"}],
                "val": [{"person_id": "2"}],
                "test": [],
            }
        )
    )
    config = {
        "paths": {
            "sam3d_root": str(tmp_path / "sam3d"),
            "split_cycle_root": str(tmp_path / "split"),
            "output_root": str(tmp_path / "out"),
            "skeleton": "configs/fuse/skeleton_mhr70.yaml",
            "fold_json": str(fold),
        }
    }
    monkeypatch.setattr(
        cli, "_cached_trials", lambda *_: [SimpleNamespace(person_id="2")]
    )

    with pytest.raises(ValueError, match="no selected canonical trials"):
        cli._cmd_train(
            Namespace(
                run_id="tiny", output_root=None, fold=None, person=["2"], ablation=None
            ),
            config,
        )


def test_cached_trials_rejects_any_selected_person_without_nonempty_cache(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "person_1").mkdir()
    (tmp_path / "person_1" / "cycle_000.npz").touch()
    monkeypatch.setattr(cli, "load_cached_trial", lambda _: (SimpleNamespace(), {}))
    monkeypatch.setattr(
        cli, "canonicalize_trial", lambda trial, _: SimpleNamespace(trial=trial)
    )

    with pytest.raises(FileNotFoundError, match="person_2"):
        _cached_trials(tmp_path, ["1", "2"], object())


def test_prepare_explicit_failure_returns_nonzero_after_writing_manifest(
    tmp_path: Path,
) -> None:
    sam3d = tmp_path / "sam3d" / "sam3d_body_results"
    _write_sam3d(sam3d, "face", "1")
    _write_sam3d(sam3d, "side", "1")
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}))
    out = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {tmp_path / 'split'}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}"
    )

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 1
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert "1" in manifest["failures"]


def test_default_prepare_returns_nonzero_after_active_aligned_person_cache_failure(
    tmp_path: Path, monkeypatch
) -> None:
    sam3d = tmp_path / "sam3d" / "sam3d_body_results"
    _write_sam3d(sam3d, "face", "1")
    _write_sam3d(sam3d, "side", "1")
    split = tmp_path / "split"
    record = split / "person_1" / "alignment_record_1.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "metadata": {"offset_side_to_face": 0},
                "cycles": [
                    {
                        "cycle_index": 0,
                        "face_video_frames": {"start": 0, "end": 4},
                        "side_video_frames": {"start": 0, "end": 4},
                    }
                ],
            }
        )
    )
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}))
    out = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}"
    )
    monkeypatch.setattr(
        cli,
        "load_person_trials",
        lambda *_: (_ for _ in ()).throw(ValueError("bad cache source")),
    )

    assert main(["prepare", "--config", str(config)]) == 1
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert "1" in manifest["failures"]


def test_default_prepare_keeps_aligned_people_outside_fold_membership(
    tmp_path: Path,
) -> None:
    sam3d = tmp_path / "sam3d" / "sam3d_body_results"
    split = tmp_path / "split"
    for person in ("1", "2"):
        _write_sam3d(sam3d, "face", person)
        _write_sam3d(sam3d, "side", person)
        record = split / f"person_{person}" / f"alignment_record_{person}.json"
        record.parent.mkdir(parents=True)
        record.write_text(
            json.dumps(
                {
                    "metadata": {"offset_side_to_face": 0},
                    "cycles": [
                        {
                            "cycle_index": 0,
                            "face_video_frames": {"start": 0, "end": 4},
                            "side_video_frames": {"start": 0, "end": 4},
                        }
                    ],
                }
            )
        )
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}))
    out = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}"
    )

    assert main(["prepare", "--config", str(config)]) == 0
    assert json.loads((out / "split_manifest.json").read_text())["selected_people"] == [
        "1",
        "2",
    ]
    assert (out / "cache" / "person_2" / "cycle_000.npz").exists()


def test_cache_paths_require_every_manifest_declared_cycle(tmp_path: Path) -> None:
    person = tmp_path / "person_1"
    person.mkdir()
    (person / "cycle_000.npz").touch()
    (person / "manifest.json").write_text(
        json.dumps({"person_id": "1", "trials": ["cycle_000", "cycle_001"]})
    )

    with pytest.raises(FileNotFoundError, match="person_1"):
        _cache_trial_paths(tmp_path, ["1"])


def test_evaluate_combines_a4_a5_a6_runs_with_deterministic_a0_a3(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    values = np.ones((4, len(mhr_names), 3), dtype=np.float32)
    values[:, 9, 0], values[:, 10, 0] = -1, 1
    values[:, 5, 1], values[:, 6, 1], values[:, 2, 1] = 2, 2, 3
    for run_id, ablation in (("a4", "A4"), ("a5", "A5"), ("a6", "A6")):
        root = out / "inference" / run_id / "person_1" / "cycle_000"
        root.mkdir(parents=True)
        np.savez_compressed(
            root / "fused_sequence.npz",
            kpts_world=values,
            kpts_face_world=values,
            kpts_side_world=values,
            kpts_arithmetic_world=values,
            kpts_base_world=values,
            frame_valid=np.ones(4, dtype=bool),
            joint_valid=np.ones(values.shape[:2], dtype=bool),
            face_map=np.arange(4),
            side_map=np.arange(4),
            timestamps=np.arange(4) / 60.0,
            metadata=np.asarray(json.dumps({"ablation": ablation})),
            diagnostics=np.asarray(json.dumps({ablation: {"swap_error": 0.0}})),
        )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {tmp_path / 'sam3d'}\n  split_cycle_root: {tmp_path / 'split'}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {tmp_path / 'fold.json'}"
    )

    assert (
        main(
            [
                "evaluate",
                "--config",
                str(config),
                "--run-id",
                "a4",
                "--run-id",
                "a5",
                "--run-id",
                "a6",
            ]
        )
        == 0
    )

    report = json.loads((out / "evaluation" / "a4+a5+a6" / "report.json").read_text())
    assert {row["method"] for row in report["person_metrics"]} >= {
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
    }
    assert len(report["person_metrics"]) == 7


def test_inference_rejects_checkpoint_with_different_skeleton_contract() -> None:
    skeleton = load_skeleton_spec("configs/fuse/skeleton_mhr70.yaml")

    with pytest.raises(ValueError, match="skeleton"):
        cli._validate_checkpoint_skeleton({"skeleton": {}}, skeleton)
