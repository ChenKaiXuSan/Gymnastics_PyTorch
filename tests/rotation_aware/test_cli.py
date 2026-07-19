import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware import cli
from fuse.rotation_aware.cli import (
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
    assert "2" in manifest["failures"]


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
