import json
from pathlib import Path

import numpy as np
import pytest

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware.cli import main, make_parser


def _write_sam3d(root: Path, view: str) -> None:
    directory = root / "person" / "1" / view
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
