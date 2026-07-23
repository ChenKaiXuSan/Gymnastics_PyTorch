import fcntl
import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fuse.metadata.mhr70 import mhr_names
from fuse.rotation_aware import cli
from fuse.rotation_aware.config import load_skeleton_spec
from fuse.rotation_aware.prefetch import ThroughputConfig
from fuse.rotation_aware.cli import (
    _cache_trial_paths,
    _cached_trials,
    _training_config_for_ablation,
    loss_config_for_ablation,
    main,
    make_parser,
    resolve_fold,
)


def test_training_schedule_resolves_batch64_method_epochs() -> None:
    config = {
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs_by_ablation": {"A4": 200, "A5": 200, "A6": 100},
        }
    }

    assert _training_config_for_ablation(config, "A4")["epochs"] == 200
    assert _training_config_for_ablation(config, "A5")["epochs"] == 200
    assert _training_config_for_ablation(config, "A6")["epochs"] == 100
    assert _training_config_for_ablation(config, "A6")["batch_size"] == 64


def test_no_validation_fallback_matches_reference_shared_generator_order() -> None:
    windows = [{"window_id": f"window-{index}"} for index in range(8)]
    cycles = [{"window_id": f"cycle-{index}"} for index in range(3)]
    reference_generator = torch.Generator().manual_seed(17)
    reference_windows = DataLoader(
        windows, batch_size=2, shuffle=True, generator=reference_generator,
        collate_fn=cli.collate_pose_pair_windows,
    )
    reference_cycles = DataLoader(
        cycles, batch_size=1, shuffle=True, generator=reference_generator,
        collate_fn=cli.collate_pose_pair_windows,
    )
    reference_orders = []
    for _ in range(3):
        reference_orders.append([item for batch in reference_windows for item in batch["window_id"]])
        list(reference_cycles)
        list(reference_windows)

    loader, val_loader, complete_cycle_loader, val_complete_cycle_loader, uses_training_validation = cli._build_training_loaders(
        windows,
        None,
        cycles,
        cycles,
        batch_size=2,
        generator=torch.Generator().manual_seed(17),
    )
    observed_orders = []
    for _ in range(3):
        observed_orders.append([item for batch in loader for item in batch["window_id"]])
        list(complete_cycle_loader)
        list(val_loader)
        list(val_complete_cycle_loader)

    assert uses_training_validation
    assert val_loader is loader
    assert observed_orders == reference_orders


def test_batch64_protocol_validates_token_and_protects_existing_run_before_cache_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "paths": {
            "sam3d_root": str(tmp_path / "sam3d"),
            "split_cycle_root": str(tmp_path / "split"),
            "output_root": str(tmp_path / "batch64"),
        },
        "training": {
            "epochs_by_ablation": {"A4": 200, "A5": 200, "A6": 100},
            "batch_size": 64,
            "protocol": {"run_id_token_template": "{ablation_lower}_b{batch_size}_e{epochs}"},
        },
    }
    args = Namespace(run_id="paper_a4", ablation="A4", output_root=None, fold=None, person=None)
    monkeypatch.setattr(cli, "_cached_trials_with_provenance", lambda *args, **kwargs: pytest.fail("cache must not load"))

    resolved = _training_config_for_ablation(config, "A4")
    assert cli._validate_protocol_run_id("paper_a4_b64_e200_seed0", resolved)

    with pytest.raises(ValueError, match="a4_b64_e200"):
        cli._cmd_train(args, config)

    for misleading in ("paper_a4_b64_e2000", "xa4_b64_e200"):
        args.run_id = misleading
        with pytest.raises(ValueError, match="a4_b64_e200"):
            cli._cmd_train(args, config)

    args.run_id = "../../runs/paper_a4_b64_e200"
    with pytest.raises(ValueError, match="safe run-ID component"):
        cli._cmd_train(args, config)
    assert not (tmp_path / "runs" / "paper_a4_b64_e200").exists()

    protected = tmp_path / "batch64" / "runs" / "paper_a4_b64_e200"
    protected.mkdir(parents=True)
    (protected / "checkpoint.pt").write_text("do not overwrite", encoding="utf-8")
    args.run_id = "paper_a4_b64_e200"
    with pytest.raises(FileExistsError, match="protected batch-64 run directory"):
        cli._cmd_train(args, config)


@pytest.mark.parametrize("run_id", ["", ".", "..", "/tmp/run", "paper/run", r"paper\\run", "a..b"])
def test_safe_run_id_component_rejects_empty_and_path_like_values(run_id: str) -> None:
    with pytest.raises(ValueError, match="safe run-ID component"):
        cli._validate_safe_run_id_component(run_id)


def test_protected_infer_and_evaluate_reject_unsafe_run_id_components() -> None:
    config = {
        "training": {
            "epochs_by_ablation": {"A4": 200, "A5": 200, "A6": 100},
            "batch_size": 64,
            "protocol": {"run_id_token_template": "{ablation_lower}_b{batch_size}_e{epochs}"}
        }
    }

    with pytest.raises(ValueError, match="safe run-ID component"):
        cli._cmd_infer(Namespace(run_id="../../escape"), config)
    with pytest.raises(ValueError, match="safe run-ID component"):
        cli._cmd_evaluate(Namespace(run_id=["paper_a4_b64_e200", "../../escape"]), config)


def test_protected_infer_requires_loaded_checkpoint_protocol_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "paths": {
            "sam3d_root": str(tmp_path / "sam3d"),
            "split_cycle_root": str(tmp_path / "split"),
            "output_root": str(tmp_path / "batch64"),
        },
        "training": {
            "epochs_by_ablation": {"A4": 200, "A5": 200, "A6": 100},
            "batch_size": 64,
            "protocol": {"run_id_token_template": "{ablation_lower}_b{batch_size}_e{epochs}"},
        },
    }
    checkpoint = tmp_path / "external.pt"
    checkpoint_training = _training_config_for_ablation(config, "A4")
    monkeypatch.setattr(cli.torch, "load", lambda *args, **kwargs: {"training_config": checkpoint_training})
    monkeypatch.setattr(cli, "load_skeleton_spec", lambda path: object())
    monkeypatch.setattr(cli, "_validate_checkpoint_skeleton", lambda *args: None)
    monkeypatch.setattr(cli, "RotationAwareFusionModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "load_checkpoint", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("after token validation")))

    mismatched = Namespace(run_id="paper_a4_b64_e2000", output_root=None, checkpoint=str(checkpoint))
    with pytest.raises(ValueError, match="a4_b64_e200"):
        cli._cmd_infer(mismatched, config)
    exact = Namespace(run_id="paper_a4_b64_e200_seed0", output_root=None, checkpoint=str(checkpoint))
    with pytest.raises(RuntimeError, match="after token validation"):
        cli._cmd_infer(exact, config)

    mismatched_training = dict(checkpoint_training)
    mismatched_training["epochs"] = 10
    monkeypatch.setattr(
        cli.torch,
        "load",
        lambda *args, **kwargs: {"training_config": mismatched_training},
    )
    disguised = Namespace(
        run_id="paper_a4_b64_e200_a4_b64_e10",
        output_root=None,
        checkpoint=str(checkpoint),
    )
    with pytest.raises(ValueError, match="does not match active config"):
        cli._cmd_infer(disguised, config)


def test_protected_evaluate_requires_one_resolved_protocol_token_before_output_paths(
    tmp_path: Path,
) -> None:
    config = {
        "paths": {
            "sam3d_root": str(tmp_path / "sam3d"),
            "split_cycle_root": str(tmp_path / "split"),
            "output_root": str(tmp_path / "batch64"),
        },
        "training": {
            "epochs_by_ablation": {"A4": 200, "A5": 200, "A6": 100},
            "batch_size": 64,
            "protocol": {"run_id_token_template": "{ablation_lower}_b{batch_size}_e{epochs}"},
        },
    }

    for run_id in ("tiny", "paper_a4_b64_e2000"):
        with pytest.raises(ValueError, match="exactly one"):
            cli._cmd_evaluate(Namespace(run_id=[run_id], output_root=None), config)
    assert not (tmp_path / "batch64" / "evaluation").exists()
    for run_id in ("paper_a4_b64_e200", "paper_a5_b64_e200", "paper_a6_b64_e100_seed0"):
        cli._validate_config_protocol_run_id(run_id, config)


@pytest.mark.parametrize(
    ("schedule", "ablation", "message"),
    [
        ({"A4": 200, "A5": 200}, "A6", "exactly A4, A5, and A6"),
        ({"A4": 200, "A5": 200, "A6": 0}, "A6", "positive"),
        ({"A4": 200, "A5": 200, "A6": 100}, "A7", "A4, A5, or A6"),
        ({"A4": 200, "A5": 200, "A6": 100, "A7": 100}, "A6", "exactly A4, A5, and A6"),
    ],
)
def test_training_schedule_rejects_invalid_ablation_or_epochs(
    schedule: dict[str, int], ablation: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _training_config_for_ablation(
            {"training": {"epochs_by_ablation": schedule}}, ablation
        )


def test_train_rejects_invalid_schedule_before_loading_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [], "val": [], "test": []}))
    config = {
        "paths": {
            "sam3d_root": str(tmp_path / "sam3d"),
            "split_cycle_root": str(tmp_path / "split"),
            "output_root": str(tmp_path / "out"),
            "skeleton": "configs/fuse/skeleton_mhr70.yaml",
            "fold_json": str(fold),
        },
        "training": {"epochs_by_ablation": {"A4": 200, "A5": 200}},
    }

    def fail_if_cache_is_loaded(*args, **kwargs):
        raise AssertionError("cache loading must follow schedule resolution")

    monkeypatch.setattr(cli, "_cached_trials_with_provenance", fail_if_cache_is_loaded)

    with pytest.raises(ValueError, match="exactly A4, A5, and A6"):
        cli._cmd_train(
            Namespace(
                run_id="ordering-test",
                output_root=None,
                fold=None,
                person=None,
                ablation="A6",
            ),
            config,
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


def _declared_cache_cycle(person_cache: Path, trial_id: str) -> Path:
    manifest = json.loads((person_cache / "manifest.json").read_text(encoding="utf-8"))
    generation = manifest.get("generation")
    root = (
        person_cache / ".generations" / generation
        if isinstance(generation, str)
        else person_cache
    )
    return root / f"{trial_id}.npz"


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
    assert _declared_cache_cycle(
        tmp_path / "out" / "cache" / "person_1", "cycle_000"
    ).exists()
    assert (tmp_path / "out" / "split_manifest.json").exists()


def test_configured_training_device_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    fold.write_text(
        json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []})
    )
    output = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n"
        f"  output_root: {output}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n"
        f"  fold_json: {fold}\ntraining:\n  epochs: 1\n  batch_size: 1\n"
        "  hidden_channels: 8\n  seed: 3\n  device: cuda:1\n"
        "performance:\n  prefetch_batches: 2\n  pin_memory: true\n"
        "  non_blocking_transfer: true\n  cache_validation_batches: true\n"
        "  profile_stages: true"
    )
    seen: list[tuple[str, object]] = []

    def fake_train(*args, **kwargs):
        seen.append(("train", kwargs.get("device")))
        assert kwargs["throughput_config"] in {
            ThroughputConfig(
                prefetch_batches=2,
                pin_memory=True,
                non_blocking_transfer=True,
                cache_validation_batches=True,
                profile_stages=True,
            ),
            ThroughputConfig(
                prefetch_batches=2,
                pin_memory=True,
                non_blocking_transfer=True,
                cache_validation_batches=True,
                profile_stages=False,
            ),
        }
        return {"loss": 1.0}

    def fake_validate(*args, **kwargs):
        seen.append(("validate", kwargs.get("device")))
        assert kwargs["throughput_config"].cache_validation_batches
        assert kwargs["prepared_loader"] is None
        assert kwargs["prepared_complete_cycle_loader"] is None
        assert kwargs["scalar_forward"] is True
        return {"score": 0.5}

    monkeypatch.setattr(cli, "train_one_epoch", fake_train)
    monkeypatch.setattr(cli, "validate", fake_validate)

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    assert main(["train", "--config", str(config), "--run-id", "device-test"]) == 0
    assert seen == [("train", "cuda:1"), ("validate", "cuda:1")]
    assert "[epoch] run_id=device-test epoch=1/1 loss=1 val_score=0.5" in capsys.readouterr().out
    profile = output / "runs" / "device-test" / "stage_profile.jsonl"
    assert profile.exists()
    assert json.loads(profile.read_text().strip())["epoch"] == 0
    assert main(["train", "--config", str(config), "--run-id", "device-test"]) == 0
    assert len(profile.read_text().splitlines()) == 1
    config.write_text(
        config.read_text().replace("profile_stages: true", "profile_stages: false")
    )
    assert main(["train", "--config", str(config), "--run-id", "device-test"]) == 0
    assert not profile.exists()


def test_prepare_reports_empty_alignment_cycles_without_index_error(tmp_path: Path) -> None:
    sam3d = tmp_path / "sam3d" / "sam3d_body_results"
    _write_sam3d(sam3d, "face")
    _write_sam3d(sam3d, "side")
    split = tmp_path / "split"
    record = split / "person_1" / "alignment_record_1.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"metadata": {"offset_side_to_face": 0}, "cycles": []}))
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps({"train": [{"person_id": "1"}], "val": [], "test": []}))
    output = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n  output_root: {output}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}"
    )

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 1
    report = json.loads((output / "split_manifest.json").read_text())
    assert "alignment record has no cycles" in report["failures"]["1"]


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
    assert not (out / "runs" / "tiny" / "stage_profile.jsonl").exists()
    checkpoint = torch.load(
        out / "runs" / "tiny" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    cache_identity = checkpoint["provenance"]["cache_manifests"]["1"]
    run_metadata = json.loads((out / "runs" / "tiny" / "run_metadata.json").read_text())
    inference_metadata = json.loads(
        (
            out
            / "inference"
            / "tiny"
            / "person_1"
            / "cycle_000"
            / "metadata.json"
        ).read_text()
    )
    assert run_metadata["provenance"]["cache_manifests"]["1"] == cache_identity
    assert inference_metadata["consumed_cache_manifest"] == cache_identity


def test_infer_rejects_cache_generation_republished_after_training(
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
    out = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  sam3d_root: {sam3d}\n  split_cycle_root: {split}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  fold_json: {fold}\ntraining:\n  epochs: 1\n  batch_size: 1\n  hidden_channels: 8\n  seed: 3"
    )

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    assert main(["train", "--config", str(config), "--run-id", "stale"]) == 0
    checkpoint = torch.load(
        out / "runs" / "stale" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    trained_generation = checkpoint["provenance"]["cache_manifests"]["1"][
        "generation"
    ]

    assert main(["prepare", "--config", str(config), "--person", "1"]) == 0
    current_manifest = json.loads(
        (out / "cache" / "person_1" / "manifest.json").read_text()
    )
    assert current_manifest["generation"] != trained_generation

    with pytest.raises(ValueError, match="cache manifest identity mismatch.*person 1"):
        main(["infer", "--config", str(config), "--run-id", "stale", "--person", "1"])


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
        cli,
        "_cached_trials_with_provenance",
        lambda *_: ([SimpleNamespace(person_id="2")], {}),
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


def test_default_prepare_reports_oserror_from_cache_write(tmp_path: Path, monkeypatch) -> None:
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
        "write_person_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cache write failed")),
    )

    assert main(["prepare", "--config", str(config)]) == 1
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert "cache write failed" in manifest["failures"]["1"]


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
    assert _declared_cache_cycle(out / "cache" / "person_2", "cycle_000").exists()


def test_cache_paths_require_every_manifest_declared_cycle(tmp_path: Path) -> None:
    person = tmp_path / "person_1"
    person.mkdir()
    (person / "cycle_000.npz").touch()
    (person / "manifest.json").write_text(
        json.dumps({"person_id": "1", "trials": ["cycle_000", "cycle_001"]})
    )

    with pytest.raises(FileNotFoundError, match="person_1"):
        _cache_trial_paths(tmp_path, ["1"])


def _write_complete_cache_manifest(cache: Path) -> Path:
    person = cache / "person_1"
    person.mkdir(parents=True)
    (person / "cycle_000.npz").touch()
    (person / "manifest.json").write_text(
        json.dumps({"person_id": "1", "trials": ["cycle_000"]}),
        encoding="utf-8",
    )
    return person


def test_cache_paths_keeps_legacy_manifest_compatible_while_lock_exists(
    tmp_path: Path, monkeypatch
) -> None:
    person = _write_complete_cache_manifest(tmp_path)
    lock = person / ".publishing.lock"
    lock.write_text("new-writer", encoding="utf-8")
    waits: list[float] = []

    def record_wait(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr(
        cli, "time", SimpleNamespace(sleep=record_wait), raising=False
    )
    monkeypatch.setattr(cli, "_CACHE_PUBLICATION_MAX_ATTEMPTS", 2, raising=False)

    assert _cache_trial_paths(tmp_path, ["1"]) == {
        "1": [person / "cycle_000.npz"]
    }
    assert not waits


def test_cache_paths_reports_publication_timeout_without_long_sleep(
    tmp_path: Path, monkeypatch
) -> None:
    person = tmp_path / "person_1"
    person.mkdir()
    (person / ".publishing.lock").write_text("first-writer", encoding="utf-8")

    class ExclusiveGuard:
        LOCK_SH = 1
        LOCK_NB = 2
        LOCK_UN = 4

        def flock(self, _fd: int, operation: int) -> None:
            if operation & self.LOCK_SH and not operation & self.LOCK_UN:
                raise BlockingIOError("writer active")

    monkeypatch.setattr(
        cli, "time", SimpleNamespace(sleep=lambda _delay: None), raising=False
    )
    monkeypatch.setattr(cli, "fcntl", ExclusiveGuard(), raising=False)
    monkeypatch.setattr(cli, "_CACHE_PUBLICATION_MAX_ATTEMPTS", 1, raising=False)

    with pytest.raises(FileNotFoundError, match="publication.*timed out"):
        _cache_trial_paths(tmp_path, ["1"])


def _write_generation_pointer(cache: Path, generation: str = "generation_old") -> Path:
    person = cache / "person_1"
    generation_dir = person / ".generations" / generation
    generation_dir.mkdir(parents=True)
    (generation_dir / "cycle_000.npz").touch()
    manifest = {
        "person_id": "1",
        "trials": ["cycle_000"],
        "generation": generation,
    }
    (generation_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    person.mkdir(parents=True, exist_ok=True)
    (person / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return person


def test_cache_paths_reads_old_generation_while_new_writer_holds_lock(
    tmp_path: Path,
) -> None:
    person = _write_generation_pointer(tmp_path)
    descriptor = os.open(person / ".publishing.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert _cache_trial_paths(tmp_path, ["1"]) == {
            "1": [person / ".generations" / "generation_old" / "cycle_000.npz"]
        }
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_cache_paths_waits_for_first_generation_pointer_then_reads_it(
    tmp_path: Path, monkeypatch
) -> None:
    person = tmp_path / "person_1"
    person.mkdir(parents=True)
    lock = person / ".publishing.lock"
    lock.write_text("first-writer", encoding="utf-8")
    waits: list[float] = []

    class FirstWriterGuard:
        LOCK_SH = 1
        LOCK_NB = 2
        LOCK_UN = 4

        def __init__(self) -> None:
            self.active = True

        def flock(self, _fd: int, operation: int) -> None:
            if operation & self.LOCK_SH and not operation & self.LOCK_UN and self.active:
                raise BlockingIOError("writer active")

    def publish_generation(delay: float) -> None:
        waits.append(delay)
        generation_dir = person / ".generations" / "generation_first"
        generation_dir.mkdir(parents=True)
        (generation_dir / "cycle_000.npz").touch()
        manifest = {
            "person_id": "1",
            "trials": ["cycle_000"],
            "generation": "generation_first",
        }
        (generation_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (person / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        guard.active = False

    guard = FirstWriterGuard()
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(sleep=publish_generation), raising=False
    )
    monkeypatch.setattr(cli, "fcntl", guard, raising=False)
    monkeypatch.setattr(cli, "_CACHE_PUBLICATION_MAX_ATTEMPTS", 2, raising=False)

    assert _cache_trial_paths(tmp_path, ["1"]) == {
        "1": [person / ".generations" / "generation_first" / "cycle_000.npz"]
    }
    assert waits


def test_cache_paths_rechecks_pointer_after_shared_guard_interleaving(
    tmp_path: Path, monkeypatch
) -> None:
    person = tmp_path / "person_1"
    person.mkdir(parents=True)
    (person / ".publishing.lock").touch()

    class SharedGuard:
        LOCK_SH = 1
        LOCK_NB = 2
        LOCK_UN = 4

        def __init__(self) -> None:
            self.published = False

        def flock(self, _fd: int, operation: int) -> None:
            if operation & self.LOCK_SH and not operation & self.LOCK_UN:
                if not self.published:
                    self.published = True
                    generation_dir = person / ".generations" / "generation_race"
                    generation_dir.mkdir(parents=True)
                    (generation_dir / "cycle_000.npz").touch()
                    manifest = {
                        "person_id": "1",
                        "trials": ["cycle_000"],
                        "generation": "generation_race",
                    }
                    (generation_dir / "manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    (person / "manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )

    guard = SharedGuard()
    monkeypatch.setattr(cli, "fcntl", guard, raising=False)
    monkeypatch.setattr(
        cli,
        "time",
        SimpleNamespace(
            sleep=lambda _delay: (_ for _ in ()).throw(
                AssertionError("reader should recheck under the shared guard")
            )
        ),
        raising=False,
    )

    assert _cache_trial_paths(tmp_path, ["1"]) == {
        "1": [person / ".generations" / "generation_race" / "cycle_000.npz"]
    }
    assert guard.published


def test_cache_paths_reports_missing_only_after_shared_guard_check(
    tmp_path: Path, monkeypatch
) -> None:
    person = tmp_path / "person_1"
    person.mkdir(parents=True)
    (person / ".publishing.lock").touch()
    shared_operations: list[int] = []

    class SharedGuard:
        LOCK_SH = 1
        LOCK_NB = 2
        LOCK_UN = 4

        def flock(self, _fd: int, operation: int) -> None:
            shared_operations.append(operation)

    guard = SharedGuard()
    monkeypatch.setattr(cli, "fcntl", guard, raising=False)
    monkeypatch.setattr(
        cli,
        "time",
        SimpleNamespace(
            sleep=lambda _delay: (_ for _ in ()).throw(
                AssertionError("no active writer should not wait")
            )
        ),
        raising=False,
    )

    with pytest.raises(FileNotFoundError, match="person_1"):
        _cache_trial_paths(tmp_path, ["1"])
    assert any(operation & guard.LOCK_SH for operation in shared_operations)


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
        # old_fuse_root must be pinned inside tmp_path: it otherwise defaults to
        # the repository's real logs/fuse_experiments and this test then picks up
        # the nine deterministic methods stored there for person_1.
        f"paths:\n  sam3d_root: {tmp_path / 'sam3d'}\n  split_cycle_root: {tmp_path / 'split'}\n  output_root: {out}\n  skeleton: configs/fuse/skeleton_mhr70.yaml\n  old_fuse_root: {tmp_path / 'old_fuse'}\n  fold_json: {tmp_path / 'fold.json'}"
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
