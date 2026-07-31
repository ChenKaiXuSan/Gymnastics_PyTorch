"""Standalone runner for the collected-data fitted-camera pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import load_skeleton_spec
from .data import load_cached_trial, resolve_cache_manifest
from .inference import canonicalize_trial
from .real_camera_data import (
    RealCameraTrial,
    load_real_camera_trials,
    prepare_real_camera_observation_cache,
)
from .real_camera_training import (
    RealCameraRun,
    RealCameraTrainingConfig,
    infer_real_camera_cell,
    train_real_camera_cell,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train-matrix")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seed", type=int, action="append")
    train.add_argument("--device", default="cpu")
    prepare = subparsers.add_parser("prepare-inputs")
    prepare.add_argument("--config", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--config", type=Path, required=True)
    return parser


def _config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Pilot configuration must be a mapping")
    return value


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    values = config.get("paths")
    if not isinstance(values, Mapping):
        raise ValueError("Configuration requires paths")
    required = (
        "skeleton",
        "fold",
        "cache_root",
        "sam3d_person_root",
        "camera_audit",
        "face_calibration",
        "side_calibration",
        "triangulated_root",
        "output_root",
        "observation_cache",
    )
    paths = {name: Path(str(values[name])) for name in required}
    generated = {"output_root", "observation_cache"}
    missing = [
        str(path)
        for name, path in paths.items()
        if name not in generated and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Configured inputs do not exist: {missing}")
    return paths


def _split(path: Path) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = {
        name: tuple(str(person) for person in payload[name])
        for name in ("train", "val", "test")
    }
    if any(set(result[left]) & set(result[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Fold partitions overlap")
    return result


def _raw_trials(
    cache_root: Path, people: Sequence[str]
) -> list:
    trials = []
    for person_id in people:
        person_dir = cache_root / f"person_{person_id}"
        _, manifest = resolve_cache_manifest(person_dir)
        trial_ids = manifest.get("trials")
        if not isinstance(trial_ids, list) or not trial_ids:
            raise ValueError(f"No declared cache trials for person {person_id}")
        for trial_id in trial_ids:
            trial, _ = load_cached_trial(person_dir, str(trial_id))
            trials.append(trial)
    return trials


def _camera_trials(
    canonical_trials: Sequence,
    *,
    paths: Mapping[str, Path],
    skeleton,
    ablation: str,
) -> list[RealCameraTrial]:
    return load_real_camera_trials(
        raw_trials=canonical_trials,
        skeleton=skeleton,
        sam3d_person_root=paths["sam3d_person_root"],
        camera_audit_path=paths["camera_audit"],
        face_calibration_path=paths["face_calibration"],
        side_calibration_path=paths["side_calibration"],
        observation_cache_root=paths["observation_cache"],
        ablation=ablation,
    )


def _training_config(
    config: Mapping[str, Any], device: str
) -> RealCameraTrainingConfig:
    value = config.get("training")
    if not isinstance(value, Mapping):
        raise ValueError("Configuration requires training settings")
    return RealCameraTrainingConfig(
        epochs=int(value["epochs"]),
        batch_size=int(value["batch_size"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        window_length=int(value["window_length"]),
        train_stride=int(value["train_stride"]),
        eval_stride=int(value["eval_stride"]),
        device=device,
    )


def _source_checkpoints(config: Mapping[str, Any]) -> dict[int, Path]:
    values = config.get("source_checkpoints")
    if not isinstance(values, Mapping):
        raise ValueError("Configuration requires source checkpoints")
    result = {int(seed): Path(str(path)) for seed, path in values.items()}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source checkpoints are missing: {missing}")
    return result


def _declared_matrix(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("Configuration requires experiment settings")
    ablations = tuple(str(value) for value in experiment["ablations"])
    seeds = tuple(int(value) for value in experiment["seeds"])
    if ablations != ("G0", "G1", "G2", "G3", "G4", "G5"):
        raise ValueError("Pilot must declare exactly G0--G5")
    if len(set(seeds)) != 3:
        raise ValueError("Pilot must declare exactly three unique seeds")
    return ablations, seeds


def _run_from_root(
    output_root: Path, ablation: str, seed: int
) -> RealCameraRun:
    root = output_root / ablation / f"seed_{seed}"
    return RealCameraRun(
        ablation=ablation,
        seed=seed,
        run_root=root,
        checkpoint=root / "best.pt",
        history_path=root / "history.json",
        provenance_path=root / "provenance.json",
    )


def _train_matrix(
    config: Mapping[str, Any],
    *,
    selected_seeds: Sequence[int] | None,
    device: str,
) -> None:
    paths = _paths(config)
    ablations, declared_seeds = _declared_matrix(config)
    seeds = tuple(selected_seeds) if selected_seeds else declared_seeds
    if not seeds or not set(seeds) <= set(declared_seeds):
        raise ValueError("Selected seed is outside the declared matrix")
    checkpoints = _source_checkpoints(config)
    split = _split(paths["fold"])
    skeleton = load_skeleton_spec(paths["skeleton"])
    raw_by_split = {
        name: _raw_trials(paths["cache_root"], people)
        for name, people in split.items()
    }
    canonical_by_split = {
        name: [canonicalize_trial(trial, skeleton) for trial in trials]
        for name, trials in raw_by_split.items()
    }
    training = _training_config(config, device)
    expected_test_cycles = len(canonical_by_split["test"])

    for ablation in ablations:
        print(
            json.dumps(
                {"event": "build_features", "ablation": ablation},
                sort_keys=True,
            ),
            flush=True,
        )
        trials = {
            name: _camera_trials(
                values,
                paths=paths,
                skeleton=skeleton,
                ablation=ablation,
            )
            for name, values in canonical_by_split.items()
        }
        for seed in seeds:
            run = _run_from_root(paths["output_root"], ablation, seed)
            if not run.checkpoint.is_file() or not run.provenance_path.is_file():
                print(
                    json.dumps(
                        {
                            "event": "train_start",
                            "ablation": ablation,
                            "seed": seed,
                            "device": device,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                run = train_real_camera_cell(
                    train_trials=trials["train"],
                    val_trials=trials["val"],
                    ablation=ablation,
                    seed=seed,
                    source_checkpoint=checkpoints[seed],
                    skeleton_path=paths["skeleton"],
                    output_root=paths["output_root"],
                    config=training,
                )
            existing = tuple(
                (run.run_root / "inference").glob(
                    "person_*/*/fused_sequence.npz"
                )
            )
            if len(existing) != expected_test_cycles:
                print(
                    json.dumps(
                        {
                            "event": "inference_start",
                            "ablation": ablation,
                            "seed": seed,
                            "expected_cycles": expected_test_cycles,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                infer_real_camera_cell(
                    run,
                    test_trials=trials["test"],
                    skeleton_path=paths["skeleton"],
                    window_length=training.window_length,
                    stride=training.eval_stride,
                    device=device,
                )
            print(
                json.dumps(
                    {"event": "cell_complete", "ablation": ablation, "seed": seed},
                    sort_keys=True,
                ),
                flush=True,
            )


def _prepare_inputs(config: Mapping[str, Any]) -> None:
    paths = _paths(config)
    split = _split(paths["fold"])
    trials = [
        trial
        for people in split.values()
        for trial in _raw_trials(paths["cache_root"], people)
    ]
    print(
        json.dumps(
            {"event": "prepare_inputs_start", "cycles": len(trials)},
            sort_keys=True,
        ),
        flush=True,
    )
    outputs = prepare_real_camera_observation_cache(
        raw_trials=trials,
        sam3d_person_root=paths["sam3d_person_root"],
        face_calibration_path=paths["face_calibration"],
        side_calibration_path=paths["side_calibration"],
        output_root=paths["observation_cache"],
    )
    print(
        json.dumps(
            {"event": "prepare_inputs_complete", "cycles": len(outputs)},
            sort_keys=True,
        ),
        flush=True,
    )


def _camera_audit_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = list(payload["persons"].values())
    holdout = np.asarray(
        [record["holdout_reproj_px"] for record in records], dtype=np.float64
    )
    return {
        "people": len(records),
        "median_holdout_reprojection_px": float(np.median(holdout)),
        "fixed_rig_holdout_reprojection_px": 23.3569,
    }


def _evaluate(config: Mapping[str, Any]) -> None:
    # Delayed import keeps triangulated-reference code outside training.
    from .real_camera_evaluation import evaluate_real_camera_runs

    paths = _paths(config)
    ablations, seeds = _declared_matrix(config)
    runs = [
        _run_from_root(paths["output_root"], ablation, seed)
        for ablation in ablations
        for seed in seeds
    ]
    missing = [
        str(run.run_root)
        for run in runs
        if not run.checkpoint.is_file() or not run.provenance_path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Matrix is incomplete: {missing}")
    experiment = config["experiment"]
    summary = evaluate_real_camera_runs(
        runs,
        triangulated_root=paths["triangulated_root"],
        skeleton_path=paths["skeleton"],
        output_root=paths["output_root"] / "evaluation",
        camera_audit=_camera_audit_summary(paths["camera_audit"]),
        bootstrap_samples=int(experiment["bootstrap_samples"]),
    )
    print(
        json.dumps(
            {
                "event": "evaluation_complete",
                "camera_claim_supported": summary.camera_claim_supported,
                "methods": summary.method_rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args.config)
    if args.command == "prepare-inputs":
        _prepare_inputs(config)
    elif args.command == "train-matrix":
        _train_matrix(config, selected_seeds=args.seed, device=args.device)
    elif args.command == "evaluate":
        _evaluate(config)
    elif args.command == "report":
        target = _paths(config)["output_root"] / "evaluation" / "real_camera_feature_report.md"
        if not target.is_file():
            raise FileNotFoundError(f"Evaluation report does not exist: {target}")
        print(target.read_text(encoding="utf-8"))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
