"""Unity benchmark stages for the extrinsic-conditioned fusion models
(``train-extrinsic``, ``train-extrinsic-matrix``, ``evaluate-extrinsic``,
``report-extrinsic``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .dataset import load_unity_benchmark
from .extrinsic_evaluation import (
    evaluate_extrinsic_runs,
    write_extrinsic_report,
)
from .extrinsic_training import (
    EXTRINSIC_METHODS,
    ExtrinsicTrainingConfig,
    _run_contract as _extrinsic_run_contract,
    build_extrinsic_sequences,
    run_extrinsic_inference,
    train_extrinsic_run,
    validate_extrinsic_run,
)
from .supervised_data import UNITY_SUPERVISED_FOLDS
from fusion.benchmarks.unity.config import (
    _git_commit,
    _required_mapping,
)
from fusion.benchmarks.unity.stages_supervised import _supervised_context


def _extrinsic_settings(
    config: Mapping[str, object],
    *,
    device: str | None = None,
) -> tuple[
    Mapping[str, object],
    Path,
    Path,
    Path,
    ExtrinsicTrainingConfig,
]:
    base, dataset_root, zero_shot_root, _, _ = _supervised_context(config)
    paths = _required_mapping(config, "paths")
    if "extrinsic_output_root" not in paths:
        raise ValueError(
            "Unity supervised config requires paths.extrinsic_output_root"
        )
    output_root = Path(str(paths["extrinsic_output_root"]))
    raw = _required_mapping(config, "extrinsic")
    methods = tuple(str(value) for value in raw.get("methods", ()))
    folds = tuple(str(value) for value in raw.get("folds", ()))
    seeds = tuple(int(value) for value in raw.get("seeds", ()))
    if set(methods) != set(EXTRINSIC_METHODS) or len(methods) != 3:
        raise ValueError("extrinsic matrix must contain exactly three methods")
    if set(folds) != set(UNITY_SUPERVISED_FOLDS) or len(folds) != 2:
        raise ValueError("extrinsic matrix must contain exactly two folds")
    if set(seeds) != {0, 1, 2} or len(seeds) != 3:
        raise ValueError("extrinsic matrix must contain exactly seeds 0,1,2")
    training = ExtrinsicTrainingConfig(
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        hidden_channels=int(raw["hidden_channels"]),
        max_delta_m=float(raw["max_delta_m"]),
        smooth_l1_beta_m=float(raw["smooth_l1_beta_m"]),
        device=str(device or raw["device"]),
    )
    return base, dataset_root, zero_shot_root, output_root, training


def _extrinsic_cells(
    config: Mapping[str, object],
) -> tuple[tuple[str, str, int], ...]:
    raw = _required_mapping(config, "extrinsic")
    return tuple(
        (str(method), str(fold), int(seed))
        for fold in raw["folds"]
        for method in raw["methods"]
        for seed in raw["seeds"]
    )


def _train_one_extrinsic(
    config: Mapping[str, object],
    *,
    method: str,
    fold_name: str,
    seed: int,
    device: str | None,
    sequences=None,
):
    base, dataset_root, zero_shot_root, output_root, training = (
        _extrinsic_settings(config, device=device)
    )
    benchmark = load_unity_benchmark(dataset_root)
    prepared = sequences or build_extrinsic_sequences(
        benchmark, zero_shot_root / "sam3d"
    )
    fold = UNITY_SUPERVISED_FOLDS[fold_name]
    return train_extrinsic_run(
        prepared[fold.train_sequence],
        method=method,
        fold=fold,
        seed=seed,
        output_root=output_root,
        config=training,
    )


def _train_extrinsic(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    run = _train_one_extrinsic(
        config,
        method=args.method,
        fold_name=args.fold,
        seed=args.seed,
        device=args.device,
    )
    print(f"run_root={run.run_root}")
    print(f"valid={validate_extrinsic_run(run)}")
    return 0


def _train_extrinsic_matrix(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    _, dataset_root, zero_shot_root, output_root, training = (
        _extrinsic_settings(config, device=args.device)
    )
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_extrinsic_sequences(
        benchmark, zero_shot_root / "sam3d"
    )
    counts = {"completed": 0, "reused": 0, "failed": 0}
    for method, fold_name, seed in _extrinsic_cells(config):
        fold = UNITY_SUPERVISED_FOLDS[fold_name]
        run = _extrinsic_run_contract(
            output_root, method, fold, seed
        )
        if validate_extrinsic_run(run):
            counts["reused"] += 1
            continue
        try:
            train_extrinsic_run(
                sequences[fold.train_sequence],
                method=method,
                fold=fold,
                seed=seed,
                output_root=output_root,
                config=training,
            )
        except Exception as error:
            counts["failed"] += 1
            print(
                f"failed={method}/{fold_name}/seed_{seed}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        counts["completed"] += 1
        print(f"completed={method}/{fold_name}/seed_{seed}")
    print(
        f"completed={counts['completed']} reused={counts['reused']} "
        f"failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


def _evaluate_extrinsic_artifacts(
    config: Mapping[str, object],
    *,
    run_missing_inference: bool,
) -> Path:
    _, dataset_root, zero_shot_root, output_root, _ = _extrinsic_settings(
        config
    )
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_extrinsic_sequences(
        benchmark, zero_shot_root / "sam3d"
    )
    runs = []
    for method, fold_name, seed in _extrinsic_cells(config):
        fold = UNITY_SUPERVISED_FOLDS[fold_name]
        run = _extrinsic_run_contract(output_root, method, fold, seed)
        if not validate_extrinsic_run(run):
            raise RuntimeError(f"invalid or incomplete extrinsic run: {run.run_root}")
        expected = tuple(
            run.run_root / "inference" / f"{sequence_id}.npz"
            for sequence_id in (run.test_sequence, "static_sweep")
        )
        if not all(path.is_file() for path in expected):
            if not run_missing_inference:
                raise FileNotFoundError(
                    f"missing extrinsic inference below {run.run_root}"
                )
            run_extrinsic_inference(run, sequences, device="cpu")
        runs.append(run)
    heldout, static = evaluate_extrinsic_runs(benchmark, runs)
    return write_extrinsic_report(
        heldout,
        static_rows=static,
        output_root=output_root,
        provenance={
            "git_commit": _git_commit(),
            "protocol": "direction-held-out-2x3",
            "runs": len(runs),
            "alignment": "one_sim3_per_sequence",
            "unity_gt_supervision": True,
            "exact_camera_geometry": True,
        },
        baseline_results=json.loads(
            (zero_shot_root / "report/results.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def _evaluate_extrinsic(config: Mapping[str, object]) -> int:
    report = _evaluate_extrinsic_artifacts(
        config, run_missing_inference=True
    )
    print(f"report={report}")
    return 0


def _report_extrinsic(config: Mapping[str, object]) -> int:
    report = _evaluate_extrinsic_artifacts(
        config, run_missing_inference=False
    )
    print(f"report={report}")
    return 0

