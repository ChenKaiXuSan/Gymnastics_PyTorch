"""Unity benchmark stages for the camera-feature-guided fusion ablations
(``train-camera-feature``, ``train-camera-feature-matrix``,
``evaluate-camera-feature``, ``report-camera-feature``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .camera_guided_data import (
    CAMERA_GUIDED_ABLATIONS,
    build_camera_guided_sequences,
)
from .camera_guided_evaluation import (
    evaluate_camera_guided_runs,
    write_camera_guided_report,
)
from .camera_guided_training import (
    CameraGuidedRun,
    CameraGuidedTrainingConfig,
    _run_contract as _camera_run_contract,
    run_camera_guided_inference,
    train_camera_guided_run,
)
from .dataset import load_unity_benchmark
from .supervised_data import UNITY_SUPERVISED_FOLDS
from fusion.benchmarks.unity.config import (
    _load_config,
    _paths,
    _required_mapping,
)


def _camera_feature_context(
    config: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Path,
    Path,
    Path,
    Path,
    float,
]:
    base_path = config.get("base_config")
    if not isinstance(base_path, str) or not base_path:
        raise ValueError("Unity camera-feature config requires base_config")
    base = _load_config(Path(base_path))
    dataset_root, benchmark_output_root = _paths(base)
    paths = _required_mapping(config, "paths")
    settings = _required_mapping(config, "camera_feature")
    output_root = Path(str(paths["camera_feature_output_root"]))
    skeleton_path = Path(str(paths["skeleton"]))
    checkpoints = _required_mapping(base, "checkpoints")
    source_checkpoint = Path(str(checkpoints["A6"]))
    fps = float(_required_mapping(base, "data")["fps"])
    return (
        base,
        settings,
        dataset_root,
        benchmark_output_root / "sam3d",
        output_root,
        skeleton_path,
        fps,
    )


def _camera_feature_cells(
    settings: Mapping[str, object],
) -> tuple[tuple[str, str, int], ...]:
    ablations = tuple(str(value) for value in settings["ablations"])
    folds = tuple(str(value) for value in settings["folds"])
    seeds = tuple(int(value) for value in settings["seeds"])
    if set(ablations) != set(CAMERA_GUIDED_ABLATIONS) or len(ablations) != 6:
        raise ValueError("camera-feature matrix must contain exactly G0--G5")
    if set(folds) != {"left_to_right", "right_to_left"} or len(folds) != 2:
        raise ValueError("camera-feature matrix requires two direction folds")
    if set(seeds) != {0, 1, 2} or len(seeds) != 3:
        raise ValueError("camera-feature matrix requires seeds 0,1,2")
    return tuple(
        (ablation, fold, seed)
        for fold in folds
        for ablation in ablations
        for seed in seeds
    )


def _camera_training_config(
    settings: Mapping[str, object], device: str | None
) -> CameraGuidedTrainingConfig:
    return CameraGuidedTrainingConfig(
        epochs=int(settings["epochs"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        window_length=int(settings["window_length"]),
        train_stride=int(settings["train_stride"]),
        batch_size=int(settings["batch_size"]),
        device=str(device or settings["device"]),
    )


def _camera_run_for_cell(
    output_root: Path, ablation: str, fold_name: str, seed: int
) -> CameraGuidedRun:
    return _camera_run_contract(
        output_root=output_root,
        ablation=ablation,
        fold=UNITY_SUPERVISED_FOLDS[fold_name],
        seed=seed,
    )


def _camera_cell_complete(run: CameraGuidedRun) -> bool:
    if not all(
        path.is_file()
        for path in (
            run.final_checkpoint,
            run.history_path,
            run.provenance_path,
            run.run_root / "inference" / f"{run.test_sequence}.npz",
            run.run_root / "inference" / "static_sweep.npz",
        )
    ):
        return False
    provenance = json.loads(
        run.provenance_path.read_text(encoding="utf-8")
    )
    return (
        provenance.get("ablation") == run.ablation
        and provenance.get("fold") == run.fold
        and provenance.get("seed") == run.seed
        and provenance.get("unity_native_3d_available_to_training") is False
    )


def _run_one_camera_feature(
    config: Mapping[str, object],
    *,
    ablation: str,
    fold_name: str,
    seed: int,
    device: str | None,
) -> CameraGuidedRun:
    (
        _,
        settings,
        dataset_root,
        sam3d_root,
        output_root,
        skeleton_path,
        fps,
    ) = _camera_feature_context(config)
    fold = UNITY_SUPERVISED_FOLDS[fold_name]
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_camera_guided_sequences(
        benchmark,
        sam3d_root,
        skeleton_path=skeleton_path,
        fps=fps,
        fold=fold,
        ablation=ablation,
        threshold_px=float(settings["threshold_px"]),
    )
    base_path = Path(str(config["base_config"]))
    base = _load_config(base_path)
    source_checkpoint = Path(
        str(_required_mapping(base, "checkpoints")["A6"])
    )
    run = train_camera_guided_run(
        sequences[fold.train_sequence],
        ablation=ablation,
        fold=fold,
        seed=seed,
        source_checkpoint=source_checkpoint,
        skeleton_path=skeleton_path,
        output_root=output_root,
        config=_camera_training_config(settings, device),
    )
    run_camera_guided_inference(
        run,
        sequences,
        skeleton_path=skeleton_path,
        window_length=int(settings["window_length"]),
        stride=int(settings["inference_stride"]),
        device=str(device or settings["device"]),
    )
    return run


def _train_camera_feature(
    args: argparse.Namespace, config: Mapping[str, object]
) -> int:
    run = _run_one_camera_feature(
        config,
        ablation=str(args.ablation),
        fold_name=str(args.fold),
        seed=int(args.seed),
        device=args.device,
    )
    print(f"camera_feature_run={run.run_root}")
    return 0


def _train_camera_feature_matrix(
    args: argparse.Namespace, config: Mapping[str, object]
) -> int:
    _, settings, _, _, output_root, _, _ = _camera_feature_context(config)
    counts = {"completed": 0, "reused": 0, "failed": 0}
    for ablation, fold_name, seed in _camera_feature_cells(settings):
        run = _camera_run_for_cell(
            output_root, ablation, fold_name, seed
        )
        if _camera_cell_complete(run):
            counts["reused"] += 1
            continue
        try:
            _run_one_camera_feature(
                config,
                ablation=ablation,
                fold_name=fold_name,
                seed=seed,
                device=args.device,
            )
        except Exception as error:
            counts["failed"] += 1
            print(
                f"failed={ablation}/{fold_name}/seed_{seed}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        counts["completed"] += 1
    print(f"camera_feature_matrix={counts}")
    return 1 if counts["failed"] else 0


def _camera_feature_runs(
    config: Mapping[str, object],
) -> tuple[CameraGuidedRun, ...]:
    _, settings, _, _, output_root, _, _ = _camera_feature_context(config)
    runs = tuple(
        _camera_run_for_cell(output_root, ablation, fold, seed)
        for ablation, fold, seed in _camera_feature_cells(settings)
    )
    incomplete = [run.run_root for run in runs if not _camera_cell_complete(run)]
    if incomplete:
        raise ValueError(
            f"camera-feature matrix has {len(incomplete)} incomplete cells"
        )
    return runs


def _evaluate_camera_feature(config: Mapping[str, object]) -> int:
    (
        _,
        _,
        dataset_root,
        _,
        output_root,
        _,
        _,
    ) = _camera_feature_context(config)
    runs = _camera_feature_runs(config)
    benchmark = load_unity_benchmark(dataset_root)
    rows = evaluate_camera_guided_runs(benchmark, runs)
    outputs = write_camera_guided_report(
        output_root / "evaluation",
        run_rows=rows,
        provenance={
            "runs": len(runs),
            "unity_native_3d_loaded_after_training": True,
        },
    )
    print(f"camera_feature_report={outputs['report']}")
    return 0

