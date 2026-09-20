"""Unity benchmark stages that fine-tune the archived rotation-aware model on
Unity supervision (``finetune``, ``finetune-matrix``, ``evaluate-finetuned``,
``report-finetuned``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import (
    Mapping,
    Sequence,
)

import torch

from fusion.archive.rotation_aware.corruptions import CorruptionConfig
from fusion.archive.rotation_aware.losses import LossConfig
from .dataset import load_unity_benchmark
from .supervised import (
    UnityFineTuneConfig,
    _resolved_config,
    _run_contract,
    _sha256_file,
    discover_completed_runs,
    run_finetuned_inference,
    run_supervised_finetune,
    validate_completed_run,
)
from .supervised_data import (
    UNITY_SUPERVISED_FOLDS,
    UnitySupervisedSequence,
    build_supervised_sequence,
    build_supervised_sequences,
)
from .supervised_evaluation import (
    build_finetuned_bundle,
    evaluate_finetuned_runs,
    write_finetuned_report,
)
from .supervised_loss import UnitySupervisedLossConfig
from fusion.benchmarks.unity.config import (
    _checkpoints,
    _data_fps,
    _git_commit,
    _load_config,
    _paths,
    _required_mapping,
)


def _supervised_context(
    config: Mapping[str, object],
) -> tuple[Mapping[str, object], Path, Path, Path, Path]:
    base_path = config.get("base_config")
    if not isinstance(base_path, str) or not base_path:
        raise ValueError("Unity supervised config requires base_config")
    base = _load_config(Path(base_path))
    dataset_root, zero_shot_root = _paths(base)
    supervised_paths = _required_mapping(config, "paths")
    output_root = Path(str(supervised_paths["output_root"]))
    skeleton_path = Path(str(supervised_paths["skeleton"]))
    return base, dataset_root, zero_shot_root, output_root, skeleton_path


def _supervised_matrix_cells(
    config: Mapping[str, object] | None = None,
) -> tuple[tuple[str, str, int], ...]:
    if config is None:
        ablations = ("A4", "A5", "A6", "A7", "A8", "A9")
        folds = ("left_to_right", "right_to_left")
        seeds = (0, 1, 2)
    else:
        matrix = _required_mapping(config, "matrix")
        ablations = tuple(str(value) for value in matrix.get("ablations", ()))
        folds = tuple(str(value) for value in matrix.get("folds", ()))
        seeds = tuple(int(value) for value in matrix.get("seeds", ()))
        if set(ablations) != {"A4", "A5", "A6", "A7", "A8", "A9"} or len(
            ablations
        ) != 6:
            raise ValueError("supervised matrix must contain exactly A4--A9")
        if set(folds) != {"left_to_right", "right_to_left"} or len(folds) != 2:
            raise ValueError(
                "supervised matrix must contain exactly the two direction folds"
            )
        if set(seeds) != {0, 1, 2} or len(seeds) != 3:
            raise ValueError("supervised matrix must contain exactly seeds 0,1,2")
    return tuple(
        (ablation, fold, seed)
        for fold in folds
        for ablation in ablations
        for seed in seeds
    )


def _supervised_cell_is_complete(cell: tuple[str, str, int]) -> bool:
    """Default test seam; real matrix execution supplies a strict validator."""
    return False


def _run_supervised_cell(cell: tuple[str, str, int]) -> None:
    """Default test seam; real matrix execution supplies the cell runner."""
    raise RuntimeError(f"no Unity supervised cell runner configured for {cell}")


def _dispatch_supervised_matrix(
    cells: Sequence[tuple[str, str, int]],
    *,
    is_complete=None,
    run_cell=None,
) -> dict[str, int]:
    validator = is_complete or _supervised_cell_is_complete
    runner = run_cell or _run_supervised_cell
    counts = {"completed": 0, "reused": 0, "failed": 0}
    for cell in cells:
        if validator(cell):
            counts["reused"] += 1
            continue
        try:
            runner(cell)
        except Exception as error:
            counts["failed"] += 1
            print(
                f"failed={cell[0]}/{cell[1]}/seed_{cell[2]}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        counts["completed"] += 1
    return counts


def _fine_tune_config(
    config: Mapping[str, object],
    *,
    device: str | None,
) -> UnityFineTuneConfig:
    training = _required_mapping(config, "training")
    window = _required_mapping(config, "window")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ValueError("Unity supervised optimizer must be adamw")
    return UnityFineTuneConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        window_length=int(window["length"]),
        train_stride=int(window["train_stride"]),
        device=str(device or training["device"]),
    )


def _unity_loss_config(
    config: Mapping[str, object],
) -> UnitySupervisedLossConfig:
    loss = _required_mapping(config, "loss")
    return UnitySupervisedLossConfig(
        unity_3d_weight=float(loss["unity_3d_weight"]),
        self_supervised_weight=float(loss["self_supervised_weight"]),
        smooth_l1_beta_m=float(loss["smooth_l1_beta_m"]),
    )


def _source_training_configs(
    checkpoint: Path,
) -> tuple[LossConfig, CorruptionConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_loss = payload.get("loss_config")
    raw_corruption = payload.get("corruption_config")
    if not isinstance(raw_loss, Mapping) or not isinstance(
        raw_corruption, Mapping
    ):
        raise ValueError(f"source checkpoint lacks training configs: {checkpoint}")
    return LossConfig(**dict(raw_loss)), CorruptionConfig(**dict(raw_corruption))


def _training_sequence(
    *,
    dataset_root: Path,
    sam3d_root: Path,
    skeleton_path: Path,
    sequence_id: str,
    fps: float,
) -> UnitySupervisedSequence:
    benchmark = load_unity_benchmark(
        dataset_root,
        sequence_ids=(sequence_id,),
    )
    return build_supervised_sequence(
        benchmark,
        sam3d_root,
        sequence_id,
        skeleton_path=skeleton_path,
        fps=fps,
    )


def _run_one_finetune(
    config: Mapping[str, object],
    *,
    ablation: str,
    fold_name: str,
    seed: int,
    device: str | None,
    prepared_sequence: UnitySupervisedSequence | None = None,
):
    base, dataset_root, zero_shot_root, output_root, skeleton_path = (
        _supervised_context(config)
    )
    _supervised_matrix_cells(config)
    fold = UNITY_SUPERVISED_FOLDS[fold_name]
    checkpoints = _checkpoints(base, (ablation,))
    source_checkpoint = checkpoints[ablation]
    fine_config = _fine_tune_config(config, device=device)
    sequence = prepared_sequence or _training_sequence(
        dataset_root=dataset_root,
        sam3d_root=zero_shot_root / "sam3d",
        skeleton_path=skeleton_path,
        sequence_id=fold.train_sequence,
        fps=_data_fps(base),
    )
    self_config, corruption_config = _source_training_configs(source_checkpoint)
    return run_supervised_finetune(
        sequence,
        ablation=ablation,
        fold=fold,
        seed=seed,
        source_checkpoint=source_checkpoint,
        skeleton_path=skeleton_path,
        output_root=output_root,
        config=fine_config,
        loss_config=_unity_loss_config(config),
        self_supervised_config=self_config,
        corruption_config=corruption_config,
    )


def _finetune(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    run = _run_one_finetune(
        config,
        ablation=args.ablation,
        fold_name=args.fold,
        seed=args.seed,
        device=args.device,
    )
    provenance = json.loads(run.provenance_path.read_text(encoding="utf-8"))
    history = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    final = history[-1]
    print(f"run_root={run.run_root}")
    print(f"source_checkpoint_sha256={provenance['source_checkpoint_sha256']}")
    print(f"final_checkpoint_sha256={provenance['final_checkpoint_sha256']}")
    print(
        "final_losses="
        f"unity_3d:{float(final['unity_3d_loss']):.6f},"
        f"self_supervised:{float(final['self_supervised_loss']):.6f},"
        f"total:{float(final['total_loss']):.6f}"
    )
    return 0


def _finetune_matrix(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> int:
    base, dataset_root, zero_shot_root, output_root, skeleton_path = (
        _supervised_context(config)
    )
    cells = _supervised_matrix_cells(config)
    fine_config = _fine_tune_config(config, device=args.device)
    unity_loss = _unity_loss_config(config)
    sequences = {
        fold_name: _training_sequence(
            dataset_root=dataset_root,
            sam3d_root=zero_shot_root / "sam3d",
            skeleton_path=skeleton_path,
            sequence_id=fold.train_sequence,
            fps=_data_fps(base),
        )
        for fold_name, fold in UNITY_SUPERVISED_FOLDS.items()
    }

    def cell_contract(cell: tuple[str, str, int]):
        ablation, fold_name, seed = cell
        fold = UNITY_SUPERVISED_FOLDS[fold_name]
        checkpoint = _checkpoints(base, (ablation,))[ablation]
        self_config, corruption = _source_training_configs(checkpoint)
        resolved = _resolved_config(
            ablation=ablation,
            fold=fold,
            seed=seed,
            source_checkpoint=checkpoint,
            skeleton_path=skeleton_path,
            config=fine_config,
            loss_config=unity_loss,
            self_supervised_config=self_config,
            corruption_config=corruption,
        )
        run = _run_contract(
            ablation=ablation,
            fold=fold,
            seed=seed,
            output_root=output_root,
        )
        return run, checkpoint, resolved

    def completed(cell: tuple[str, str, int]) -> bool:
        run, checkpoint, resolved = cell_contract(cell)
        sequence = sequences[cell[1]]
        return validate_completed_run(
            run,
            source_checkpoint_sha256=_sha256_file(checkpoint),
            resolved_config=resolved,
            unity_manifest_sha256=str(
                sequence.raw_trial.source_metadata["unity_manifest_sha256"]
            ),
        )

    def execute(cell: tuple[str, str, int]) -> None:
        ablation, fold_name, seed = cell
        _run_one_finetune(
            config,
            ablation=ablation,
            fold_name=fold_name,
            seed=seed,
            device=args.device,
            prepared_sequence=sequences[fold_name],
        )
        print(f"completed={ablation}/{fold_name}/seed_{seed}")

    counts = _dispatch_supervised_matrix(
        cells,
        is_complete=completed,
        run_cell=execute,
    )
    print(
        f"completed={counts['completed']} reused={counts['reused']} "
        f"failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


def _evaluate_supervised_artifacts(
    config: Mapping[str, object],
    *,
    run_missing_inference: bool,
) -> Path:
    base, dataset_root, zero_shot_root, output_root, skeleton_path = (
        _supervised_context(config)
    )
    cells = _supervised_matrix_cells(config)
    runs = discover_completed_runs(
        output_root,
        expected_cells=cells,
        resolved_config={},
    )
    if len(runs) != len(cells):
        raise RuntimeError(
            f"expected {len(cells)} valid fine-tuned runs, found {len(runs)}"
        )
    benchmark = load_unity_benchmark(dataset_root)
    sequences = build_supervised_sequences(
        benchmark,
        zero_shot_root / "sam3d",
        skeleton_path=skeleton_path,
        fps=_data_fps(base),
    )
    evaluation = _required_mapping(config, "evaluation")
    window_length = int(evaluation["window_length"])
    stride = int(evaluation["stride"])
    for run in runs:
        expected = tuple(
            run.run_root / "inference" / f"{sequence_id}.npz"
            for sequence_id in (run.test_sequence, "static_sweep")
        )
        if all(path.is_file() for path in expected):
            continue
        if not run_missing_inference:
            raise FileNotFoundError(
                f"missing fine-tuned inference artifacts below {run.run_root}"
            )
        run_finetuned_inference(
            run,
            sequences,
            skeleton_path=skeleton_path,
            window_length=window_length,
            stride=stride,
            device="cpu",
        )
    results = evaluate_finetuned_runs(benchmark, runs, sequences)
    provenance = {
        "git_commit": _git_commit(),
        "protocol": "direction-held-out-2x3",
        "runs": len(runs),
        "alignment": str(evaluation["alignment"]),
        "unity_gt_supervision": True,
        "static_evaluation_only": True,
    }
    bundle = build_finetuned_bundle(
        results,
        failures=(),
        provenance=provenance,
    )
    baseline_path = zero_shot_root / "report/results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    return write_finetuned_report(
        bundle,
        output_root,
        baseline_results=baseline,
    )


def _evaluate_finetuned(config: Mapping[str, object]) -> int:
    report = _evaluate_supervised_artifacts(
        config, run_missing_inference=True
    )
    print(f"report={report}")
    return 0


def _report_finetuned(config: Mapping[str, object]) -> int:
    report = _evaluate_supervised_artifacts(
        config, run_missing_inference=False
    )
    print(f"report={report}")
    return 0

