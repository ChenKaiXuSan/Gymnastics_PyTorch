"""Measure rotation-aware training throughput on an already prepared cache.

The benchmark follows the training command's dataset, loss, validation, and
performance configuration paths without writing a run directory or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from fuse.rotation_aware.cli import (
    _cached_trials_with_provenance,
    _manifest_people,
    _paths,
    _training_config_for_ablation,
    load_config,
    loss_config_for_ablation,
    resolve_fold,
)
from fuse.rotation_aware.config import SkeletonSpec, load_skeleton_spec
from fuse.rotation_aware.corruptions import CorruptionConfig
from fuse.rotation_aware.dataset import (
    PosePairCompleteCycleDataset,
    PosePairWindowDataset,
    WindowConfig,
    build_split_manifest,
    collate_pose_pair_windows,
)
from fuse.rotation_aware.model import RotationAwareFusionModel
from fuse.rotation_aware.prefetch import ThroughputConfig
from fuse.rotation_aware.profiling import StageProfiler
from fuse.rotation_aware.training import (
    prepare_validation_batches,
    train_one_epoch,
    validate,
)


@dataclass
class BenchmarkWorkload:
    """All state needed to execute one training-command-equivalent epoch."""

    model: RotationAwareFusionModel
    optimizer: torch.optim.Optimizer
    skeleton: SkeletonSpec
    train_loader: DataLoader
    validation_loader: DataLoader
    complete_cycle_loader: DataLoader
    validation_complete_cycle_loader: DataLoader
    loss_config: Any
    corruption_config: CorruptionConfig
    throughput_config: ThroughputConfig
    training_config: dict[str, Any]
    prepared_validation_loader: list[dict[str, object]] | None
    prepared_validation_complete_cycle_loader: list[dict[str, object]] | None
    train_window_count: int
    train_complete_cycle_count: int
    validation_window_count: int
    validation_complete_cycle_count: int


def make_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Rotation-aware YAML config")
    parser.add_argument("--ablation", required=True, choices=("A4", "A5", "A6"))
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (defaults to training.device from the config)",
    )
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--measured-epochs", type=int, default=3)
    parser.add_argument("--output", required=True, help="Destination JSON path")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if args.measured_epochs < 2:
        raise ValueError("measured_epochs must be at least two")


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def _build_workload(
    config: Mapping[str, Any], ablation: str
) -> BenchmarkWorkload:
    training = _training_config_for_ablation(config, ablation)
    performance = config.get("performance", {})
    if not isinstance(performance, Mapping):
        raise ValueError("rotation-aware performance config must be a mapping")
    throughput = ThroughputConfig(**dict(performance))
    paths = _paths(config, None)
    skeleton = load_skeleton_spec(paths["skeleton"])
    manifest = build_split_manifest(resolve_fold(config, None))
    selected = _manifest_people(manifest, None)
    trials, _ = _cached_trials_with_provenance(paths["cache"], selected, skeleton)
    actual_people = tuple(sorted({trial.person_id for trial in trials}))
    if actual_people != selected:
        raise ValueError("cached trial people do not exactly match the selected fold people")

    window = WindowConfig(**dict(config.get("window", {})))
    train_trials = [trial for trial in trials if trial.person_id in manifest.train]
    val_trials = [trial for trial in trials if trial.person_id in manifest.val]
    if not train_trials:
        raise ValueError("no selected canonical trials belong to the training split")
    train_set = PosePairWindowDataset(
        train_trials, skeleton=skeleton, manifest=manifest, split="train", config=window
    )
    val_set = (
        PosePairWindowDataset(
            val_trials, skeleton=skeleton, manifest=manifest, split="val", config=window
        )
        if val_trials
        else None
    )
    train_cycles = PosePairCompleteCycleDataset(
        train_trials, skeleton=skeleton, manifest=manifest, split="train"
    )
    validation_cycle_trials = val_trials or train_trials
    validation_cycle_split = "val" if val_trials else "train"
    validation_cycles = PosePairCompleteCycleDataset(
        validation_cycle_trials,
        skeleton=skeleton,
        manifest=manifest,
        split=validation_cycle_split,
    )
    batch_size = int(training.get("batch_size", 4))
    generator = torch.Generator().manual_seed(int(training.get("seed", 0)))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_pose_pair_windows,
    )
    validation_loader = DataLoader(
        val_set if val_set is not None else train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pose_pair_windows,
    )
    complete_cycle_loader = DataLoader(
        train_cycles,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collate_pose_pair_windows,
    )
    validation_complete_cycle_loader = DataLoader(
        validation_cycles,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_pose_pair_windows,
    )
    corruption = CorruptionConfig()
    prepared_validation_loader = None
    prepared_validation_complete_cycle_loader = None
    if throughput.cache_validation_batches:
        prepared_validation_loader = prepare_validation_batches(
            validation_loader,
            skeleton,
            seed=int(training.get("seed", 0)),
            corruption_config=corruption,
            throughput_config=throughput,
        )
        prepared_validation_complete_cycle_loader = prepare_validation_batches(
            validation_complete_cycle_loader,
            skeleton,
            seed=int(training.get("seed", 0)),
            corruption_config=corruption,
            throughput_config=throughput,
        )
    model = RotationAwareFusionModel(
        skeleton, hidden_channels=int(training.get("hidden_channels", 128))
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3))
    )
    return BenchmarkWorkload(
        model=model,
        optimizer=optimizer,
        skeleton=skeleton,
        train_loader=train_loader,
        validation_loader=validation_loader,
        complete_cycle_loader=complete_cycle_loader,
        validation_complete_cycle_loader=validation_complete_cycle_loader,
        loss_config=loss_config_for_ablation(ablation),
        corruption_config=corruption,
        throughput_config=throughput,
        training_config=training,
        prepared_validation_loader=prepared_validation_loader,
        prepared_validation_complete_cycle_loader=prepared_validation_complete_cycle_loader,
        train_window_count=len(train_set),
        train_complete_cycle_count=len(train_cycles),
        validation_window_count=len(validation_loader.dataset),
        validation_complete_cycle_count=len(validation_cycles),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _require_finite(value: object, description: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(item, f"{description}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f"{description}.{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite benchmark value: {description}")


def _run_epoch(
    workload: BenchmarkWorkload,
    *,
    epoch: int,
    device: torch.device,
    profiler_enabled: bool = False,
) -> tuple[dict[str, float], dict[str, Any], StageProfiler]:
    profiler = StageProfiler(
        enabled=profiler_enabled,
        device=device,
    )
    train_metrics = train_one_epoch(
        workload.model,
        workload.train_loader,
        workload.optimizer,
        workload.skeleton,
        loss_config=workload.loss_config,
        corruption_config=workload.corruption_config,
        complete_cycle_loader=workload.complete_cycle_loader,
        seed=int(workload.training_config.get("seed", 0)),
        epoch=epoch,
        device=device,
        profiler=profiler,
        throughput_config=workload.throughput_config,
    )
    validation_metrics = validate(
        workload.model,
        workload.validation_loader,
        workload.skeleton,
        loss_config=workload.loss_config,
        corruption_config=workload.corruption_config,
        complete_cycle_loader=workload.validation_complete_cycle_loader,
        seed=int(workload.training_config.get("seed", 0)),
        device=device,
        profiler=profiler,
        throughput_config=workload.throughput_config,
        prepared_loader=workload.prepared_validation_loader,
        prepared_complete_cycle_loader=workload.prepared_validation_complete_cycle_loader,
    )
    _require_finite(train_metrics, "train")
    _require_finite(validation_metrics, "validation")
    return train_metrics, validation_metrics, profiler


def _validation_result(
    workload: BenchmarkWorkload, *, device: torch.device, scalar_forward: bool
) -> dict[str, Any]:
    prepared_loader = None if scalar_forward else workload.prepared_validation_loader
    prepared_complete_cycle_loader = (
        None
        if scalar_forward
        else workload.prepared_validation_complete_cycle_loader
    )
    return validate(
        workload.model,
        workload.validation_loader,
        workload.skeleton,
        loss_config=workload.loss_config,
        corruption_config=workload.corruption_config,
        complete_cycle_loader=workload.validation_complete_cycle_loader,
        seed=int(workload.training_config.get("seed", 0)),
        device=device,
        throughput_config=workload.throughput_config,
        prepared_loader=prepared_loader,
        prepared_complete_cycle_loader=prepared_complete_cycle_loader,
        scalar_forward=scalar_forward,
    )


def _metric_comparison(
    reference: float, optimized: float, *, relative_tolerance: float, absolute_tolerance: float
) -> dict[str, float | bool]:
    absolute_delta = abs(optimized - reference)
    relative_delta = absolute_delta / max(abs(reference), 1e-12)
    return {
        "scalar_reference": reference,
        "optimized": optimized,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "equivalent": math.isclose(
            reference,
            optimized,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ),
    }


def _mapping_comparison(
    reference: Mapping[str, object],
    optimized: Mapping[str, object],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    description: str,
) -> dict[str, dict[str, float | bool]]:
    if set(reference) != set(optimized):
        raise ValueError(f"scalar and optimized {description} keys differ")
    return {
        name: _metric_comparison(
            float(value),
            float(optimized[name]),
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        for name, value in reference.items()
    }


def _checkpoint_decision(score: float, prior_best_score: float | None) -> dict[str, float | bool | None]:
    """Apply the CLI's ``score >= best`` selection rule for one path."""
    selected = prior_best_score is None or score >= prior_best_score
    return {
        "prior_best_score": prior_best_score,
        "score": score,
        "selected": selected,
        "next_best_score": score if selected else prior_best_score,
    }


def _validation_history_entry(
    reference: Mapping[str, object],
    optimized: Mapping[str, object],
    *,
    epoch: int,
    phase: str,
    scalar_prior_best_score: float | None,
    optimized_prior_best_score: float | None,
) -> dict[str, object]:
    """Compare one trained state and replay both checkpoint decisions."""
    reference_losses = reference.get("losses")
    optimized_losses = optimized.get("losses")
    reference_components = reference.get("components")
    optimized_components = optimized.get("components")
    if not all(
        isinstance(value, Mapping)
        for value in (
            reference_losses,
            optimized_losses,
            reference_components,
            optimized_components,
        )
    ):
        raise ValueError("validation paths require loss and component mappings")
    losses = _mapping_comparison(
        reference_losses,
        optimized_losses,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-6,
        description="loss",
    )
    components = _mapping_comparison(
        reference_components,
        optimized_components,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-7,
        description="component",
    )
    score = _metric_comparison(
        float(reference["score"]),
        float(optimized["score"]),
        relative_tolerance=1e-7,
        absolute_tolerance=1e-7,
    )
    scalar_decision = _checkpoint_decision(
        float(reference["score"]), scalar_prior_best_score
    )
    optimized_decision = _checkpoint_decision(
        float(optimized["score"]), optimized_prior_best_score
    )
    equivalent = all(
        bool(comparison["equivalent"])
        for comparisons in (losses, components)
        for comparison in comparisons.values()
    ) and bool(score["equivalent"])
    checkpoint_selection = {
        "rule": "score >= best_score",
        "scalar_reference": scalar_decision,
        "optimized": optimized_decision,
        "agreement": scalar_decision["selected"] == optimized_decision["selected"],
    }
    return {
        "epoch": epoch,
        "phase": phase,
        "scalar_reference": dict(reference),
        "optimized": dict(optimized),
        "losses": losses,
        "components": components,
        "score": score,
        "equivalent": equivalent,
        "checkpoint_selection": checkpoint_selection,
        "accepted": equivalent and bool(checkpoint_selection["agreement"]),
    }


def _require_validation_acceptance(acceptance: Mapping[str, object]) -> None:
    if not acceptance.get("accepted"):
        raise AssertionError("validation acceptance failed")


def _validation_history_entry_for_state(
    workload: BenchmarkWorkload,
    *,
    device: torch.device,
    epoch: int,
    phase: str,
    scalar_prior_best_score: float | None,
    optimized_prior_best_score: float | None,
) -> dict[str, object]:
    """Gate one trained state against scalar validation outside epoch timing."""
    reference = _validation_result(workload, device=device, scalar_forward=True)
    optimized = _validation_result(workload, device=device, scalar_forward=False)
    _require_finite(reference, "scalar_reference")
    _require_finite(optimized, "optimized")
    acceptance = _validation_history_entry(
        reference,
        optimized,
        epoch=epoch,
        phase=phase,
        scalar_prior_best_score=scalar_prior_best_score,
        optimized_prior_best_score=optimized_prior_best_score,
    )
    _require_finite(acceptance, "validation_acceptance")
    _require_validation_acceptance(acceptance)
    return acceptance


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run warmups and measured epochs, returning a JSON-serializable report."""
    _validate_arguments(args)
    config = load_config(args.config)
    workload = _build_workload(config, args.ablation)
    device = _resolve_device(str(args.device or workload.training_config.get("device", "cpu")))
    source_training = dict(workload.training_config)
    effective_training = {**source_training, "device": str(device)}
    validation_entries: list[dict[str, object]] = []
    scalar_best_score: float | None = None
    optimized_best_score: float | None = None
    scalar_selected_epoch: int | None = None
    optimized_selected_epoch: int | None = None

    def record_validation_history(epoch: int, phase: str) -> None:
        nonlocal scalar_best_score, optimized_best_score
        nonlocal scalar_selected_epoch, optimized_selected_epoch
        entry = _validation_history_entry_for_state(
            workload,
            device=device,
            epoch=epoch,
            phase=phase,
            scalar_prior_best_score=scalar_best_score,
            optimized_prior_best_score=optimized_best_score,
        )
        selection = entry["checkpoint_selection"]
        if not isinstance(selection, Mapping):
            raise ValueError("validation history requires checkpoint selection data")
        scalar_decision = selection.get("scalar_reference")
        optimized_decision = selection.get("optimized")
        if not isinstance(scalar_decision, Mapping) or not isinstance(optimized_decision, Mapping):
            raise ValueError("validation history requires both checkpoint decisions")
        scalar_best_score = float(scalar_decision["next_best_score"])
        optimized_best_score = float(optimized_decision["next_best_score"])
        if scalar_decision["selected"]:
            scalar_selected_epoch = epoch
        if optimized_decision["selected"]:
            optimized_selected_epoch = epoch
        validation_entries.append(entry)

    for epoch in range(args.warmup_epochs):
        _synchronize(device)
        _run_epoch(workload, epoch=epoch, device=device, profiler_enabled=False)
        _synchronize(device)
        record_validation_history(epoch, "warmup")

    epoch_timings: list[float] = []
    measured_peak_cuda_memory_bytes: list[int] = []
    train_metrics_by_epoch: list[dict[str, float]] = []
    validation_metrics_by_epoch: list[dict[str, Any]] = []
    for measurement_index in range(args.measured_epochs):
        epoch = args.warmup_epochs + measurement_index
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started = time.perf_counter()
        train_metrics, validation_metrics, profiler = _run_epoch(
            workload, epoch=epoch, device=device, profiler_enabled=False
        )
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise FloatingPointError("benchmark epoch time must be finite and positive")
        if profiler.summary() != {}:
            raise AssertionError("timed benchmark epochs must not collect stage timings")
        measured_peak_cuda_memory_bytes.append(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        epoch_timings.append(elapsed)
        train_metrics_by_epoch.append(train_metrics)
        validation_metrics_by_epoch.append(validation_metrics)
        record_validation_history(epoch, "measured")

    median_epoch_seconds = statistics.median(epoch_timings)
    effective_train_window_rate = [
        workload.train_window_count / value for value in epoch_timings
    ]
    peak_cuda_memory_bytes = max(measured_peak_cuda_memory_bytes, default=0)
    validation_history = {
        "rule": "score >= best_score",
        "initial_best_score": None,
        "epochs": validation_entries,
        "scalar_reference": {
            "best_score": scalar_best_score,
            "selected_epoch": scalar_selected_epoch,
        },
        "optimized": {
            "best_score": optimized_best_score,
            "selected_epoch": optimized_selected_epoch,
        },
        "final_selected_epoch_agreement": (
            scalar_selected_epoch == optimized_selected_epoch
        ),
        "accepted": all(bool(entry["accepted"]) for entry in validation_entries)
        and scalar_selected_epoch == optimized_selected_epoch,
    }
    _require_finite(validation_history, "validation_history")
    _require_validation_acceptance(validation_history)
    diagnostic_epoch = args.warmup_epochs + args.measured_epochs
    _synchronize(device)
    _, _, diagnostic_profiler = _run_epoch(
        workload,
        epoch=diagnostic_epoch,
        device=device,
        profiler_enabled=True,
    )
    _synchronize(device)
    diagnostic_stages = diagnostic_profiler.summary()
    _require_finite(diagnostic_stages, "diagnostic_stages")
    stage_timings: dict[str, object] = {
        "mode": "untimed_diagnostic_epoch",
        "configured_profile_stages": workload.throughput_config.profile_stages,
        "measured_epochs": [],
        "diagnostic": {
            "epoch": diagnostic_epoch,
            "timed": False,
            "stages": diagnostic_stages,
        },
    }
    device_report: dict[str, Any] = {
        "resolved": str(device),
        "type": device.type,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        device_report["name"] = torch.cuda.get_device_name(device)
        device_report["index"] = device.index
    report = {
        "config": {
            "path": str(Path(args.config).resolve()),
            "source_training": source_training,
            "effective_training": effective_training,
            "performance": asdict(workload.throughput_config),
            "window": dict(config.get("window", {})),
        },
        "device": device_report,
        "warmup_epochs": args.warmup_epochs,
        "measured_epochs": args.measured_epochs,
        "epoch_timings_seconds": epoch_timings,
        "median_epoch_seconds": median_epoch_seconds,
        "workload_counts": {
            "train_windows": workload.train_window_count,
            "train_complete_cycles": workload.train_complete_cycle_count,
            "validation_windows": workload.validation_window_count,
            "validation_complete_cycles": workload.validation_complete_cycle_count,
        },
        "effective_train_window_rate": {
            "definition": "train windows divided by end-to-end epoch wall time",
            "per_epoch_windows_per_second": effective_train_window_rate,
            "median_windows_per_second": workload.train_window_count
            / median_epoch_seconds,
        },
        "validation_history": validation_history,
        "stage_timings": stage_timings,
        "train_metrics": train_metrics_by_epoch,
        "validation_metrics": validation_metrics_by_epoch,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "peak_cuda_memory_gib": peak_cuda_memory_bytes / (1024**3),
        "measured_peak_cuda_memory_bytes": measured_peak_cuda_memory_bytes,
    }
    _require_finite(report, "report")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and write one benchmark report."""
    args = make_parser().parse_args(argv)
    _validate_arguments(args)
    report = run_benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[benchmark] "
        f"device={report['device']['resolved']} "
        f"median_epoch_seconds={report['median_epoch_seconds']:.6f} "
        f"effective_train_window_rate="
        f"{report['effective_train_window_rate']['median_windows_per_second']:.3f} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
