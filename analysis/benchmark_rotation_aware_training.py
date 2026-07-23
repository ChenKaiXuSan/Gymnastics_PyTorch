"""Measure rotation-aware training throughput on an already prepared cache.

The benchmark follows the training command's dataset, loss, validation, and
performance configuration paths without writing a run directory or checkpoint.
"""

from __future__ import annotations

import argparse
import gc
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
    _build_training_loaders,
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
    TrainingTrace,
    prepare_validation_batches,
    train_one_epoch,
    train_one_epoch_reference,
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
    diagnostic_validation_loader: DataLoader
    diagnostic_validation_complete_cycle_loader: DataLoader
    uses_training_validation: bool
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
    config: Mapping[str, Any],
    ablation: str,
    *,
    throughput_override: ThroughputConfig | None = None,
) -> BenchmarkWorkload:
    training = _training_config_for_ablation(config, ablation)
    performance = config.get("performance", {})
    if not isinstance(performance, Mapping):
        raise ValueError("rotation-aware performance config must be a mapping")
    throughput = throughput_override or ThroughputConfig(**dict(performance))
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
    (
        train_loader,
        validation_loader,
        complete_cycle_loader,
        validation_complete_cycle_loader,
        uses_training_validation,
    ) = _build_training_loaders(
        train_set,
        val_set,
        train_cycles,
        validation_cycles,
        batch_size=batch_size,
        generator=generator,
    )
    diagnostic_validation_loader = DataLoader(
        val_set if val_set is not None else train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pose_pair_windows,
    )
    diagnostic_validation_complete_cycle_loader = DataLoader(
        validation_cycles,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_pose_pair_windows,
    )
    corruption = CorruptionConfig()
    prepared_validation_loader = None
    prepared_validation_complete_cycle_loader = None
    if throughput.cache_validation_batches and not uses_training_validation:
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(training.get("seed", 0)))
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
        diagnostic_validation_loader=diagnostic_validation_loader,
        diagnostic_validation_complete_cycle_loader=diagnostic_validation_complete_cycle_loader,
        uses_training_validation=uses_training_validation,
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
        scalar_forward=True,
    )
    _require_finite(train_metrics, "train")
    _require_finite(validation_metrics, "validation")
    return train_metrics, validation_metrics, profiler


def _validation_input_paths(workload: BenchmarkWorkload) -> dict[str, object]:
    cache_available = workload.prepared_validation_loader is not None
    return {
        "scalar_reference": "uncached",
        "optimized": (
            "cached_validation_inputs"
            if cache_available
            else "uncached_no_validation_fallback"
        ),
        "cache_equivalence_applicable": cache_available,
    }


def _validation_result(
    workload: BenchmarkWorkload, *, device: torch.device, use_prepared_inputs: bool
) -> dict[str, Any]:
    # Benchmark-only validation history must not advance the production shared generator.
    return validate(
        workload.model,
        workload.diagnostic_validation_loader,
        workload.skeleton,
        loss_config=workload.loss_config,
        corruption_config=workload.corruption_config,
        complete_cycle_loader=workload.diagnostic_validation_complete_cycle_loader,
        seed=int(workload.training_config.get("seed", 0)),
        device=device,
        throughput_config=workload.throughput_config,
        prepared_loader=(
            workload.prepared_validation_loader if use_prepared_inputs else None
        ),
        prepared_complete_cycle_loader=(
            workload.prepared_validation_complete_cycle_loader
            if use_prepared_inputs
            else None
        ),
        scalar_forward=True,
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


def _clone_state(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value


def _flatten_state(value: object, prefix: str, output: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _flatten_state(value[key], f"{prefix}.{key}" if prefix else str(key), output)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_state(item, f"{prefix}.{index}" if prefix else str(index), output)
        return
    output[prefix] = value


def _state_comparison(
    reference: Mapping[str, object],
    optimized: Mapping[str, object],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    reference_leaves: dict[str, object] = {}
    optimized_leaves: dict[str, object] = {}
    _flatten_state(reference, "", reference_leaves)
    _flatten_state(optimized, "", optimized_leaves)
    keys_match = reference_leaves.keys() == optimized_leaves.keys()
    entries: dict[str, object] = {}
    equivalent = keys_match
    for name in sorted(reference_leaves.keys() & optimized_leaves.keys()):
        reference_value = reference_leaves[name]
        optimized_value = optimized_leaves[name]
        if isinstance(reference_value, torch.Tensor) and isinstance(
            optimized_value, torch.Tensor
        ):
            shape_match = reference_value.shape == optimized_value.shape
            dtype_match = reference_value.dtype == optimized_value.dtype
            floating = reference_value.is_floating_point() or reference_value.is_complex()
            finite = (
                bool(torch.isfinite(reference_value).all())
                and bool(torch.isfinite(optimized_value).all())
                if floating and dtype_match
                else True
            )
            if shape_match and dtype_match and floating and reference_value.numel():
                delta = (optimized_value - reference_value).abs()
                max_absolute_delta = float(delta.max())
                max_relative_delta = float(
                    (delta / reference_value.abs().clamp_min(1e-12)).max()
                )
            else:
                max_absolute_delta = None
                max_relative_delta = None
            tensor_equivalent = (
                shape_match
                and dtype_match
                and finite
                and (
                    torch.allclose(
                        reference_value,
                        optimized_value,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                    )
                    if floating
                    else torch.equal(reference_value, optimized_value)
                )
            )
            entries[name] = {
                "shape": list(reference_value.shape),
                "dtype": str(reference_value.dtype),
                "max_absolute_delta": max_absolute_delta,
                "max_relative_delta": max_relative_delta,
                "finite": finite,
                "equivalent": tensor_equivalent,
            }
            equivalent = equivalent and tensor_equivalent
        else:
            leaf_equivalent = (
                type(reference_value) is type(optimized_value)
                and reference_value == optimized_value
            )
            entries[name] = {
                "reference": reference_value,
                "optimized": optimized_value,
                "equivalent": leaf_equivalent,
            }
            equivalent = equivalent and leaf_equivalent
    return {
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "keys_match": keys_match,
        "entries": entries,
        "equivalent": bool(equivalent),
    }


def _trace_projection(trace: Mapping[str, object], field: str) -> list[object]:
    samples = trace.get("samples")
    if not isinstance(samples, list):
        raise ValueError("training equivalence trace requires a sample list")
    projection: list[object] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("training equivalence trace samples must be mappings")
        if field == "order":
            projection.append((sample.get("phase"), sample.get("window_id")))
        else:
            projection.append(sample.get(field))
    return projection


def _compare_training_probe_results(
    reference: Mapping[str, object],
    optimized: Mapping[str, object],
    *,
    expected_optimizer_steps: Mapping[str, int],
    expected_phase_samples: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Compare one synchronous and optimized epoch without hiding divergence."""
    reference_training_trace = reference.get("training_trace")
    optimized_training_trace = optimized.get("training_trace")
    reference_validation_trace = reference.get("validation_trace")
    optimized_validation_trace = optimized.get("validation_trace")
    if not all(
        isinstance(value, Mapping)
        for value in (
            reference_training_trace,
            optimized_training_trace,
            reference_validation_trace,
            optimized_validation_trace,
        )
    ):
        raise ValueError("training equivalence requires training and validation traces")

    training_order = _trace_projection(reference_training_trace, "order") == _trace_projection(
        optimized_training_trace, "order"
    )
    validation_membership = _trace_projection(
        reference_validation_trace, "order"
    ) == _trace_projection(optimized_validation_trace, "order")
    corruption_digests = (
        _trace_projection(reference_training_trace, "corruption_digests")
        == _trace_projection(optimized_training_trace, "corruption_digests")
        and _trace_projection(reference_validation_trace, "corruption_digests")
        == _trace_projection(optimized_validation_trace, "corruption_digests")
    )
    expected_steps = dict(expected_optimizer_steps)
    reference_steps = reference_training_trace.get("optimizer_steps")
    optimized_steps = optimized_training_trace.get("optimizer_steps")
    reference_steps_before = _trace_projection(
        reference_training_trace, "optimizer_steps_before"
    )
    optimized_steps_before = _trace_projection(
        optimized_training_trace, "optimizer_steps_before"
    )
    total_expected_steps = sum(expected_steps.values())
    step_trace_valid = all(
        isinstance(value, int) for value in reference_steps_before
    ) and (
        reference_steps_before == sorted(reference_steps_before)
        and sorted(set(reference_steps_before)) == list(range(total_expected_steps))
    )
    optimizer_steps = (
        reference_steps == expected_steps
        and optimized_steps == expected_steps
        and reference_training_trace.get("total_optimizer_steps")
        == total_expected_steps
        and optimized_training_trace.get("total_optimizer_steps")
        == total_expected_steps
        and reference_steps_before == optimized_steps_before
        and step_trace_valid
    )
    phase_sample_counts = {
        phase: sum(
            sample_phase == phase
            for sample_phase, _ in _trace_projection(reference_training_trace, "order")
        )
        for phase in (expected_phase_samples or {})
    }
    expected_samples = dict(expected_phase_samples or phase_sample_counts)
    optimized_phase_sample_counts = {
        phase: sum(
            sample_phase == phase
            for sample_phase, _ in _trace_projection(optimized_training_trace, "order")
        )
        for phase in expected_samples
    }
    phase_samples_match = (
        phase_sample_counts == expected_samples
        and optimized_phase_sample_counts == expected_samples
    )

    reference_train_metrics = reference.get("train_metrics")
    optimized_train_metrics = optimized.get("train_metrics")
    reference_validation = reference.get("validation_metrics")
    optimized_validation = optimized.get("validation_metrics")
    if not all(
        isinstance(value, Mapping)
        for value in (
            reference_train_metrics,
            optimized_train_metrics,
            reference_validation,
            optimized_validation,
        )
    ):
        raise ValueError("training equivalence requires metric mappings")
    train_metrics = _mapping_comparison(
        reference_train_metrics,
        optimized_train_metrics,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-6,
        description="training metric",
    )
    validation = _validation_history_entry(
        reference_validation,
        optimized_validation,
        epoch=0,
        phase="training_equivalence_probe",
        scalar_prior_best_score=None,
        optimized_prior_best_score=None,
    )
    model_state = _state_comparison(
        reference["model_state"],
        optimized["model_state"],
        relative_tolerance=1e-6,
        absolute_tolerance=1e-7,
    )
    adam_state = _state_comparison(
        reference["optimizer_state"],
        optimized["optimizer_state"],
        relative_tolerance=1e-6,
        absolute_tolerance=1e-7,
    )
    finite_metrics = True
    try:
        _require_finite(reference_train_metrics, "reference_train")
        _require_finite(optimized_train_metrics, "optimized_train")
        _require_finite(reference_validation, "reference_validation")
        _require_finite(optimized_validation, "optimized_validation")
    except FloatingPointError:
        finite_metrics = False
    gates = {
        "training_sample_order": training_order,
        "corruption_digests": corruption_digests,
        "optimizer_steps": optimizer_steps,
        "phase_sample_counts": phase_samples_match,
        "finite_metrics": finite_metrics,
        "train_metrics": all(bool(value["equivalent"]) for value in train_metrics.values()),
        "model_state": bool(model_state["equivalent"]),
        "adam_state": bool(adam_state["equivalent"]),
        "validation_membership": validation_membership,
        "validation_metrics": bool(validation["equivalent"]),
        "checkpoint_decision": bool(validation["checkpoint_selection"]["agreement"]),
    }
    return {
        "gates": gates,
        "train_metrics": train_metrics,
        "validation": validation,
        "state": {"model": model_state, "adam": adam_state},
        "optimizer_steps": {
            "expected": expected_steps,
            "reference": {
                "by_phase": reference_training_trace.get("optimizer_steps"),
                "total_optimizer_steps": reference_training_trace.get(
                    "total_optimizer_steps"
                ),
            },
            "optimized": {
                "by_phase": optimized_training_trace.get("optimizer_steps"),
                "total_optimizer_steps": optimized_training_trace.get(
                    "total_optimizer_steps"
                ),
            },
        },
        "phase_sample_counts": {
            "expected": expected_samples,
            "reference": phase_sample_counts,
            "optimized": optimized_phase_sample_counts,
        },
        "traces": {
            "reference": {
                "training": dict(reference_training_trace),
                "validation": dict(reference_validation_trace),
            },
            "optimized": {
                "training": dict(optimized_training_trace),
                "validation": dict(optimized_validation_trace),
            },
        },
        "training_order": {
            "reference": _trace_projection(reference_training_trace, "order"),
            "optimized": _trace_projection(optimized_training_trace, "order"),
        },
        "validation_membership": {
            "reference": _trace_projection(reference_validation_trace, "order"),
            "optimized": _trace_projection(optimized_validation_trace, "order"),
        },
        "accepted": all(gates.values()),
    }


def _require_training_equivalence(comparison: Mapping[str, object]) -> None:
    if not comparison.get("accepted"):
        gates = comparison.get("gates", {})
        failed = [name for name, accepted in gates.items() if not accepted]
        raise AssertionError(f"training equivalence failed: {', '.join(failed)}")


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
    reference = _validation_result(
        workload, device=device, use_prepared_inputs=False
    )
    optimized = _validation_result(
        workload, device=device, use_prepared_inputs=True
    )
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
    acceptance["input_paths"] = _validation_input_paths(workload)
    return acceptance


def _probe_result(
    workload: BenchmarkWorkload,
    *,
    device: torch.device,
    synchronous_reference: bool,
) -> dict[str, object]:
    training_trace = TrainingTrace()
    validation_trace = TrainingTrace()
    train_function = train_one_epoch_reference if synchronous_reference else train_one_epoch
    train_kwargs: dict[str, object] = {
        "loss_config": workload.loss_config,
        "corruption_config": workload.corruption_config,
        "complete_cycle_loader": workload.complete_cycle_loader,
        "seed": int(workload.training_config.get("seed", 0)),
        "epoch": 0,
        "device": device,
        "trace": training_trace,
    }
    if not synchronous_reference:
        train_kwargs["throughput_config"] = workload.throughput_config
    train_metrics = train_function(
        workload.model,
        workload.train_loader,
        workload.optimizer,
        workload.skeleton,
        **train_kwargs,
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
        throughput_config=(ThroughputConfig() if synchronous_reference else workload.throughput_config),
        prepared_loader=(None if synchronous_reference else workload.prepared_validation_loader),
        prepared_complete_cycle_loader=(
            None
            if synchronous_reference
            else workload.prepared_validation_complete_cycle_loader
        ),
        scalar_forward=True,
        trace=validation_trace,
    )
    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_trace": training_trace.as_dict(),
        "validation_trace": validation_trace.as_dict(),
        "model_state": _clone_state(workload.model.state_dict()),
        "optimizer_state": _clone_state(workload.optimizer.state_dict()),
    }


def _run_training_equivalence(
    config: Mapping[str, Any],
    ablation: str,
    device: torch.device,
    *,
    config_path: str,
) -> dict[str, object]:
    """Run an untimed seeded reference/optimized epoch on independent state."""
    reference_workload = _build_workload(
        config,
        ablation,
        throughput_override=ThroughputConfig(),
    )
    optimized_workload = _build_workload(config, ablation)
    initial_model = _state_comparison(
        reference_workload.model.state_dict(),
        optimized_workload.model.state_dict(),
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    initial_adam = _state_comparison(
        reference_workload.optimizer.state_dict(),
        optimized_workload.optimizer.state_dict(),
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    expected_optimizer_steps = {"train_window": len(reference_workload.train_loader)}
    if reference_workload.loss_config.complete_cycle_rom_weight > 0:
        expected_optimizer_steps["train_complete_cycle"] = len(
            reference_workload.complete_cycle_loader
        )
    reference = _probe_result(
        reference_workload,
        device=device,
        synchronous_reference=True,
    )
    optimized = _probe_result(
        optimized_workload,
        device=device,
        synchronous_reference=False,
    )
    comparison = _compare_training_probe_results(
        reference,
        optimized,
        expected_optimizer_steps=expected_optimizer_steps,
        expected_phase_samples={
            "train_window": reference_workload.train_window_count,
            **(
                {
                    "train_complete_cycle": reference_workload.train_complete_cycle_count
                }
                if reference_workload.loss_config.complete_cycle_rom_weight > 0
                else {}
            ),
        },
    )
    comparison["gates"]["initial_model_state"] = bool(initial_model["equivalent"])
    comparison["gates"]["initial_adam_state"] = bool(initial_adam["equivalent"])
    comparison["accepted"] = all(comparison["gates"].values())
    training = reference_workload.training_config
    comparison.update(
        {
            "protocol": {
                "ablation": ablation,
                "batch_size": int(training.get("batch_size", 4)),
                "epochs": int(training.get("epochs", 1)),
                "probe_epochs": 1,
                "seed": int(training.get("seed", 0)),
                "learning_rate": float(training.get("learning_rate", 1e-3)),
                "hidden_channels": int(training.get("hidden_channels", 128)),
                "precision": "FP32",
                "optimizer": "Adam",
                "device": str(device),
                "validation_forward": "scalar",
            },
            "provenance": {
                "config_path": str(Path(config_path).resolve()),
                "resolved_training": dict(training),
                "resolved_performance": asdict(optimized_workload.throughput_config),
                "loss_config": asdict(reference_workload.loss_config),
                "corruption_config": asdict(reference_workload.corruption_config),
                "independently_seeded_workloads": True,
                "workload_counts": {
                    "train_windows": reference_workload.train_window_count,
                    "train_complete_cycles": reference_workload.train_complete_cycle_count,
                    "validation_windows": reference_workload.validation_window_count,
                    "validation_complete_cycles": reference_workload.validation_complete_cycle_count,
                },
            },
            "reference_path": {
                "ordered_prefetch": False,
                "pin_memory": False,
                "non_blocking_transfer": False,
                "validation_cache": False,
                "batched_validation": False,
            },
            "initial_state": {"model": initial_model, "adam": initial_adam},
            "exact_gates": {
                name: comparison["gates"][name]
                for name in (
                    "training_sample_order",
                    "corruption_digests",
                    "optimizer_steps",
                    "phase_sample_counts",
                    "validation_membership",
                    "checkpoint_decision",
                    "initial_model_state",
                    "initial_adam_state",
                )
            },
        }
    )
    _require_finite(comparison, "training_equivalence")
    _require_training_equivalence(comparison)
    return comparison


def _release_probe_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run warmups and measured epochs, returning a JSON-serializable report."""
    _validate_arguments(args)
    config = load_config(args.config)
    resolved_training = _training_config_for_ablation(config, args.ablation)
    device = _resolve_device(
        str(args.device or resolved_training.get("device", "cpu"))
    )
    try:
        training_equivalence = _run_training_equivalence(
            config,
            args.ablation,
            device,
            config_path=args.config,
        )
    finally:
        _release_probe_memory(device)
    workload = _build_workload(config, args.ablation)
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
        "validation_forward": "scalar",
        "input_paths": _validation_input_paths(workload),
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
            "validation_forward": "scalar",
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
        "training_equivalence": training_equivalence,
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
