"""Deterministic CPU training, validation, and checkpointing for fusion."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from .config import SkeletonSpec
from .corruptions import CorruptionConfig, apply_corruptions, stable_window_seed
from .data import validate_cache_manifest_identities
from .features import FeatureBundle, compute_disagreement_features, compute_quality_features, extract_pose_features
from .losses import (
    LossConfig,
    compute_complete_cycle_rom_loss,
    compute_self_supervised_losses,
)
from .model import FusionOutput, RotationAwareFusionModel
from .prefetch import ThroughputConfig, ordered_prefetch, pin_tensor_batch
from .profiling import StageProfiler
from .trunk import extract_trunk_features


def _tensor_batch(
    batch: Mapping[str, object], device: torch.device, *, non_blocking: bool = False
) -> dict[str, object]:
    return {
        name: value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _required_tensor(batch: Mapping[str, object], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"training batch requires tensor field {name!r}")
    return value


def _feature_bundle(points: Tensor, valid: Tensor, skeleton: SkeletonSpec, dt: Tensor) -> FeatureBundle:
    trunk = extract_trunk_features(points, valid, skeleton, dt=dt)
    return FeatureBundle(
        pose=extract_pose_features(points, valid, skeleton, dt=dt),
        quality=compute_quality_features(points, valid, trunk, skeleton),
    )


def _corrupt_batch(
    batch: Mapping[str, object],
    *,
    seed: int,
    skeleton: SkeletonSpec,
    corruption_config: CorruptionConfig | None,
    epoch: int = 0,
) -> dict[str, object]:
    face = _required_tensor(batch, "face")
    side = _required_tensor(batch, "side")
    valid_face = _required_tensor(batch, "valid_face")
    valid_side = _required_tensor(batch, "valid_side")
    if face.device.type != "cpu":
        raise ValueError("deterministic synthetic corruption requires CPU batches")
    window_ids = batch.get("window_id")
    if not isinstance(window_ids, (list, tuple)) or len(window_ids) != face.shape[0] or not all(isinstance(value, str) for value in window_ids):
        raise ValueError("training batches require one stable string window_id per sample")
    epoch_seed = int(seed) + int(epoch) * 1_000_003
    results = [
        apply_corruptions(
            face[index], side[index], valid_face[index].bool(), valid_side[index].bool(),
            seed=stable_window_seed(epoch_seed, window_id), config=corruption_config, skeleton=skeleton,
        )
        for index, window_id in enumerate(window_ids)
    ]
    prepared = dict(batch)
    prepared.update(
        {
            "face": torch.stack([item.corrupted_face for item in results]),
            "side": torch.stack([item.corrupted_side for item in results]),
            "corrupted_valid_face": torch.stack([item.corrupted_valid_face for item in results]),
            "corrupted_valid_side": torch.stack([item.corrupted_valid_side for item in results]),
            "reference_face": torch.stack([item.reference_face for item in results]),
            "reference_side": torch.stack([item.reference_side for item in results]),
            "reference_valid_face": torch.stack([item.valid_face for item in results]),
            "reference_valid_side": torch.stack([item.valid_side for item in results]),
            "face_corruption_mask": torch.stack([item.face_corruption_mask for item in results]),
            "side_corruption_mask": torch.stack([item.side_corruption_mask for item in results]),
        }
    )
    return prepared


def _prepare_window(
    batch: Mapping[str, object],
    *,
    seed: int,
    skeleton: SkeletonSpec,
    corruption_config: CorruptionConfig | None,
    epoch: int = 0,
) -> dict[str, object]:
    """Prepare deterministic synthetic corruptions on the CPU."""
    return _corrupt_batch(
        batch, seed=seed, skeleton=skeleton, corruption_config=corruption_config, epoch=epoch
    )


def _forward_prepared(
    model: RotationAwareFusionModel,
    prepared: Mapping[str, object],
    skeleton: SkeletonSpec,
    *,
    device: torch.device,
    non_blocking: bool = False,
    profiler: StageProfiler | None = None,
) -> tuple[FusionOutput, dict[str, object]]:
    """Transfer one prepared CPU batch and run the model and feature pipeline."""
    if profiler is None:
        profiler = StageProfiler(enabled=False, device=device)
    with profiler.stage("transfer"):
        prepared = _tensor_batch(prepared, device, non_blocking=non_blocking)
    prepared = dict(prepared)
    face = _required_tensor(prepared, "face")
    side = _required_tensor(prepared, "side")
    valid_face = _required_tensor(prepared, "corrupted_valid_face")
    valid_side = _required_tensor(prepared, "corrupted_valid_side")
    temporal_valid = _required_tensor(prepared, "padding_mask")
    dt = _required_tensor(prepared, "dt")

    # This must precede all feature extraction because Task 5 validates provenance.
    effective_face_valid = valid_face.bool() & temporal_valid.bool()[..., None]
    effective_side_valid = valid_side.bool() & temporal_valid.bool()[..., None]
    safe_face = torch.where(effective_face_valid[..., None], face, torch.zeros_like(face))
    safe_side = torch.where(effective_side_valid[..., None], side, torch.zeros_like(side))
    with profiler.stage("features"):
        face_features = _feature_bundle(safe_face, effective_face_valid, skeleton, dt)
        side_features = _feature_bundle(safe_side, effective_side_valid, skeleton, dt)
        face_trunk = extract_trunk_features(safe_face, effective_face_valid, skeleton, dt=dt)
        side_trunk = extract_trunk_features(safe_side, effective_side_valid, skeleton, dt=dt)
        cross = compute_disagreement_features(
            safe_face, safe_side, face_trunk, side_trunk, effective_face_valid, effective_side_valid
        )
    with profiler.stage("forward"):
        output = model(
            safe_face, safe_side, face_features, side_features, cross, effective_face_valid, effective_side_valid,
            temporal_valid=temporal_valid, dt=dt,
        )
    reference_valid_face = _required_tensor(prepared, "reference_valid_face")
    reference_valid_side = _required_tensor(prepared, "reference_valid_side")
    loss_mask = _required_tensor(prepared, "loss_mask")
    reference_face = _required_tensor(prepared, "reference_face")
    reference_side = _required_tensor(prepared, "reference_side")
    effective_reference_face_valid = reference_valid_face.bool() & temporal_valid.bool()[..., None]
    effective_reference_side_valid = reference_valid_side.bool() & temporal_valid.bool()[..., None]
    safe_reference_face = torch.where(
        effective_reference_face_valid[..., None], reference_face, torch.zeros_like(reference_face)
    )
    safe_reference_side = torch.where(
        effective_reference_side_valid[..., None], reference_side, torch.zeros_like(reference_side)
    )
    with profiler.stage("reference_features"):
        reference_face_features = _feature_bundle(safe_reference_face, effective_reference_face_valid, skeleton, dt)
        reference_side_features = _feature_bundle(safe_reference_side, effective_reference_side_valid, skeleton, dt)
    prepared["valid_face"] = reference_valid_face.bool()
    prepared["valid_side"] = reference_valid_side.bool()
    prepared["loss_mask"] = loss_mask.bool() & temporal_valid.bool()[..., None]
    prepared["quality_face"] = reference_face_features.quality.loss_weight
    prepared["quality_side"] = reference_side_features.quality.loss_weight
    complete_cycle = prepared.get("complete_cycle")
    if complete_cycle is None:
        prepared["complete_cycle"] = torch.zeros(temporal_valid.shape[0], dtype=torch.bool, device=device)
    elif not isinstance(complete_cycle, Tensor):
        prepared["complete_cycle"] = torch.as_tensor(complete_cycle, dtype=torch.bool, device=device)
    return output, prepared


def _forward_window(
    model: RotationAwareFusionModel,
    batch: Mapping[str, object],
    skeleton: SkeletonSpec,
    *,
    seed: int,
    corruption_config: CorruptionConfig | None,
    device: torch.device,
    epoch: int = 0,
    profiler: StageProfiler | None = None,
) -> tuple[FusionOutput, dict[str, object]]:
    if profiler is None:
        profiler = StageProfiler(enabled=False, device=device)
    with profiler.stage("corruption"):
        prepared = _prepare_window(
            batch, seed=seed, skeleton=skeleton, corruption_config=corruption_config, epoch=epoch
        )
    return _forward_prepared(model, prepared, skeleton, device=device, profiler=profiler)


def _mean_metrics(losses: list[dict[str, float]]) -> dict[str, float]:
    if not losses:
        raise ValueError("loader produced no batches")
    keys = losses[0].keys()
    return {key: sum(values[key] for values in losses) / len(losses) for key in keys}


def _single_sample(batch: Mapping[str, object], index: int) -> dict[str, object]:
    """Preserve a collated sample's batch dimension and stable metadata."""
    sample: dict[str, object] = {}
    for name, value in batch.items():
        if isinstance(value, Tensor):
            sample[name] = value[index : index + 1]
        elif isinstance(value, (list, tuple)):
            sample[name] = [value[index]]
        else:
            sample[name] = value
    return sample


def _single_output(output: FusionOutput, index: int) -> FusionOutput:
    """Select one output while retaining the batch dimension for loss code."""
    return FusionOutput(
        fused_kpts=output.fused_kpts[index : index + 1],
        base_kpts=output.base_kpts[index : index + 1],
        delta_kpts=output.delta_kpts[index : index + 1],
        valid=output.valid[index : index + 1],
        fused_theta=output.fused_theta[index : index + 1],
        fused_theta_valid=output.fused_theta_valid[index : index + 1],
        fused_r_pt=output.fused_r_pt[index : index + 1],
        fused_r_pt_valid=output.fused_r_pt_valid[index : index + 1],
    )


def prepare_validation_batches(
    loader: Iterable[Mapping[str, object]],
    skeleton: SkeletonSpec,
    *,
    seed: int,
    corruption_config: CorruptionConfig | None,
    throughput_config: ThroughputConfig | None = None,
) -> list[dict[str, object]]:
    """Materialize fixed CPU validation corruptions without model-derived values."""
    throughput = throughput_config or ThroughputConfig()

    def prepare(batch: Mapping[str, object]) -> dict[str, object]:
        return _prepare_window(
            batch,
            seed=int(seed),
            skeleton=skeleton,
            corruption_config=corruption_config,
        )

    return list(
        ordered_prefetch(loader, prepare, depth=throughput.prefetch_batches)
    )


def train_one_epoch(
    model: RotationAwareFusionModel,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    skeleton: SkeletonSpec,
    *,
    loss_config: LossConfig | None = None,
    corruption_config: CorruptionConfig | None = None,
    complete_cycle_loader: Iterable[Mapping[str, object]] | None = None,
    seed: int = 0,
    epoch: int = 0,
    device: str | torch.device = "cpu",
    profiler: StageProfiler | None = None,
    throughput_config: ThroughputConfig | None = None,
) -> dict[str, float]:
    """Run one seeded epoch and return finite averages of the nine objectives."""
    model.train()
    target_device = torch.device(device)
    profiler = profiler if profiler is not None else StageProfiler(enabled=False, device=target_device)
    model.to(target_device)
    config = loss_config or LossConfig()
    throughput = throughput_config or ThroughputConfig()
    window_config = (
        replace(config, complete_cycle_rom_weight=0.0)
        if complete_cycle_loader is not None
        else config
    )
    torch.manual_seed(int(seed) + int(epoch))
    history: list[dict[str, float]] = []
    def prepare(batch: Mapping[str, object]) -> dict[str, object]:
        return _prepare_window(
            batch,
            seed=int(seed),
            skeleton=skeleton,
            corruption_config=corruption_config,
            epoch=epoch,
        )

    for batch_index, prepared in enumerate(
        ordered_prefetch(loader, prepare, depth=throughput.prefetch_batches)
    ):
        optimizer.zero_grad(set_to_none=True)
        if throughput.pin_memory:
            prepared = pin_tensor_batch(prepared)
        output, prepared = _forward_prepared(
            model,
            prepared,
            skeleton,
            device=target_device,
            non_blocking=throughput.non_blocking_transfer,
            profiler=profiler,
        )
        with profiler.stage("loss"):
            losses = compute_self_supervised_losses(output, prepared, window_config, skeleton)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("self-supervised loss is non-finite")
        if losses.total.requires_grad:
            with profiler.stage("backward"):
                losses.total.backward()
            with profiler.stage("optimizer"):
                optimizer.step()
        history.append({name: float(value.detach().cpu()) for name, value in losses.as_dict().items()})
    means = _mean_metrics(history)
    if complete_cycle_loader is not None and config.complete_cycle_rom_weight > 0:
        rom_values: list[float] = []
        for prepared in ordered_prefetch(
            complete_cycle_loader, prepare, depth=throughput.prefetch_batches
        ):
            optimizer.zero_grad(set_to_none=True)
            if throughput.pin_memory:
                prepared = pin_tensor_batch(prepared)
            output, prepared = _forward_prepared(
                model,
                prepared,
                skeleton,
                device=target_device,
                non_blocking=throughput.non_blocking_transfer,
                profiler=profiler,
            )
            with profiler.stage("complete_cycle_loss"):
                rom = compute_complete_cycle_rom_loss(output, prepared, config, skeleton)
            if not torch.isfinite(rom):
                raise FloatingPointError("complete-cycle ROM loss is non-finite")
            weighted_rom = config.complete_cycle_rom_weight * rom
            if weighted_rom.requires_grad:
                with profiler.stage("complete_cycle_backward"):
                    weighted_rom.backward()
                with profiler.stage("complete_cycle_optimizer"):
                    optimizer.step()
            rom_values.append(float(rom.detach().cpu()))
        if not rom_values:
            raise ValueError("complete-cycle ROM loader produced no batches")
        means["complete_cycle_rom"] = sum(rom_values) / len(rom_values)
        means["total"] += (
            config.complete_cycle_rom_weight * means["complete_cycle_rom"]
        )
    return {"loss": means["total"], **means}


def validate(
    model: RotationAwareFusionModel,
    loader: Iterable[Mapping[str, object]],
    skeleton: SkeletonSpec,
    *,
    loss_config: LossConfig | None = None,
    corruption_config: CorruptionConfig | None = None,
    complete_cycle_loader: Iterable[Mapping[str, object]] | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
    profiler: StageProfiler | None = None,
    throughput_config: ThroughputConfig | None = None,
    prepared_loader: Iterable[Mapping[str, object]] | None = None,
    prepared_complete_cycle_loader: Iterable[Mapping[str, object]] | None = None,
    scalar_forward: bool = False,
) -> dict[str, Any]:
    """Evaluate fixed corruptions and derive a score from five self-supervised measures."""
    was_training = model.training
    model.eval()
    target_device = torch.device(device)
    profiler = profiler if profiler is not None else StageProfiler(enabled=False, device=target_device)
    model.to(target_device)
    config = loss_config or LossConfig()
    throughput = throughput_config or ThroughputConfig()
    window_config = (
        replace(config, complete_cycle_rom_weight=0.0)
        if complete_cycle_loader is not None
        else config
    )
    samples: list[tuple[str, dict[str, float], FusionOutput, dict[str, object]]] = []

    def append_sample(output: FusionOutput, prepared: dict[str, object]) -> None:
        window_ids = prepared.get("window_id")
        if not isinstance(window_ids, list) or len(window_ids) != 1 or not isinstance(window_ids[0], str):
            raise ValueError("validation samples require one stable string window_id")
        with profiler.stage("loss"):
            losses = compute_self_supervised_losses(output, prepared, window_config, skeleton)
        samples.append(
            (
                window_ids[0],
                {name: float(value.cpu()) for name, value in losses.as_dict().items()},
                output,
                prepared,
            )
        )

    def forward_prepared_batch(batch: Mapping[str, object]) -> tuple[FusionOutput, dict[str, object]]:
        prepared = pin_tensor_batch(batch) if throughput.pin_memory else dict(batch)
        return _forward_prepared(
            model,
            prepared,
            skeleton,
            device=target_device,
            non_blocking=throughput.non_blocking_transfer,
            profiler=profiler,
        )

    with torch.no_grad():
        if scalar_forward and prepared_loader is None:
            for batch in loader:
                face = _required_tensor(batch, "face")
                for sample_index in range(face.shape[0]):
                    output, prepared = _forward_window(
                        model,
                        _single_sample(batch, sample_index),
                        skeleton,
                        seed=int(seed),
                        corruption_config=corruption_config,
                        device=target_device,
                        profiler=profiler,
                    )
                    append_sample(output, prepared)
        else:
            batches = prepared_loader
            if batches is None:
                batches = prepare_validation_batches(
                    loader,
                    skeleton,
                    seed=int(seed),
                    corruption_config=corruption_config,
                    throughput_config=throughput,
                )
            for batch in batches:
                if scalar_forward:
                    face = _required_tensor(batch, "face")
                    for sample_index in range(face.shape[0]):
                        output, prepared = forward_prepared_batch(
                            _single_sample(batch, sample_index)
                        )
                        append_sample(output, prepared)
                    continue
                output, prepared = forward_prepared_batch(batch)
                for sample_index in range(output.fused_kpts.shape[0]):
                    append_sample(
                        _single_output(output, sample_index),
                        _single_sample(prepared, sample_index),
                    )
    samples.sort(key=lambda value: value[0])
    means = _mean_metrics([metrics for _, metrics, _, _ in samples])
    if complete_cycle_loader is not None:
        rom_values: list[float] = []
        with torch.no_grad():
            cycle_batches = prepared_complete_cycle_loader
            if cycle_batches is None:
                cycle_batches = prepare_validation_batches(
                    complete_cycle_loader,
                    skeleton,
                    seed=int(seed),
                    corruption_config=corruption_config,
                    throughput_config=throughput,
                )
            for batch in cycle_batches:
                output, prepared = forward_prepared_batch(batch)
                with profiler.stage("complete_cycle_loss"):
                    rom = compute_complete_cycle_rom_loss(output, prepared, config, skeleton)
                rom_values.append(float(rom.cpu()))
        if not rom_values:
            raise ValueError("complete-cycle ROM loader produced no batches")
        means["complete_cycle_rom"] = sum(rom_values) / len(rom_values)
        means["total"] += (
            config.complete_cycle_rom_weight * means["complete_cycle_rom"]
        )
    if was_training:
        model.train()
    bone_cv = _trial_level_bone_cv(samples, skeleton)
    components = {
        "corruption_recovery": 1.0 / (1.0 + means["corruption_recovery"]),
        "bone_cv": 1.0 / (1.0 + bone_cv),
        "rotation_consistency": 1.0 / (1.0 + (means["circular_axial_rotation"] + means["so3_rotation"]) / 2.0),
        "identity_preservation": 1.0 / (1.0 + means["high_consensus_identity"]),
        "rom_retention": 1.0 / (1.0 + means["complete_cycle_rom"]),
    }
    return {"loss": means["total"], "score": sum(components.values()) / len(components), "components": components, "losses": means}


def _fused_bone_cv(output: FusionOutput, dt: Tensor, skeleton: SkeletonSpec) -> float:
    """Return mean per-trial, per-bone coefficient of variation of fused lengths."""
    pose = extract_pose_features(output.fused_kpts, output.valid, skeleton, dt=dt)
    values: list[Tensor] = []
    for batch_index in range(pose.bone_lengths.shape[0]):
        for bone_index in range(pose.bone_lengths.shape[2]):
            lengths = pose.bone_lengths[batch_index, :, bone_index][pose.bone_valid[batch_index, :, bone_index]]
            if lengths.numel():
                values.append(lengths.std(unbiased=False) / lengths.mean().abs().clamp_min(1e-6))
    return float(torch.stack(values).mean().cpu()) if values else 0.0


def _trial_level_bone_cv(
    samples: list[tuple[str, dict[str, float], FusionOutput, dict[str, object]]],
    skeleton: SkeletonSpec,
) -> float:
    """Average overlapped fused frames before computing per-trial bone variation."""
    frames: dict[tuple[str, str, int], tuple[Tensor, Tensor]] = {}
    for _, _, output, prepared in samples:
        person_ids = prepared.get("person_id")
        trial_ids = prepared.get("trial_id")
        if not isinstance(person_ids, list) or not isinstance(trial_ids, list) or len(person_ids) != 1 or len(trial_ids) != 1:
            raise ValueError("validation samples require one person_id and trial_id")
        if not isinstance(person_ids[0], str) or not isinstance(trial_ids[0], str):
            raise ValueError("person_id and trial_id must be strings")
        global_index = _required_tensor(prepared, "global_frame_index")[0]
        for local_index, global_frame in enumerate(global_index.tolist()):
            if global_frame < 0:
                continue
            valid = output.valid[0, local_index]
            points = torch.where(valid[:, None], output.fused_kpts[0, local_index], torch.zeros_like(output.fused_kpts[0, local_index]))
            key = (person_ids[0], trial_ids[0], int(global_frame))
            previous_points, previous_count = frames.get(
                key, (torch.zeros_like(points), torch.zeros_like(valid, dtype=points.dtype))
            )
            frames[key] = (previous_points + points, previous_count + valid.to(dtype=points.dtype))
    grouped: dict[tuple[str, str], list[tuple[int, Tensor, Tensor]]] = {}
    for (person_id, trial_id, frame), (point_sum, count) in frames.items():
        grouped.setdefault((person_id, trial_id), []).append((frame, point_sum / count.clamp_min(1)[:, None], count > 0))
    values: list[Tensor] = []
    for sequence in grouped.values():
        sequence.sort(key=lambda item: item[0])
        for start, end in skeleton.bones:
            lengths = [torch.linalg.vector_norm(points[end] - points[start]) for _, points, valid in sequence if valid[start] and valid[end]]
            if lengths:
                value = torch.stack(lengths)
                values.append(value.std(unbiased=False) / value.mean().abs().clamp_min(1e-6))
    return float(torch.stack(values).mean().cpu()) if values else 0.0


def _skeleton_metadata(skeleton: SkeletonSpec) -> dict[str, object]:
    return {
        "name": skeleton.name,
        "joint_names": list(skeleton.joint_names),
        "bones": [list(bone) for bone in skeleton.bones],
        "roles": {
            name: {"kind": role.kind, "joints": list(role.joints), "fallback": list(role.fallback)}
            for name, role in skeleton.roles.items()
        },
        "required_roles": list(skeleton.required_roles),
        "joint_groups": {
            name: list(joints) for name, joints in skeleton.joint_groups.items()
        },
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    loss_config: LossConfig,
    skeleton: SkeletonSpec,
    provenance: Mapping[str, object],
    training_config: Mapping[str, object],
    corruption_config: CorruptionConfig,
    score: float,
    scheduler: object | None = None,
) -> None:
    """Persist enough provenance to reproduce a self-supervised selection decision."""
    required = ("split_hash", "corruption_manifest_hash", "git_commit")
    for name in required:
        value = provenance.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"provenance requires non-empty {name}")
    normalized_provenance = dict(provenance)
    normalized_provenance["cache_manifests"] = validate_cache_manifest_identities(
        provenance.get("cache_manifests")
    )
    if not training_config:
        raise ValueError("training_config must be non-empty")
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_config": asdict(loss_config),
        "training_config": dict(training_config),
        "corruption_config": asdict(corruption_config),
        "skeleton": _skeleton_metadata(skeleton),
        "provenance": normalized_provenance,
        "score": float(score),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: object | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load model state and return checkpoint metadata for callers to inspect."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint provenance must be a mapping")
    for name in ("split_hash", "corruption_manifest_hash", "git_commit"):
        value = provenance.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"provenance requires non-empty {name}")
    normalized_provenance = dict(provenance)
    normalized_provenance["cache_manifests"] = validate_cache_manifest_identities(
        provenance.get("cache_manifests")
    )
    payload["provenance"] = normalized_provenance
    if not isinstance(payload.get("training_config"), Mapping) or not payload["training_config"]:
        raise ValueError("checkpoint requires non-empty training_config")
    if not isinstance(payload.get("corruption_config"), Mapping):
        raise ValueError("checkpoint requires corruption_config")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload
