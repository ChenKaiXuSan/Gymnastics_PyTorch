"""Deterministic CPU training, validation, and checkpointing for fusion."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from .config import SkeletonSpec
from .corruptions import CorruptionConfig, apply_corruptions
from .features import FeatureBundle, compute_disagreement_features, compute_quality_features, extract_pose_features
from .losses import LossConfig, compute_self_supervised_losses
from .model import FusionOutput, RotationAwareFusionModel
from .trunk import extract_trunk_features


def _tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {name: value.to(device) if isinstance(value, Tensor) else value for name, value in batch.items()}


def _required_tensor(batch: Mapping[str, object], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"training batch requires tensor field {name!r}")
    return value


def _feature_bundle(points: Tensor, valid: Tensor, skeleton: SkeletonSpec) -> FeatureBundle:
    trunk = extract_trunk_features(points, valid, skeleton, dt=1.0)
    return FeatureBundle(
        pose=extract_pose_features(points, valid, skeleton, dt=1.0),
        quality=compute_quality_features(points, valid, trunk, skeleton),
    )


def _corrupt_batch(
    batch: Mapping[str, object],
    *,
    seed: int,
    skeleton: SkeletonSpec,
    corruption_config: CorruptionConfig | None,
) -> dict[str, object]:
    face = _required_tensor(batch, "face")
    side = _required_tensor(batch, "side")
    valid_face = _required_tensor(batch, "valid_face")
    valid_side = _required_tensor(batch, "valid_side")
    if face.device.type != "cpu":
        raise ValueError("deterministic synthetic corruption requires CPU batches")
    results = [
        apply_corruptions(
            face[index], side[index], valid_face[index].bool(), valid_side[index].bool(),
            seed=seed + index, config=corruption_config, skeleton=skeleton,
        )
        for index in range(face.shape[0])
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


def _forward_window(
    model: RotationAwareFusionModel,
    batch: Mapping[str, object],
    skeleton: SkeletonSpec,
    *,
    seed: int,
    corruption_config: CorruptionConfig | None,
    device: torch.device,
) -> tuple[FusionOutput, dict[str, object]]:
    prepared = _corrupt_batch(batch, seed=seed, skeleton=skeleton, corruption_config=corruption_config)
    prepared = _tensor_batch(prepared, device)
    face = _required_tensor(prepared, "face")
    side = _required_tensor(prepared, "side")
    valid_face = _required_tensor(prepared, "corrupted_valid_face")
    valid_side = _required_tensor(prepared, "corrupted_valid_side")
    temporal_valid = _required_tensor(prepared, "padding_mask")

    # This must precede all feature extraction because Task 5 validates provenance.
    effective_face_valid = valid_face.bool() & temporal_valid.bool()[..., None]
    effective_side_valid = valid_side.bool() & temporal_valid.bool()[..., None]
    safe_face = torch.where(effective_face_valid[..., None], face, torch.zeros_like(face))
    safe_side = torch.where(effective_side_valid[..., None], side, torch.zeros_like(side))
    face_features = _feature_bundle(safe_face, effective_face_valid, skeleton)
    side_features = _feature_bundle(safe_side, effective_side_valid, skeleton)
    face_trunk = extract_trunk_features(safe_face, effective_face_valid, skeleton, dt=1.0)
    side_trunk = extract_trunk_features(safe_side, effective_side_valid, skeleton, dt=1.0)
    cross = compute_disagreement_features(
        safe_face, safe_side, face_trunk, side_trunk, effective_face_valid, effective_side_valid
    )
    output = model(
        safe_face, safe_side, face_features, side_features, cross, effective_face_valid, effective_side_valid,
        temporal_valid=temporal_valid,
    )
    reference_valid_face = _required_tensor(prepared, "reference_valid_face")
    reference_valid_side = _required_tensor(prepared, "reference_valid_side")
    loss_mask = _required_tensor(prepared, "loss_mask")
    prepared["valid_face"] = reference_valid_face.bool()
    prepared["valid_side"] = reference_valid_side.bool()
    prepared["loss_mask"] = loss_mask.bool() & temporal_valid.bool()[..., None]
    prepared["quality_face"] = face_features.quality.loss_weight
    prepared["quality_side"] = side_features.quality.loss_weight
    complete_cycle = prepared.get("complete_cycle")
    if complete_cycle is None:
        prepared["complete_cycle"] = temporal_valid.bool().all(dim=1)
    elif not isinstance(complete_cycle, Tensor):
        prepared["complete_cycle"] = torch.as_tensor(complete_cycle, dtype=torch.bool, device=device)
    return output, prepared


def _mean_metrics(losses: list[dict[str, float]]) -> dict[str, float]:
    if not losses:
        raise ValueError("loader produced no batches")
    keys = losses[0].keys()
    return {key: sum(values[key] for values in losses) / len(losses) for key in keys}


def train_one_epoch(
    model: RotationAwareFusionModel,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    skeleton: SkeletonSpec,
    *,
    loss_config: LossConfig | None = None,
    corruption_config: CorruptionConfig | None = None,
    seed: int = 0,
    epoch: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Run one seeded epoch and return finite averages of the nine objectives."""
    model.train()
    target_device = torch.device(device)
    model.to(target_device)
    config = loss_config or LossConfig()
    torch.manual_seed(int(seed) + int(epoch))
    history: list[dict[str, float]] = []
    for batch_index, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        output, prepared = _forward_window(
            model, batch, skeleton, seed=int(seed) + int(epoch) * 1_000_003 + batch_index * 997,
            corruption_config=corruption_config, device=target_device,
        )
        losses = compute_self_supervised_losses(output, prepared, config, skeleton)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("self-supervised loss is non-finite")
        losses.total.backward()
        optimizer.step()
        history.append({name: float(value.detach().cpu()) for name, value in losses.as_dict().items()})
    means = _mean_metrics(history)
    return {"loss": means["total"], **means}


def validate(
    model: RotationAwareFusionModel,
    loader: Iterable[Mapping[str, object]],
    skeleton: SkeletonSpec,
    *,
    loss_config: LossConfig | None = None,
    corruption_config: CorruptionConfig | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate fixed corruptions and derive a score from five self-supervised measures."""
    was_training = model.training
    model.eval()
    target_device = torch.device(device)
    model.to(target_device)
    config = loss_config or LossConfig()
    history: list[dict[str, float]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            output, prepared = _forward_window(
                model, batch, skeleton, seed=int(seed) + batch_index * 997,
                corruption_config=corruption_config, device=target_device,
            )
            losses = compute_self_supervised_losses(output, prepared, config, skeleton)
            history.append({name: float(value.cpu()) for name, value in losses.as_dict().items()})
    if was_training:
        model.train()
    means = _mean_metrics(history)
    components = {
        "corruption_recovery": 1.0 / (1.0 + means["corruption_recovery"]),
        "bone_cv": 1.0 / (1.0 + means["trial_bone_length"]),
        "rotation_consistency": 1.0 / (1.0 + (means["circular_axial_rotation"] + means["so3_rotation"]) / 2.0),
        "identity_preservation": 1.0 / (1.0 + means["high_consensus_identity"]),
        "rom_retention": 1.0 / (1.0 + means["complete_cycle_rom"]),
    }
    return {"loss": means["total"], "score": sum(components.values()) / len(components), "components": components, "losses": means}


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
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    loss_config: LossConfig,
    skeleton: SkeletonSpec,
    provenance: Mapping[str, object],
    score: float,
    scheduler: object | None = None,
) -> None:
    """Persist enough provenance to reproduce a self-supervised selection decision."""
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_config": asdict(loss_config),
        "skeleton": _skeleton_metadata(skeleton),
        "provenance": dict(provenance),
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
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload
