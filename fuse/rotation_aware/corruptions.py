"""Deterministic synthetic corruptions for paired 3D pose windows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .config import SkeletonSpec

_FAMILIES = frozenset(
    {
        "joint_dropout",
        "temporal_block_dropout",
        "spike_noise",
        "random_walk_drift",
        "thorax_rotation_bias",
        "freeze_segment",
        "integer_time_shift",
    }
)


@dataclass(frozen=True)
class CorruptionConfig:
    """Parameters for all supported corruption families.

    A family runs only when listed in ``enabled_families``.  The probabilities
    control the affected joints or frames within that family, allowing a fixed
    evaluation manifest to replay the exact same sampled corruption.
    """

    enabled_families: tuple[str, ...] = tuple(sorted(_FAMILIES))
    joint_dropout_probability: float = 0.05
    temporal_block_probability: float = 0.25
    block_length: int = 16
    spike_probability: float = 0.02
    spike_scale: float = 0.10
    drift_probability: float = 0.10
    drift_scale: float = 0.01
    rotation_probability: float = 0.10
    rotation_degrees: float = 12.0
    freeze_probability: float = 0.10
    freeze_length: int = 12
    time_shift_probability: float = 0.10
    max_time_shift: int = 4

    def __post_init__(self) -> None:
        unknown = set(self.enabled_families) - _FAMILIES
        if unknown:
            raise ValueError(f"Unknown corruption families: {sorted(unknown)}")
        for name in (
            "joint_dropout_probability",
            "temporal_block_probability",
            "spike_probability",
            "drift_probability",
            "rotation_probability",
            "freeze_probability",
            "time_shift_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.block_length < 1 or self.freeze_length < 1 or self.max_time_shift < 1:
            raise ValueError("block_length, freeze_length, and max_time_shift must be positive")


@dataclass(frozen=True)
class CorruptionBatch:
    """Corrupted inputs plus untouched references and exact per-joint masks."""

    corrupted_face: torch.Tensor
    corrupted_side: torch.Tensor
    corrupted_valid_face: torch.Tensor
    corrupted_valid_side: torch.Tensor
    reference_face: torch.Tensor
    reference_side: torch.Tensor
    valid_face: torch.Tensor
    valid_side: torch.Tensor
    face_corruption_mask: torch.Tensor
    side_corruption_mask: torch.Tensor


def _check_inputs(points: torch.Tensor, valid: torch.Tensor, name: str) -> None:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T, J, 3]")
    if valid.shape != points.shape[:2] or valid.dtype is not torch.bool:
        raise ValueError(f"valid_{name} must be a bool tensor with shape [T, J]")
    if points.device.type != "cpu" or valid.device.type != "cpu":
        raise ValueError("Synthetic corruption currently requires CPU tensors")


def _bernoulli(shape: tuple[int, ...], probability: float, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator) < probability


def _apply_view(
    points: torch.Tensor,
    valid: torch.Tensor,
    generator: torch.Generator,
    config: CorruptionConfig,
    skeleton: SkeletonSpec | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, output_valid = points.clone(), valid.clone()
    frames, joints = valid.shape

    if "joint_dropout" in config.enabled_families:
        selected = _bernoulli((frames, joints), config.joint_dropout_probability, generator) & output_valid
        output[selected] = 0
        output_valid[selected] = False

    if "temporal_block_dropout" in config.enabled_families:
        selected_joints = _bernoulli((joints,), config.temporal_block_probability, generator)
        length = min(config.block_length, frames)
        start = int(torch.randint(frames - length + 1, (), generator=generator))
        selected = torch.zeros_like(output_valid)
        selected[start : start + length, selected_joints] = True
        selected &= output_valid
        output[selected] = 0
        output_valid[selected] = False

    if "spike_noise" in config.enabled_families:
        selected = _bernoulli((frames, joints), config.spike_probability, generator) & output_valid
        noise = torch.randn(output.shape, generator=generator, dtype=output.dtype) * config.spike_scale
        output = output + noise * selected.unsqueeze(-1)

    if "random_walk_drift" in config.enabled_families:
        selected = _bernoulli((joints,), config.drift_probability, generator)
        increments = torch.randn(output.shape, generator=generator, dtype=output.dtype) * config.drift_scale
        drift = torch.cumsum(increments, dim=0) * selected.view(1, joints, 1)
        output = output + drift * output_valid.unsqueeze(-1)

    if "thorax_rotation_bias" in config.enabled_families and skeleton is not None:
        selected_frames = _bernoulli((frames,), config.rotation_probability, generator)
        pivot, pivot_valid = _resolve_thorax_pivot(output, output_valid, skeleton)
        angles = (torch.rand((frames,), generator=generator) * 2 - 1) * torch.deg2rad(
            torch.tensor(config.rotation_degrees, dtype=output.dtype)
        )
        cos, sin = torch.cos(angles), torch.sin(angles)
        rotation = torch.zeros((frames, 3, 3), dtype=output.dtype)
        rotation[:, 0, 0], rotation[:, 0, 1] = cos, -sin
        rotation[:, 1, 0], rotation[:, 1, 1] = sin, cos
        rotation[:, 2, 2] = 1
        rotated = torch.einsum("tjc,tdc->tjd", output - pivot, rotation) + pivot
        output = torch.where(
            selected_frames[:, None, None] & pivot_valid[:, None, None] & output_valid.unsqueeze(-1),
            rotated,
            output,
        )

    if "freeze_segment" in config.enabled_families:
        selected_joints = _bernoulli((joints,), config.freeze_probability, generator)
        length = min(config.freeze_length, frames)
        start = int(torch.randint(frames - length + 1, (), generator=generator))
        if start + length > 1:
            frozen = output[start : start + 1].expand(length, -1, -1)
            mask = selected_joints.view(1, joints, 1) & output_valid[start : start + length].unsqueeze(-1)
            output[start : start + length] = torch.where(mask, frozen, output[start : start + length])

    if "integer_time_shift" in config.enabled_families:
        selected_joints = _bernoulli((joints,), config.time_shift_probability, generator)
        magnitude = int(torch.randint(1, config.max_time_shift + 1, (), generator=generator))
        direction = -1 if bool(torch.randint(2, (), generator=generator)) else 1
        shift = direction * magnitude
        source = (torch.arange(frames) - shift).clamp(0, frames - 1)
        shifted_points, shifted_valid = output[source], output_valid[source]
        target_mask = selected_joints.view(1, joints) & output_valid
        output = torch.where(target_mask.unsqueeze(-1), shifted_points, output)
        output_valid = torch.where(target_mask, shifted_valid, output_valid)

    return output, output_valid


def _resolve_thorax_pivot(
    points: torch.Tensor, valid: torch.Tensor, skeleton: SkeletonSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the configured thorax role per frame, including midpoint fallback."""
    if points.shape[1] != len(skeleton.joint_names):
        raise ValueError("pose joint dimension must match the supplied SkeletonSpec")
    role = skeleton.role("thorax")
    indices = tuple(skeleton.joint_index(name) for name in role.joints)
    primary = points[:, indices].mean(dim=1, keepdim=True)
    primary_valid = valid[:, indices].all(dim=1)
    if not role.fallback:
        return primary, primary_valid
    fallback_indices = tuple(skeleton.joint_index(name) for name in role.fallback)
    fallback = points[:, fallback_indices].mean(dim=1, keepdim=True)
    fallback_valid = valid[:, fallback_indices].all(dim=1)
    return torch.where(primary_valid[:, None, None], primary, fallback), primary_valid | fallback_valid


def apply_corruptions(
    face: torch.Tensor,
    side: torch.Tensor,
    valid_face: torch.Tensor,
    valid_side: torch.Tensor,
    *,
    seed: int,
    config: CorruptionConfig | None = None,
    skeleton: SkeletonSpec | None = None,
) -> CorruptionBatch:
    """Apply seeded corruptions without modifying references.

    Thorax rotation is applied only when a ``SkeletonSpec`` is supplied, so the
    default corruption configuration remains usable for generic pose tensors.
    """
    _check_inputs(face, valid_face, "face")
    _check_inputs(side, valid_side, "side")
    if face.shape != side.shape:
        raise ValueError("face and side must have equal shape")
    if skeleton is not None and len(skeleton.joint_names) != face.shape[1]:
        raise ValueError("pose joint dimension must match the supplied SkeletonSpec")
    config = config or CorruptionConfig()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    reference_face, reference_side = face.clone(), side.clone()
    reference_valid_face, reference_valid_side = valid_face.clone(), valid_side.clone()
    corrupted_face, corrupted_valid_face = _apply_view(face, valid_face, generator, config, skeleton)
    corrupted_side, corrupted_valid_side = _apply_view(side, valid_side, generator, config, skeleton)
    face_mask = ((corrupted_face != reference_face).any(dim=-1) | (corrupted_valid_face != reference_valid_face))
    side_mask = ((corrupted_side != reference_side).any(dim=-1) | (corrupted_valid_side != reference_valid_side))
    return CorruptionBatch(
        corrupted_face=corrupted_face,
        corrupted_side=corrupted_side,
        corrupted_valid_face=corrupted_valid_face,
        corrupted_valid_side=corrupted_valid_side,
        reference_face=reference_face,
        reference_side=reference_side,
        valid_face=reference_valid_face,
        valid_side=reference_valid_side,
        face_corruption_mask=face_mask,
        side_corruption_mask=side_mask,
    )


def stable_window_seed(seed: int, window_id: str) -> int:
    """Derive a batch-order-independent corruption seed for one stable window id."""
    encoded = f"{int(seed)}:{window_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


def write_corruption_manifest(
    path: str | Path,
    window_ids: Sequence[str],
    *,
    seed: int,
    config: CorruptionConfig | None = None,
    window_length: int | None = None,
    stride: int | None = None,
) -> dict[str, object]:
    """Persist stable per-window seeds for a fixed evaluation corruption suite."""
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("window_ids must be unique")
    manifest: dict[str, object] = {
        "seed": int(seed),
        "config": json.loads(json.dumps(asdict(config or CorruptionConfig()))),
        "windows": {window_id: stable_window_seed(seed, window_id) for window_id in sorted(window_ids)},
    }
    if window_length is not None or stride is not None:
        if not isinstance(window_length, int) or not isinstance(stride, int):
            raise ValueError("window_length and stride must be supplied together")
        manifest["window"] = {"length": window_length, "stride": stride}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
