"""Masked self-supervised objectives for rotation-aware fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from .config import SkeletonSpec
from .features import extract_pose_features
from .model import FusionOutput
from .trunk import circular_diff, extract_trunk_features


@dataclass(frozen=True)
class LossConfig:
    """Fixed weights and pseudo-target boundaries for self-supervision."""

    corruption_recovery_weight: float = 1.0
    high_consensus_identity_weight: float = 1.0
    circular_axial_rotation_weight: float = 1.0
    so3_rotation_weight: float = 1.0
    trial_bone_length_weight: float = 1.0
    local_rigidity_weight: float = 1.0
    adaptive_temporal_acceleration_weight: float = 1.0
    minimal_residual_weight: float = 0.05
    complete_cycle_rom_weight: float = 1.0
    consensus_distance: float = 0.10
    quality_advantage_ratio: float = 1.5
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.consensus_distance < 0 or self.quality_advantage_ratio < 1 or self.epsilon <= 0:
            raise ValueError("loss boundaries must be non-negative and quality_advantage_ratio must be at least one")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("loss weights must be non-negative")

    @property
    def weights(self) -> Mapping[str, float]:
        return {
            "corruption_recovery": self.corruption_recovery_weight,
            "high_consensus_identity": self.high_consensus_identity_weight,
            "circular_axial_rotation": self.circular_axial_rotation_weight,
            "so3_rotation": self.so3_rotation_weight,
            "trial_bone_length": self.trial_bone_length_weight,
            "local_rigidity": self.local_rigidity_weight,
            "adaptive_temporal_acceleration": self.adaptive_temporal_acceleration_weight,
            "minimal_residual": self.minimal_residual_weight,
            "complete_cycle_rom": self.complete_cycle_rom_weight,
        }


@dataclass(frozen=True)
class LossBreakdown:
    """The nine finite, separately inspectable self-supervised objectives."""

    corruption_recovery: Tensor
    high_consensus_identity: Tensor
    circular_axial_rotation: Tensor
    so3_rotation: Tensor
    trial_bone_length: Tensor
    local_rigidity: Tensor
    adaptive_temporal_acceleration: Tensor
    minimal_residual: Tensor
    complete_cycle_rom: Tensor
    total: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "corruption_recovery": self.corruption_recovery,
            "high_consensus_identity": self.high_consensus_identity,
            "circular_axial_rotation": self.circular_axial_rotation,
            "so3_rotation": self.so3_rotation,
            "trial_bone_length": self.trial_bone_length,
            "local_rigidity": self.local_rigidity,
            "adaptive_temporal_acceleration": self.adaptive_temporal_acceleration,
            "minimal_residual": self.minimal_residual,
            "complete_cycle_rom": self.complete_cycle_rom,
            "total": self.total,
        }


def _require_tensor(batch: Mapping[str, object], name: str, shape: tuple[int, ...] | None = None) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"batch[{name!r}] must be a tensor")
    if shape is not None and value.shape != shape:
        raise ValueError(f"batch[{name!r}] must have shape {shape}")
    return value


def _finite_masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("loss values and masks must have equal shapes")
    usable = mask.bool() & torch.isfinite(values)
    safe = torch.where(usable, values, torch.zeros_like(values))
    return safe.sum() / usable.sum().clamp_min(1).to(dtype=values.dtype)


def _safe_points(points: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    finite = torch.isfinite(points).all(dim=-1)
    effective = valid.bool() & finite
    return torch.where(effective[..., None], points, torch.zeros_like(points)), effective


def _pseudo_target(
    face: Tensor,
    side: Tensor,
    valid_face: Tensor,
    valid_side: Tensor,
    quality_face: Tensor,
    quality_side: Tensor,
    config: LossConfig,
) -> tuple[Tensor, Tensor]:
    """Construct a quality-bounded reference without selecting weak disagreements."""
    face, valid_face = _safe_points(face, valid_face)
    side, valid_side = _safe_points(side, valid_side)
    q_face = torch.where(torch.isfinite(quality_face), quality_face.detach().clamp_min(0), torch.zeros_like(quality_face))
    q_side = torch.where(torch.isfinite(quality_side), quality_side.detach().clamp_min(0), torch.zeros_like(quality_side))
    common = valid_face & valid_side
    distance = torch.linalg.vector_norm(face - side, dim=-1)
    consensus = common & (distance <= config.consensus_distance)
    face_dominant = common & (q_face[..., None] > config.quality_advantage_ratio * q_side[..., None])
    side_dominant = common & (q_side[..., None] > config.quality_advantage_ratio * q_face[..., None])
    face_only = valid_face & ~valid_side
    side_only = valid_side & ~valid_face
    weights = q_face[..., None] + q_side[..., None]
    normalized_face = torch.where(weights > config.epsilon, q_face[..., None] / weights, torch.full_like(weights, 0.5))
    normalized_side = torch.where(weights > config.epsilon, q_side[..., None] / weights, torch.full_like(weights, 0.5))
    averaged = normalized_face[..., None] * face + normalized_side[..., None] * side
    target = torch.where(face_dominant[..., None], face, averaged)
    target = torch.where(side_dominant[..., None], side, target)
    target = torch.where(face_only[..., None], face, target)
    target = torch.where(side_only[..., None], side, target)
    valid = consensus | face_dominant | side_dominant | face_only | side_only
    return torch.where(valid[..., None], target, torch.zeros_like(target)), valid


def _rotation_distance(left: Tensor, right: Tensor) -> Tensor:
    relative = left.transpose(-1, -2) @ right
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _frame_mask(joint_mask: Tensor) -> Tensor:
    return joint_mask.bool().any(dim=-1)


def _complete_cycle_mask(batch: Mapping[str, object], frames: int, device: torch.device) -> Tensor:
    complete = batch.get("complete_cycle")
    if complete is None:
        return torch.ones((1, frames), dtype=torch.bool, device=device)
    if not isinstance(complete, Tensor) or complete.ndim != 1:
        raise ValueError("batch['complete_cycle'] must have shape [B]")
    return complete.bool().to(device=device)[:, None].expand(-1, frames)


def _rom_loss(prediction: Tensor, target: Tensor, valid: Tensor, complete: Tensor) -> Tensor:
    values: list[Tensor] = []
    for batch_index in range(prediction.shape[0]):
        usable = valid[batch_index] & complete[batch_index]
        if usable.any():
            pred_values = prediction[batch_index][usable]
            target_values = target[batch_index][usable]
            values.append((pred_values.max() - pred_values.min() - target_values.max() + target_values.min()).square())
    if not values:
        return prediction.new_zeros(())
    return torch.stack(values).mean()


def compute_self_supervised_losses(
    output: FusionOutput,
    batch: Mapping[str, object],
    config: LossConfig,
    skeleton: SkeletonSpec,
) -> LossBreakdown:
    """Compute all objectives with every coordinate masked before it is inspected."""
    fused = output.fused_kpts
    if fused.ndim != 4 or fused.shape[-1] != 3:
        raise ValueError("FusionOutput.fused_kpts must have shape [B, T, J, 3]")
    batch_size, frames, joints, _ = fused.shape
    shape = (batch_size, frames, joints)
    reference_face = _require_tensor(batch, "reference_face", fused.shape)
    reference_side = _require_tensor(batch, "reference_side", fused.shape)
    valid_face = _require_tensor(batch, "valid_face", shape).bool()
    valid_side = _require_tensor(batch, "valid_side", shape).bool()
    loss_mask = _require_tensor(batch, "loss_mask", shape).bool()
    padding_mask = _require_tensor(batch, "padding_mask", (batch_size, frames)).bool()
    face_corruption = _require_tensor(batch, "face_corruption_mask", shape).bool()
    side_corruption = _require_tensor(batch, "side_corruption_mask", shape).bool()
    quality_face = _require_tensor(batch, "quality_face", (batch_size, frames)).to(device=fused.device, dtype=fused.dtype)
    quality_side = _require_tensor(batch, "quality_side", (batch_size, frames)).to(device=fused.device, dtype=fused.dtype)
    if joints != len(skeleton.joint_names):
        raise ValueError("fused pose joint dimension must match the supplied SkeletonSpec")

    reference_face = reference_face.to(device=fused.device, dtype=fused.dtype)
    reference_side = reference_side.to(device=fused.device, dtype=fused.dtype)
    valid_face = valid_face.to(device=fused.device)
    valid_side = valid_side.to(device=fused.device)
    coordinate_mask = loss_mask.to(device=fused.device) & padding_mask.to(device=fused.device)[..., None]
    target, target_valid = _pseudo_target(
        reference_face, reference_side, valid_face, valid_side, quality_face, quality_side, config
    )
    fused, fused_valid = _safe_points(fused, output.valid.to(device=fused.device) & coordinate_mask)
    coordinate_mask = coordinate_mask & fused_valid & target_valid
    corrupted_mask = coordinate_mask & (face_corruption.to(device=fused.device) | side_corruption.to(device=fused.device))
    identity_mask = coordinate_mask & ~(face_corruption.to(device=fused.device) | side_corruption.to(device=fused.device))
    coordinate_error = (fused - target).square().mean(dim=-1)
    corruption_recovery = _finite_masked_mean(coordinate_error, corrupted_mask)
    high_consensus_identity = _finite_masked_mean(coordinate_error, identity_mask)

    target_trunk = extract_trunk_features(target, target_valid & coordinate_mask, skeleton, dt=1.0)
    fused_trunk = extract_trunk_features(fused, fused_valid, skeleton, dt=1.0)
    shared_frame = _frame_mask(coordinate_mask)
    axial_mask = shared_frame & target_trunk.angle_valid & fused_trunk.angle_valid
    circular_axial_rotation = _finite_masked_mean(circular_diff(fused_trunk.angle, target_trunk.angle).square(), axial_mask)
    rotation_mask = shared_frame & target_trunk.rotation_valid & fused_trunk.rotation_valid
    so3_rotation = _finite_masked_mean(_rotation_distance(fused_trunk.rotation, target_trunk.rotation).square(), rotation_mask)

    target_pose = extract_pose_features(target, target_valid & coordinate_mask, skeleton, dt=1.0)
    fused_pose = extract_pose_features(fused, fused_valid, skeleton, dt=1.0)
    bone_mask = target_pose.bone_valid & fused_pose.bone_valid
    relative_bone_error = (fused_pose.bone_lengths - target_pose.bone_lengths).abs() / target_pose.bone_lengths.clamp_min(config.epsilon)
    trial_bone_length = _finite_masked_mean(relative_bone_error.square(), bone_mask)
    local_bone_mask = bone_mask[:, 1:] & bone_mask[:, :-1]
    local_bone_error = (fused_pose.bone_lengths[:, 1:] - fused_pose.bone_lengths[:, :-1]) - (
        target_pose.bone_lengths[:, 1:] - target_pose.bone_lengths[:, :-1]
    )
    local_rigidity = _finite_masked_mean(local_bone_error.square(), local_bone_mask)

    acceleration_mask = coordinate_mask[:, 2:] & coordinate_mask[:, 1:-1] & coordinate_mask[:, :-2]
    target_acceleration = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    fused_acceleration = fused[:, 2:] - 2.0 * fused[:, 1:-1] + fused[:, :-2]
    adaptive_weight = 1.0 / (1.0 + torch.linalg.vector_norm(target_acceleration, dim=-1).detach())
    acceleration_error = (fused_acceleration - target_acceleration).square().mean(dim=-1) * adaptive_weight
    adaptive_temporal_acceleration = _finite_masked_mean(acceleration_error, acceleration_mask)

    delta = torch.where(fused_valid[..., None] & torch.isfinite(output.delta_kpts), output.delta_kpts, torch.zeros_like(output.delta_kpts))
    minimal_residual = _finite_masked_mean(delta.square().mean(dim=-1), coordinate_mask)
    complete = _complete_cycle_mask(batch, frames, fused.device)
    complete_cycle_rom = _rom_loss(fused_trunk.angle, target_trunk.angle, axial_mask, complete)

    values = {
        "corruption_recovery": corruption_recovery,
        "high_consensus_identity": high_consensus_identity,
        "circular_axial_rotation": circular_axial_rotation,
        "so3_rotation": so3_rotation,
        "trial_bone_length": trial_bone_length,
        "local_rigidity": local_rigidity,
        "adaptive_temporal_acceleration": adaptive_temporal_acceleration,
        "minimal_residual": minimal_residual,
        "complete_cycle_rom": complete_cycle_rom,
    }
    total = fused.new_zeros(())
    for name, value in values.items():
        total = total + config.weights[name] * value
    total = torch.where(torch.isfinite(total), total, fused.new_zeros(()))
    return LossBreakdown(total=total, **values)
