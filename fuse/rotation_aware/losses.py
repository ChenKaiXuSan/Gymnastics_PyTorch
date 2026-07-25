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
    # 改法4: anchor the fused twist range-of-motion to the larger of the two clean
    # views' per-cycle ROM instead of the averaged pseudo-target, because
    # coordinate-space averaging systematically shrinks the twist. Weight 0 keeps
    # A4/A5/A6 byte-identical; the new ablations turn it on.
    complete_cycle_rom_peak_weight: float = 0.0
    # 改法3: one-sided penalty on fused twist rate exceeding the wider per-view
    # rate (observed envelope). Bounds over-rotation without suppressing the twist.
    # Default 0 keeps A4-A8 unchanged.
    observed_twist_rate_weight: float = 0.0
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
            "complete_cycle_rom_peak": self.complete_cycle_rom_peak_weight,
            "observed_twist_rate": self.observed_twist_rate_weight,
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
    complete_cycle_rom_peak: Tensor
    observed_twist_rate: Tensor
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
            "complete_cycle_rom_peak": self.complete_cycle_rom_peak,
            "observed_twist_rate": self.observed_twist_rate,
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
) -> tuple[Tensor, Tensor, Tensor]:
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
    return torch.where(valid[..., None], target, torch.zeros_like(target)), valid, consensus


def _rotation_chordal_proxy(left: Tensor, right: Tensor) -> Tensor:
    """A smooth SO(3) proxy equal to one minus the relative rotation cosine."""
    return (left - right).square().sum(dim=(-2, -1)) * 0.25


def _frame_mask(joint_mask: Tensor) -> Tensor:
    return joint_mask.bool().any(dim=-1)


def _complete_cycle_mask(batch: Mapping[str, object], frames: int, device: torch.device, *, batch_size: int) -> Tensor:
    complete = batch.get("complete_cycle")
    if complete is None:
        return torch.zeros((batch_size, frames), dtype=torch.bool, device=device)
    if not isinstance(complete, Tensor) or complete.shape != (batch_size,):
        raise ValueError("batch['complete_cycle'] must have shape [B]")
    return complete.bool().to(device=device)[:, None].expand(-1, frames)


def _unwrap_circular(values: Tensor, valid: Tensor) -> Tensor:
    """Unwrap each valid contiguous angle run with differentiable circular steps."""
    unwrapped = torch.zeros_like(values)
    for index in range(values.shape[1]):
        current_valid = valid[:, index]
        if index == 0:
            unwrapped[:, index] = torch.where(current_valid, values[:, index], torch.zeros_like(values[:, index]))
            continue
        continuous = current_valid & valid[:, index - 1]
        continued = unwrapped[:, index - 1] + circular_diff(values[:, index], values[:, index - 1])
        unwrapped[:, index] = torch.where(continuous, continued, torch.where(current_valid, values[:, index], torch.zeros_like(values[:, index])))
    return unwrapped


def _rom_loss_reference(prediction: Tensor, target: Tensor, valid: Tensor, complete: Tensor) -> Tensor:
    """Original per-frame ROM implementation retained for equivalence tests."""
    values: list[Tensor] = []
    for batch_index in range(prediction.shape[0]):
        usable = valid[batch_index] & complete[batch_index]
        index = 0
        while index < prediction.shape[1]:
            if not bool(usable[index]):
                index += 1
                continue
            end = index + 1
            while end < prediction.shape[1] and bool(usable[end]):
                end += 1
            run_valid = torch.ones((1, end - index), dtype=torch.bool, device=prediction.device)
            pred_run = _unwrap_circular(prediction[batch_index : batch_index + 1, index:end], run_valid)[0]
            target_run = _unwrap_circular(target[batch_index : batch_index + 1, index:end], run_valid)[0]
            values.append((pred_run.max() - pred_run.min() - target_run.max() + target_run.min()).square())
            index = end
    if not values:
        return prediction.new_zeros(())
    return torch.stack(values).mean()


def _unwrap_valid_run(values: Tensor) -> Tensor:
    """Unwrap one contiguous valid angle run without per-frame branching."""
    if values.shape[0] <= 1:
        return values
    steps = circular_diff(values[1:], values[:-1])
    return torch.cat((values[:1], values[:1] + torch.cumsum(steps, dim=0)))


def _contiguous_true_runs(mask: Tensor) -> list[tuple[int, int, int]]:
    """Return [batch, start, end) spans after one host transfer of mask changes."""
    padded = torch.nn.functional.pad(mask.bool(), (1, 1), value=False)
    changes = (padded[:, 1:] ^ padded[:, :-1]).nonzero(as_tuple=False).cpu()
    boundaries: dict[int, list[int]] = {}
    for batch_index, frame_index in changes.tolist():
        boundaries.setdefault(batch_index, []).append(frame_index)
    return [
        (batch_index, points[offset], points[offset + 1])
        for batch_index, points in boundaries.items()
        for offset in range(0, len(points), 2)
    ]


def _rom_loss(prediction: Tensor, target: Tensor, valid: Tensor, complete: Tensor) -> Tensor:
    values: list[Tensor] = []
    for batch_index, start, end in _contiguous_true_runs(valid & complete):
        pred_run = _unwrap_valid_run(prediction[batch_index, start:end])
        target_run = _unwrap_valid_run(target[batch_index, start:end])
        values.append((pred_run.max() - pred_run.min() - target_run.max() + target_run.min()).square())
    if not values:
        return prediction.new_zeros(())
    return torch.stack(values).mean()


def _twist_overshoot_loss(
    fused_omega: Tensor,
    fused_valid: Tensor,
    omega_face: Tensor,
    valid_face: Tensor,
    omega_side: Tensor,
    valid_side: Tensor,
) -> Tensor:
    """改法3: penalise fused twist speed exceeding the wider per-view speed.

    One-sided (hinge) bound: zero while the fused twist rate stays within the
    observed envelope (the larger |omega| a calibrated view actually saw that
    frame), positive only on over-rotation. This blocks the twist residual from
    overshooting without suppressing twist up to the observed level, so it does
    not fight the per-view-peak ROM anchor (改法4).
    """
    observed_envelope = torch.maximum(
        torch.where(valid_face, omega_face.abs(), torch.zeros_like(omega_face)),
        torch.where(valid_side, omega_side.abs(), torch.zeros_like(omega_side)),
    )
    rate_mask = fused_valid & (valid_face | valid_side)
    overshoot = torch.relu(fused_omega.abs() - observed_envelope)
    return _finite_masked_mean(overshoot.square(), rate_mask)


def _rom_peak_loss(
    fused_angle: Tensor,
    face_angle: Tensor,
    side_angle: Tensor,
    fused_valid: Tensor,
    face_valid: Tensor,
    side_valid: Tensor,
    complete: Tensor,
) -> Tensor:
    """Push the fused twist ROM toward the larger of the two clean views' ROM.

    Coordinate-space averaging shrinks the twist, so the averaged pseudo-target
    under-states the range of motion. Per complete cycle this anchors the fused
    range to whichever calibrated view actually observed the wider twist; the
    per-view targets are detached (they come from clean reference inputs). A view
    only contributes its ROM when it is valid across the whole run, which avoids
    unwrapping across gaps.
    """
    values: list[Tensor] = []
    for batch_index, start, end in _contiguous_true_runs(fused_valid & complete):
        fused_run = _unwrap_valid_run(fused_angle[batch_index, start:end])
        fused_rom = fused_run.max() - fused_run.min()
        view_roms: list[Tensor] = []
        for angle, valid in ((face_angle, face_valid), (side_angle, side_valid)):
            if bool(valid[batch_index, start:end].all()):
                run = _unwrap_valid_run(angle[batch_index, start:end])
                view_roms.append(run.max() - run.min())
        if not view_roms:
            continue
        target_rom = torch.stack(view_roms).max().detach()
        values.append((fused_rom - target_rom).square())
    if not values:
        return fused_angle.new_zeros(())
    return torch.stack(values).mean()


def compute_complete_cycle_rom_loss(
    output: FusionOutput,
    batch: Mapping[str, object],
    config: LossConfig,
    skeleton: SkeletonSpec,
) -> Tensor:
    """Compute only complete-cycle ROM, avoiding full-loss activations on long trials."""
    fused = output.fused_kpts
    if fused.ndim != 4 or fused.shape[-1] != 3:
        raise ValueError("FusionOutput.fused_kpts must have shape [B, T, J, 3]")
    batch_size, frames, joints, _ = fused.shape
    shape = (batch_size, frames, joints)
    if output.valid.shape != shape:
        raise ValueError("FusionOutput.valid must have shape [B, T, J]")
    if joints != len(skeleton.joint_names):
        raise ValueError("fused pose joint dimension must match the supplied SkeletonSpec")
    reference_face = _require_tensor(batch, "reference_face", fused.shape).to(
        device=fused.device, dtype=fused.dtype
    )
    reference_side = _require_tensor(batch, "reference_side", fused.shape).to(
        device=fused.device, dtype=fused.dtype
    )
    valid_face = _require_tensor(batch, "valid_face", shape).bool().to(fused.device)
    valid_side = _require_tensor(batch, "valid_side", shape).bool().to(fused.device)
    loss_mask = _require_tensor(batch, "loss_mask", shape).bool().to(fused.device)
    padding_mask = (
        _require_tensor(batch, "padding_mask", (batch_size, frames))
        .bool()
        .to(fused.device)
    )
    dt = _require_tensor(batch, "dt", (batch_size, frames)).to(
        device=fused.device, dtype=fused.dtype
    )
    quality_face = _require_tensor(batch, "quality_face", (batch_size, frames)).to(
        device=fused.device, dtype=fused.dtype
    )
    quality_side = _require_tensor(batch, "quality_side", (batch_size, frames)).to(
        device=fused.device, dtype=fused.dtype
    )
    coordinate_mask = loss_mask & padding_mask[..., None]
    target, target_valid, _ = _pseudo_target(
        reference_face,
        reference_side,
        valid_face,
        valid_side,
        quality_face,
        quality_side,
        config,
    )
    fused, fused_valid = _safe_points(
        fused, output.valid.to(device=fused.device) & coordinate_mask
    )
    coordinate_mask &= fused_valid & target_valid
    target_trunk = extract_trunk_features(
        target, target_valid & coordinate_mask, skeleton, dt=dt
    )
    fused_trunk = extract_trunk_features(fused, fused_valid, skeleton, dt=dt)
    axial_mask = (
        _frame_mask(coordinate_mask)
        & target_trunk.angle_valid
        & fused_trunk.angle_valid
    )
    complete = _complete_cycle_mask(
        batch, frames, fused.device, batch_size=batch_size
    )
    return _rom_loss(fused_trunk.angle, target_trunk.angle, axial_mask, complete)


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
    if output.valid.shape != shape:
        raise ValueError("FusionOutput.valid must have shape [B, T, J]")
    if output.delta_kpts.shape != fused.shape:
        raise ValueError("FusionOutput.delta_kpts must have shape [B, T, J, 3]")
    reference_face = _require_tensor(batch, "reference_face", fused.shape)
    reference_side = _require_tensor(batch, "reference_side", fused.shape)
    valid_face = _require_tensor(batch, "valid_face", shape).bool()
    valid_side = _require_tensor(batch, "valid_side", shape).bool()
    loss_mask = _require_tensor(batch, "loss_mask", shape).bool()
    padding_mask = _require_tensor(batch, "padding_mask", (batch_size, frames)).bool()
    dt = _require_tensor(batch, "dt", (batch_size, frames)).to(device=fused.device, dtype=fused.dtype)
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
    target, target_valid, consensus_mask = _pseudo_target(
        reference_face, reference_side, valid_face, valid_side, quality_face, quality_side, config
    )
    fused, fused_valid = _safe_points(fused, output.valid.to(device=fused.device) & coordinate_mask)
    coordinate_mask = coordinate_mask & fused_valid & target_valid
    corrupted_mask = coordinate_mask & (face_corruption.to(device=fused.device) | side_corruption.to(device=fused.device))
    identity_mask = coordinate_mask & consensus_mask & ~(face_corruption.to(device=fused.device) | side_corruption.to(device=fused.device))
    coordinate_error = (fused - target).square().mean(dim=-1)
    corruption_recovery = _finite_masked_mean(coordinate_error, corrupted_mask)
    high_consensus_identity = _finite_masked_mean(coordinate_error, identity_mask)

    target_trunk = extract_trunk_features(target, target_valid & coordinate_mask, skeleton, dt=dt)
    fused_trunk = extract_trunk_features(fused, fused_valid, skeleton, dt=dt)
    shared_frame = _frame_mask(coordinate_mask)
    axial_mask = shared_frame & target_trunk.angle_valid & fused_trunk.angle_valid
    circular_axial_rotation = _finite_masked_mean(circular_diff(fused_trunk.angle, target_trunk.angle).square(), axial_mask)
    rotation_mask = shared_frame & target_trunk.rotation_valid & fused_trunk.rotation_valid
    so3_rotation = _finite_masked_mean(_rotation_chordal_proxy(fused_trunk.rotation, target_trunk.rotation), rotation_mask)

    target_pose = extract_pose_features(target, target_valid & coordinate_mask, skeleton, dt=dt)
    fused_pose = extract_pose_features(fused, fused_valid, skeleton, dt=dt)
    bone_mask = target_pose.bone_valid & fused_pose.bone_valid
    baseline_shape = (batch_size, len(skeleton.bones))
    target_bone_baseline = _require_tensor(batch, "trial_bone_baseline", baseline_shape).to(device=fused.device, dtype=fused.dtype)
    baseline_valid = _require_tensor(batch, "trial_bone_baseline_valid", baseline_shape).bool().to(device=fused.device)
    bone_mask = bone_mask & baseline_valid[:, None, :]
    relative_bone_error = (fused_pose.bone_lengths - target_bone_baseline[:, None, :]).abs() / target_bone_baseline[:, None, :].clamp_min(config.epsilon)
    trial_bone_length = _finite_masked_mean(relative_bone_error.square(), bone_mask)
    rate_dt = dt[:, 1:]
    rate_valid = torch.isfinite(rate_dt) & (rate_dt > 0)
    local_bone_mask = bone_mask[:, 1:] & bone_mask[:, :-1] & rate_valid[..., None]
    fused_rate = (fused_pose.bone_lengths[:, 1:] - fused_pose.bone_lengths[:, :-1]) / rate_dt[..., None].clamp_min(config.epsilon)
    target_rate = (target_pose.bone_lengths[:, 1:] - target_pose.bone_lengths[:, :-1]) / rate_dt[..., None].clamp_min(config.epsilon)
    local_bone_error = fused_rate - target_rate
    local_rigidity = _finite_masked_mean(local_bone_error.square(), local_bone_mask)

    previous_dt = dt[:, 1:-1]
    next_dt = dt[:, 2:]
    interval_valid = torch.isfinite(previous_dt) & torch.isfinite(next_dt) & (previous_dt > 0) & (next_dt > 0)
    acceleration_mask = coordinate_mask[:, 2:] & coordinate_mask[:, 1:-1] & coordinate_mask[:, :-2] & interval_valid[..., None]
    target_velocity_previous = (target[:, 1:-1] - target[:, :-2]) / previous_dt[..., None, None].clamp_min(config.epsilon)
    target_velocity_next = (target[:, 2:] - target[:, 1:-1]) / next_dt[..., None, None].clamp_min(config.epsilon)
    fused_velocity_previous = (fused[:, 1:-1] - fused[:, :-2]) / previous_dt[..., None, None].clamp_min(config.epsilon)
    fused_velocity_next = (fused[:, 2:] - fused[:, 1:-1]) / next_dt[..., None, None].clamp_min(config.epsilon)
    denominator = (previous_dt + next_dt)[..., None, None].clamp_min(config.epsilon)
    target_acceleration = 2.0 * (target_velocity_next - target_velocity_previous) / denominator
    fused_acceleration = 2.0 * (fused_velocity_next - fused_velocity_previous) / denominator
    adaptive_weight = 1.0 / (1.0 + torch.linalg.vector_norm(target_acceleration, dim=-1).detach())
    acceleration_error = (fused_acceleration - target_acceleration).square().mean(dim=-1) * adaptive_weight
    adaptive_temporal_acceleration = _finite_masked_mean(acceleration_error, acceleration_mask)

    delta = torch.where(fused_valid[..., None] & torch.isfinite(output.delta_kpts), output.delta_kpts, torch.zeros_like(output.delta_kpts))
    minimal_residual = _finite_masked_mean(delta.square().mean(dim=-1), coordinate_mask)
    complete = _complete_cycle_mask(batch, frames, fused.device, batch_size=batch_size)
    complete_cycle_rom = _rom_loss(fused_trunk.angle, target_trunk.angle, axial_mask, complete)

    # 改法4/改法3 both need each clean view's own trunk kinematics; extract once
    # when either weight is active (two extra trunk extractions on the references).
    need_view_trunks = (
        config.complete_cycle_rom_peak_weight > 0 or config.observed_twist_rate_weight > 0
    )
    if need_view_trunks:
        face_ref_trunk = extract_trunk_features(reference_face, valid_face, skeleton, dt=dt)
        side_ref_trunk = extract_trunk_features(reference_side, valid_side, skeleton, dt=dt)

    # 改法4: anchor the fused twist range-of-motion to the wider per-view ROM.
    if config.complete_cycle_rom_peak_weight > 0:
        complete_cycle_rom_peak = _rom_peak_loss(
            fused_trunk.angle,
            face_ref_trunk.angle,
            side_ref_trunk.angle,
            axial_mask,
            fused_trunk.angle_valid & face_ref_trunk.angle_valid,
            fused_trunk.angle_valid & side_ref_trunk.angle_valid,
            complete,
        )
    else:
        complete_cycle_rom_peak = fused.new_zeros(())

    # 改法3: one-sided anchor on the fused twist RATE. Penalise it only when it
    # exceeds the wider per-view twist speed (the observed envelope = the larger
    # |omega| a calibrated camera actually saw that frame). This blocks the twist
    # residual from over-rotating (A8's failure) WITHOUT suppressing the twist up
    # to the observed level, so it does not fight the per-view-peak ROM anchor
    # (改法4). A plain equality anchor to the average rate would instead cancel the
    # twist entirely.
    if config.observed_twist_rate_weight > 0:
        observed_twist_rate = _twist_overshoot_loss(
            fused_trunk.omega,
            fused_trunk.omega_valid,
            face_ref_trunk.omega,
            face_ref_trunk.omega_valid,
            side_ref_trunk.omega,
            side_ref_trunk.omega_valid,
        )
    else:
        observed_twist_rate = fused.new_zeros(())

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
        "complete_cycle_rom_peak": complete_cycle_rom_peak,
        "observed_twist_rate": observed_twist_rate,
    }
    total = fused.new_zeros(())
    for name, value in values.items():
        weight = config.weights[name]
        if weight > 0:
            total = total + weight * value
    total = torch.where(torch.isfinite(total), total, fused.new_zeros(()))
    return LossBreakdown(total=total, **values)
