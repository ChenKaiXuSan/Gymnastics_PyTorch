"""CPU long-sequence inference and compact face-reference exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fuse.experiment_matrix import kpts_world_to_body

from .config import SkeletonSpec
from .features import (
    compute_disagreement_features,
    compute_quality_features,
    extract_pose_features,
)
from .geometry import CanonicalTransform, canonicalize_pose, restore_pose
from .model import RotationAwareFusionModel
from .schema import PosePairTrial
from .trunk import extract_trunk_features


@dataclass(frozen=True)
class CanonicalTrial:
    """A raw cache trial transformed once, before any window is selected."""

    trial: PosePairTrial
    face_transform: CanonicalTransform
    side_transform: CanonicalTransform

    def restore_face(self, points: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(
            np.array(points, dtype=np.float32, copy=True)
        ).unsqueeze(0)
        return (
            restore_pose(tensor, self.face_transform)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


@dataclass(frozen=True)
class InferenceResult:
    sequence_path: Path
    frames: int
    metadata: Mapping[str, Any]


def canonicalize_trial(trial: PosePairTrial, skeleton: SkeletonSpec) -> CanonicalTrial:
    """Canonicalize the complete raw trial so scales cannot depend on crop size."""
    face = torch.from_numpy(np.array(trial.face, copy=True)).unsqueeze(0)
    side = torch.from_numpy(np.array(trial.side, copy=True)).unsqueeze(0)
    face_valid = torch.from_numpy(np.array(trial.valid_face, copy=True)).unsqueeze(0)
    side_valid = torch.from_numpy(np.array(trial.valid_side, copy=True)).unsqueeze(0)
    face_canonical = canonicalize_pose(face, face_valid, skeleton)
    side_canonical = canonicalize_pose(side, side_valid, skeleton)
    metadata = dict(trial.source_metadata)
    metadata["coordinate_system"] = "canonical_pelvis_trial_scale"
    canonical = PosePairTrial(
        face=face_canonical.points.squeeze(0).cpu().numpy(),
        side=side_canonical.points.squeeze(0).cpu().numpy(),
        valid_face=face_canonical.valid.squeeze(0).cpu().numpy(),
        valid_side=side_canonical.valid.squeeze(0).cpu().numpy(),
        timestamps=trial.timestamps,
        face_map=trial.face_map,
        side_map=trial.side_map,
        joint_names=trial.joint_names,
        person_id=trial.person_id,
        trial_id=trial.trial_id,
        fps=trial.fps,
        source_metadata=metadata,
    )
    return CanonicalTrial(canonical, face_canonical.transform, side_canonical.transform)


def overlap_taper(length: int) -> np.ndarray:
    """Deterministic positive Hann-like weights for overlap-add reconstruction."""
    if length < 1:
        raise ValueError("window length must be positive")
    position = (np.arange(length, dtype=np.float32) + 0.5) / float(length)
    return np.sin(np.pi * position).astype(np.float32)


def _starts(frames: int, length: int, stride: int) -> tuple[int, ...]:
    if frames < 1 or length < 1 or stride < 1:
        raise ValueError("frames, length, and stride must be positive")
    if frames <= length:
        return (0,)
    starts = list(range(0, frames - length + 1, stride))
    if starts[-1] != frames - length:
        starts.append(frames - length)
    return tuple(starts)


def _dt(trial: PosePairTrial) -> torch.Tensor:
    values: np.ndarray = np.full(
        len(trial.timestamps), 1.0 / trial.fps, dtype=np.float32
    )
    if len(values) > 1:
        values[1:] = np.diff(trial.timestamps).astype(np.float32)
    return torch.from_numpy(values[None])


def _forward(
    model: RotationAwareFusionModel,
    face: torch.Tensor,
    side: torch.Tensor,
    valid_face: torch.Tensor,
    valid_side: torch.Tensor,
    skeleton: SkeletonSpec,
    dt: torch.Tensor,
):
    temporal = valid_face.any(dim=-1) | valid_side.any(dim=-1)
    dt = torch.where(temporal, dt, torch.zeros_like(dt))
    face = torch.where(valid_face[..., None], face, torch.zeros_like(face))
    side = torch.where(valid_side[..., None], side, torch.zeros_like(side))
    face_trunk = extract_trunk_features(face, valid_face, skeleton, dt)
    side_trunk = extract_trunk_features(side, valid_side, skeleton, dt)
    face_features = (
        extract_pose_features(face, valid_face, skeleton, dt),
        compute_quality_features(face, valid_face, face_trunk, skeleton),
    )
    side_features = (
        extract_pose_features(side, valid_side, skeleton, dt),
        compute_quality_features(side, valid_side, side_trunk, skeleton),
    )
    from .features import FeatureBundle

    cross = compute_disagreement_features(
        face, side, face_trunk, side_trunk, valid_face, valid_side
    )
    output = model(
        face,
        side,
        FeatureBundle(*face_features),
        FeatureBundle(*side_features),
        cross,
        valid_face,
        valid_side,
        temporal_valid=temporal,
        dt=dt,
    )
    return output, face_features[1].loss_weight, side_features[1].loss_weight


def run_inference(
    model: RotationAwareFusionModel,
    trial: PosePairTrial,
    skeleton: SkeletonSpec,
    *,
    output_root: str | Path,
    run_id: str,
    window_length: int = 128,
    stride: int = 64,
    provenance: Mapping[str, Any] | None = None,
) -> InferenceResult:
    """Fuse a complete raw trial with 128/64-style overlap-add and save one NPZ."""
    if not run_id:
        raise ValueError("run_id is required for inference")
    canonical = canonicalize_trial(trial, skeleton)
    source = canonical.trial
    frames, joints = source.face.shape[:2]
    weights = overlap_taper(window_length)
    fused_sum = np.zeros((frames, joints, 3), dtype=np.float64)
    base_sum = np.zeros_like(fused_sum)
    point_weight = np.zeros((frames, joints), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in _starts(frames, window_length, stride):
            count = min(window_length, frames - start)
            face = torch.zeros((1, window_length, joints, 3), dtype=torch.float32)
            side = torch.zeros_like(face)
            valid_face = torch.zeros((1, window_length, joints), dtype=torch.bool)
            valid_side = torch.zeros_like(valid_face)
            face[:, :count] = torch.from_numpy(
                np.array(source.face[start : start + count], copy=True)
            )
            side[:, :count] = torch.from_numpy(
                np.array(source.side[start : start + count], copy=True)
            )
            valid_face[:, :count] = torch.from_numpy(
                np.array(source.valid_face[start : start + count], copy=True)
            )
            valid_side[:, :count] = torch.from_numpy(
                np.array(source.valid_side[start : start + count], copy=True)
            )
            dt = torch.zeros((1, window_length), dtype=torch.float32)
            dt[0, :count] = 1.0 / source.fps
            if count > 1:
                dt[0, 1:count] = torch.from_numpy(
                    np.diff(source.timestamps[start : start + count]).astype(np.float32)
                )
            output, _, _ = _forward(
                model, face, side, valid_face, valid_side, skeleton, dt
            )
            valid = output.valid[0, :count].cpu().numpy()
            weight = weights[:count, None] * valid
            fused_sum[start : start + count] += (
                output.fused_kpts[0, :count].cpu().numpy() * weight[..., None]
            )
            base_sum[start : start + count] += (
                output.base_kpts[0, :count].cpu().numpy() * weight[..., None]
            )
            point_weight[start : start + count] += weight
    fused = (fused_sum / np.maximum(point_weight[..., None], 1e-12)).astype(np.float32)
    base = (base_sum / np.maximum(point_weight[..., None], 1e-12)).astype(np.float32)
    fused[point_weight == 0] = 0
    base[point_weight == 0] = 0
    full_face = torch.from_numpy(np.array(source.face, copy=True)).unsqueeze(0)
    full_side = torch.from_numpy(np.array(source.side, copy=True)).unsqueeze(0)
    full_face_valid = torch.from_numpy(
        np.array(source.valid_face, copy=True)
    ).unsqueeze(0)
    full_side_valid = torch.from_numpy(
        np.array(source.valid_side, copy=True)
    ).unsqueeze(0)
    _, quality_face, quality_side = _forward(
        model,
        full_face,
        full_side,
        full_face_valid,
        full_side_valid,
        skeleton,
        _dt(source),
    )
    fused_tensor = torch.from_numpy(fused).unsqueeze(0)
    frame_valid = point_weight.any(axis=1)
    trunk = extract_trunk_features(
        fused_tensor,
        torch.from_numpy(point_weight > 0).unsqueeze(0),
        skeleton,
        _dt(source),
    )
    joint_valid = point_weight > 0
    world = canonical.restore_face(fused)
    world[~joint_valid] = 0
    body = kpts_world_to_body(world)
    body[~joint_valid] = 0
    face_world = np.array(trial.face, copy=True)
    side_world = np.array(trial.side, copy=True)
    face_world[~trial.valid_face] = 0
    side_world[~trial.valid_side] = 0
    arithmetic_valid = source.valid_face | source.valid_side
    arithmetic_weights = arithmetic_valid.astype(np.float32)
    arithmetic = (
        source.face * source.valid_face[..., None]
        + source.side * source.valid_side[..., None]
    ) / np.maximum(arithmetic_weights[..., None], 1.0)
    arithmetic_world = canonical.restore_face(arithmetic)
    arithmetic_world[~arithmetic_valid] = 0
    base_world = canonical.restore_face(base)
    base_world[~joint_valid] = 0
    metadata = {
        "coordinate_system": "face_reference_uncalibrated",
        "run_id": run_id,
        "checkpoint_provenance": dict(provenance or {}),
        "person_id": trial.person_id,
        "trial_id": trial.trial_id,
        "fps": trial.fps,
        "no_pseudo_gt_training": True,
        "window_length": window_length,
        "stride": stride,
        "canonical_source": "complete_trial",
    }
    target = (
        Path(output_root)
        / f"person_{trial.person_id}"
        / trial.trial_id
        / "fused_sequence.npz"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        kpts_world=world,
        kpts_body=body,
        kpts_face_world=face_world,
        kpts_side_world=side_world,
        kpts_arithmetic_world=arithmetic_world,
        kpts_base_world=base_world,
        kpts_fused_canonical=fused,
        kpts_base_canonical=base,
        theta_fused_rad=trunk.angle.squeeze(0).cpu().numpy().astype(np.float32),
        omega_fused_rad_s=trunk.omega.squeeze(0).cpu().numpy().astype(np.float32),
        quality_face=quality_face.squeeze(0).cpu().numpy().astype(np.float32),
        quality_side=quality_side.squeeze(0).cpu().numpy().astype(np.float32),
        frame_valid=frame_valid,
        joint_valid=joint_valid,
        face_map=trial.face_map,
        side_map=trial.side_map,
        kpts_face_canonical=source.face,
        kpts_side_canonical=source.side,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return InferenceResult(target, frames, metadata)
