"""Deterministic face/side fusion building blocks.

Body-frame construction, Sim3 alignment, per-joint weighting and the method
name tables shared by the private experiment matrix and the FreeMan / Unity
benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np


PELVIS_INDICES = (9, 10)


BODY_IDX = {"lhip": 9, "rhip": 10, "lsho": 5, "rsho": 6}


IDX = {
    "lhip": 9,
    "rhip": 10,
    "lsho": 5,
    "rsho": 6,
    "rwrist": 41,
    "rindex_tip": 25,
    "rmiddle_tip": 29,
    "rpinky_tip": 37,
}


STABLE_SIM3_JOINTS = (5, 6, 9, 10, 11, 12, 13, 14, 15, 16)


NO_EXTRINSIC_METHODS = (
    "avg_body_current",
    "avg_world_face_ref",
    "root_face_stable",
    "sim3_face_all",
    "sim3_face_stable",
    "sim3_face_stable_joint_weight",
    "sim3_face_stable_bodypart_weight",
    "sim3_face_stable_smooth_transform",
    "sim3_face_stable_smooth_kpt",
)


EXTRINSIC_METHODS = (
    "extrinsic_r_average",
    "extrinsic_r_quality_average",
)


# Backward-compatible name used by the FreeMan benchmark, whose pose-pair
# schema does not currently carry camera extrinsics.
# External baselines shared by the private, FreeMan and Unity evaluations
# (implemented in classical_baselines.py; dispatched through fuse_baseline).
CLASSICAL_METHODS = (
    "kalman_body_fusion",
    "kalman_rts_body_fusion",
    "jitter_weighted_body_average",
    "butterworth_body_average",
)


EXTERNAL_REFINER_METHODS = ("smoothnet_body_average",)


BASELINE_METHODS = CLASSICAL_METHODS + EXTERNAL_REFINER_METHODS


ALL_METHODS = NO_EXTRINSIC_METHODS + BASELINE_METHODS


AVAILABLE_METHODS = NO_EXTRINSIC_METHODS + BASELINE_METHODS + EXTRINSIC_METHODS


@dataclass(frozen=True)
class Sim3Transform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def build_body_frame(kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lhip = kpts[:, BODY_IDX["lhip"], :]
    rhip = kpts[:, BODY_IDX["rhip"], :]
    pelvis = 0.5 * (lhip + rhip)

    x_axis = _normalize(rhip - lhip)
    lsho = kpts[:, BODY_IDX["lsho"], :]
    rsho = kpts[:, BODY_IDX["rsho"], :]
    shoulder_center = 0.5 * (lsho + rsho)
    y_axis = _normalize(shoulder_center - pelvis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return pelvis, np.stack([x_axis, y_axis, z_axis], axis=-1)


def kpts_world_to_body(kpts_world: np.ndarray) -> np.ndarray:
    pelvis, rotation = build_body_frame(kpts_world)
    centered = kpts_world - pelvis[:, None, :]
    return np.einsum("tij,tbj->tbi", np.transpose(rotation, (0, 2, 1)), centered).astype(np.float32)


def kpts_body_to_world(kpts_body: np.ndarray, pelvis_world: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (np.einsum("tij,tbj->tbi", rotation, kpts_body) + pelvis_world[:, None, :]).astype(np.float32)


def smooth_1d(x: np.ndarray, win: int = 11) -> np.ndarray:
    win = max(3, int(win) | 1)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")


def smooth_sequence(seq: np.ndarray, win: int = 5) -> np.ndarray:
    win = max(3, int(win) | 1)
    pad = win // 2
    padded = np.pad(seq, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(seq, dtype=np.float32)
    for idx in range(len(seq)):
        out[idx] = np.nanmean(padded[idx : idx + win], axis=0)
    return out


def right_hand_point_world(kpts_world: np.ndarray) -> np.ndarray:
    wrist = kpts_world[:, IDX["rwrist"], :]
    index_tip = kpts_world[:, IDX["rindex_tip"], :]
    middle_tip = kpts_world[:, IDX["rmiddle_tip"], :]
    pinky_tip = kpts_world[:, IDX["rpinky_tip"], :]
    return 0.25 * (wrist + index_tip + middle_tip + pinky_tip)


def world_to_body_point(points_world: np.ndarray, pelvis_world: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    centered = points_world - pelvis_world
    return np.einsum("tij,tj->ti", np.transpose(rotation, (0, 2, 1)), centered)


def compute_theta_unwrap_from_world(kpts_world: np.ndarray, idx: Mapping[str, int] | None = None) -> np.ndarray:
    del idx
    pelvis, rotation = build_body_frame(kpts_world)
    hand_body = world_to_body_point(right_hand_point_world(kpts_world), pelvis, rotation)
    theta = np.arctan2(hand_body[:, 2], hand_body[:, 0]).astype(np.float32)
    return np.unwrap(smooth_1d(theta, 11))


def estimate_sim3(source: np.ndarray, target: np.ndarray, joint_indices: Sequence[int]) -> Sim3Transform:
    valid_joints = np.asarray(joint_indices, dtype=np.int32)
    src = np.asarray(source[valid_joints], dtype=np.float64)
    dst = np.asarray(target[valid_joints], dtype=np.float64)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[valid]
    dst = dst[valid]
    if len(src) < 3:
        return Sim3Transform(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float32),
            translation=np.zeros(3, dtype=np.float32),
        )

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    norm_src = np.linalg.norm(src0)
    norm_dst = np.linalg.norm(dst0)
    if norm_src < 1e-8 or norm_dst < 1e-8:
        return Sim3Transform(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float32),
            translation=(dst_mean - src_mean).astype(np.float32),
        )

    src0 /= norm_src
    dst0 /= norm_dst
    u, _, vt = np.linalg.svd(src0.T @ dst0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    scale = norm_dst / norm_src
    translation = dst_mean - scale * (src_mean @ rotation)
    return Sim3Transform(
        scale=float(scale),
        rotation=rotation.astype(np.float32),
        translation=translation.astype(np.float32),
    )


def apply_sim3(points: np.ndarray, transform: Sim3Transform) -> np.ndarray:
    return (transform.scale * (points @ transform.rotation) + transform.translation).astype(np.float32)


def sim3_align_to_reference(
    side: np.ndarray,
    face: np.ndarray,
    joint_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    aligned = np.empty_like(side, dtype=np.float32)
    scales = np.empty((len(side),), dtype=np.float32)
    for idx, (side_frame, face_frame) in enumerate(zip(side, face)):
        transform = estimate_sim3(side_frame, face_frame, joint_indices)
        aligned[idx] = apply_sim3(side_frame, transform)
        scales[idx] = transform.scale
    return aligned, scales


def root_align_to_reference(side: np.ndarray, face: np.ndarray) -> np.ndarray:
    side_root = np.nanmean(side[:, PELVIS_INDICES, :], axis=1, keepdims=True)
    face_root = np.nanmean(face[:, PELVIS_INDICES, :], axis=1, keepdims=True)
    return (side + (face_root - side_root)).astype(np.float32)


def align_side_with_extrinsic_rotation(
    side: np.ndarray,
    face: np.ndarray,
    rotation_face_to_side: np.ndarray,
) -> np.ndarray:
    """Rotate root-relative side poses into face axes and restore face pelvis.

    For the column-vector calibration convention ``X_side = R X_face + t``,
    the equivalent row-vector direction mapping from side to face is
    ``X_side @ R``. Translation is inappropriate for the root-relative SAM3D
    poses and is therefore replaced by pelvis centring/restoration.
    """
    side = np.asarray(side, dtype=np.float32)
    face = np.asarray(face, dtype=np.float32)
    rotation = np.asarray(rotation_face_to_side, dtype=np.float32)
    if side.shape != face.shape or side.ndim != 3 or side.shape[-1] != 3:
        raise ValueError("face and side must have matching shape [T, J, 3]")
    if rotation.shape != (3, 3):
        raise ValueError("rotation_face_to_side must have shape [3, 3]")
    side_root = np.nanmean(side[:, PELVIS_INDICES, :], axis=1, keepdims=True)
    face_root = np.nanmean(face[:, PELVIS_INDICES, :], axis=1, keepdims=True)
    return ((side - side_root) @ rotation + face_root).astype(np.float32)


def fuse_extrinsic_rotation(
    face: np.ndarray,
    side: np.ndarray,
    rotation_face_to_side: np.ndarray,
) -> np.ndarray:
    """Equally average face and externally rotation-aligned side poses."""
    side_aligned = align_side_with_extrinsic_rotation(
        side, face, rotation_face_to_side
    )
    return (0.5 * (face + side_aligned)).astype(np.float32)


def fuse_quality_weighted(
    face: np.ndarray,
    side_aligned: np.ndarray,
    face_quality: np.ndarray,
    side_quality: np.ndarray,
    *,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse two aligned sequences using normalized non-negative frame scores."""
    face_quality = np.asarray(face_quality, dtype=np.float32)
    side_quality = np.asarray(side_quality, dtype=np.float32)
    expected = (face.shape[0],)
    if face.shape != side_aligned.shape:
        raise ValueError("face and side_aligned must have matching shapes")
    if face_quality.shape != expected or side_quality.shape != expected:
        raise ValueError(f"quality arrays must have shape {expected}")
    scores = np.stack(
        [
            np.where(np.isfinite(face_quality), np.maximum(face_quality, 0.0), 0.0),
            np.where(np.isfinite(side_quality), np.maximum(side_quality, 0.0), 0.0),
        ],
        axis=1,
    )
    totals = scores.sum(axis=1, keepdims=True)
    weights = np.divide(
        scores,
        totals,
        out=np.full_like(scores, 0.5),
        where=totals > eps,
    )
    fused = (
        face * weights[:, None, [0]]
        + side_aligned * weights[:, None, [1]]
    )
    return fused.astype(np.float32), weights.astype(np.float32)


def rotation_aware_quality_scores(
    points: np.ndarray,
    skeleton: Any,
) -> np.ndarray:
    """Reuse the fixed quality feature from the rotation-aware mainline."""
    import torch

    from fusion.keypoints.features import compute_quality_features
    from fusion.keypoints.trunk import extract_trunk_features

    tensor = torch.from_numpy(np.asarray(points, dtype=np.float32)).unsqueeze(0)
    valid = torch.isfinite(tensor).all(dim=-1)
    tensor = torch.nan_to_num(tensor)
    trunk = extract_trunk_features(tensor, valid, skeleton, dt=1.0)
    quality = compute_quality_features(tensor, valid, trunk, skeleton)
    return quality.score[0].cpu().numpy().astype(np.float32)


def rotation_aware_quality_scores_by_trial(
    points: np.ndarray,
    skeleton: Any,
    trial_lengths: Sequence[int],
) -> np.ndarray:
    """Compute quality with the same per-cycle temporal scope as the mainline."""
    lengths = [int(length) for length in trial_lengths]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("trial_lengths must contain positive values")
    if sum(lengths) != len(points):
        raise ValueError("trial_lengths must sum to the sequence length")
    chunks: List[np.ndarray] = []
    start = 0
    for length in lengths:
        chunks.append(
            rotation_aware_quality_scores(
                points[start : start + length], skeleton
            )
        )
        start += length
    return np.concatenate(chunks).astype(np.float32)


def fuse_weighted(face: np.ndarray, side_aligned: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (face.shape[1], 2):
        raise ValueError(f"weights must have shape ({face.shape[1]}, 2), got {weights.shape}")
    return (
        face * weights[None, :, [0]]
        + side_aligned * weights[None, :, [1]]
    ).astype(np.float32)


def estimate_joint_weights(
    face_joint_errors: np.ndarray,
    side_joint_errors: np.ndarray,
    *,
    min_weight: float = 0.2,
    eps: float = 1e-8,
) -> np.ndarray:
    face_err = np.asarray(face_joint_errors, dtype=np.float32)
    side_err = np.asarray(side_joint_errors, dtype=np.float32)
    w_face = side_err / (face_err + side_err + eps)
    w_side = face_err / (face_err + side_err + eps)
    weights = np.stack([w_face, w_side], axis=1)
    weights = np.clip(weights, min_weight, 1.0 - min_weight)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def bodypart_weights(n_joints: int) -> np.ndarray:
    """Fixed face/side weights by coarse body parts.

    The weights are intentionally conservative: torso/pelvis stays 50/50, while
    hands and distal limbs lean slightly toward face after side is Sim3-aligned.
    """
    weights = np.full((n_joints, 2), 0.5, dtype=np.float32)
    face_preferred = [
        23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46,
    ]
    side_preferred = [11, 12, 13, 14, 15, 16, 17, 18]
    for joint in face_preferred:
        if joint < n_joints:
            weights[joint] = (0.6, 0.4)
    for joint in side_preferred:
        if joint < n_joints:
            weights[joint] = (0.45, 0.55)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def current_body_average(face: np.ndarray, side: np.ndarray) -> np.ndarray:
    face_body = kpts_world_to_body(face)
    side_body = kpts_world_to_body(side)
    fused_body = 0.5 * (face_body + side_body)
    pelvis_ref, rotation_ref = build_body_frame(face)
    return kpts_body_to_world(fused_body, pelvis_ref, rotation_ref)


def root_normalize(kpts: np.ndarray) -> np.ndarray:
    root = np.nanmean(kpts[:, PELVIS_INDICES, :], axis=1, keepdims=True)
    return kpts - root


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> Sim3Transform:
    """Least-squares (Umeyama) similarity mapping ``src`` onto ``dst``.

    Uses the module's ``points @ rotation`` convention so the result can be fed
    straight to :func:`apply_sim3`.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 3:
        return Sim3Transform(scale=1.0, rotation=np.eye(3), translation=np.zeros(3))
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    variance = float((src_c**2).sum())
    if variance < 1e-12:
        return Sim3Transform(scale=1.0, rotation=np.eye(3), translation=dst_mean - src_mean)
    u, singular, vt = np.linalg.svd(src_c.T @ dst_c)
    sign = float(np.sign(np.linalg.det(u @ vt)))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    scale = float(singular[0] + singular[1] + sign * singular[2]) / variance
    translation = dst_mean - scale * (src_mean @ rotation)
    return Sim3Transform(scale=scale, rotation=rotation, translation=translation)

