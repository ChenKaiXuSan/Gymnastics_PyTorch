#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run fusion experiment variants and evaluate against triangulated 3D keypoints."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from fuse.save import save_fused_kpts

DEFAULT_SAM3D_ROOT = Path("/home/data/xchen/gymnastics/sam3d_body_results")
DEFAULT_TRIANGULATED_ROOT = Path("/home/data/xchen/gymnastics/sam3d_triangulated/person")
DEFAULT_SPLIT_ROOT = Path("logs/split_cycle")
DEFAULT_OUT_DIR = Path("logs/fuse_experiments")

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
ALL_METHODS = (
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


@dataclass(frozen=True)
class Sim3Transform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class PersonMetric:
    person_id: str
    method: str
    eval_frames: int
    valid_points: int
    mpjpe: float
    median: float
    p95: float
    max_error: float


@dataclass(frozen=True)
class JointMetric:
    person_id: str
    method: str
    joint: int
    valid_points: int
    mpjpe: float
    median: float
    p95: float
    max_error: float


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


def estimate_offset_by_dtw(a: np.ndarray, b: np.ndarray) -> int:
    """Estimate global b-to-a offset from a simple DTW path."""
    a_norm = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
    b_norm = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-8)
    n = len(a_norm)
    m = len(b_norm)
    cost = np.abs(a_norm[:, None] - b_norm[None, :]).astype(np.float32)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        prev = dp[i - 1]
        cur = dp[i]
        for j in range(1, m + 1):
            cur[j] = cost[i - 1, j - 1] + min(prev[j], cur[j - 1], prev[j - 1])

    i, j = n, m
    offsets: List[int] = []
    while i > 0 and j > 0:
        offsets.append((j - 1) - (i - 1))
        choices = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        step = int(np.argmin(choices))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return int(np.median(offsets)) if offsets else 0


def align_to_common_timeline(
    face: np.ndarray,
    side: np.ndarray,
    offset_side_to_face: int,
    *,
    pad_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = int(offset_side_to_face)
    tf = len(face)
    ts = len(side)
    t_min = min(0, -s)
    t_max = max(tf, ts - s)
    t = np.arange(t_min, t_max, dtype=np.int32)
    face_idx = t.copy()
    side_idx = t + s
    face_valid = (face_idx >= 0) & (face_idx < tf)
    side_valid = (side_idx >= 0) & (side_idx < ts)
    face_map = np.where(face_valid, face_idx, -1).astype(np.int32)
    side_map = np.where(side_valid, side_idx, -1).astype(np.int32)
    out_dtype = np.float32 if np.isnan(pad_value) else face.dtype
    face_aligned = np.full((len(t),) + face.shape[1:], pad_value, dtype=out_dtype)
    side_aligned = np.full((len(t),) + side.shape[1:], pad_value, dtype=out_dtype)
    face_aligned[face_valid] = face.astype(out_dtype, copy=False)[face_idx[face_valid]]
    side_aligned[side_valid] = side.astype(out_dtype, copy=False)[side_idx[side_valid]]
    return face_aligned, side_aligned, face_map, side_map


def crop_to_overlap(
    face_aligned: np.ndarray,
    side_aligned: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    valid = (face_map >= 0) & (side_map >= 0)
    if not np.any(valid):
        return face_aligned[:0], side_aligned[:0], face_map[:0], side_map[:0], 0, 0
    t0 = int(np.argmax(valid))
    t1 = int(len(valid) - np.argmax(valid[::-1]))
    return face_aligned[t0:t1], side_aligned[t0:t1], face_map[t0:t1], side_map[t0:t1], t0, t1


def person_id_from_dir(path: Path) -> str:
    return path.name.removeprefix("person_")


def sam3d_person_root(sam3d_root: Path) -> Path:
    if sam3d_root.name == "sam3d_body_results":
        return sam3d_root / "person"
    return sam3d_root / "sam3d_body_results" / "person"


def sam3d_view_dir(sam3d_root: Path, person_id: str, view: str) -> Path:
    return sam3d_person_root(sam3d_root) / person_id / view


def load_sam3d_world_by_frame(sam3d_root: Path, person_id: str, view: str) -> Dict[int, np.ndarray]:
    view_dir = sam3d_view_dir(sam3d_root, person_id, view)
    if not view_dir.exists():
        raise FileNotFoundError(f"Missing SAM3D {view} directory: {view_dir}")
    frames: Dict[int, np.ndarray] = {}
    for frame_path in sorted(view_dir.glob("*_sam3d_body.npz")):
        with np.load(frame_path, allow_pickle=True) as data:
            output = data["output"].item()
        frames[int(output["frame_idx"])] = np.asarray(output["pred_keypoints_3d"], dtype=np.float32)
    if not frames:
        raise FileNotFoundError(f"No SAM3D frames found: {view_dir}")
    return frames


def load_split_alignment_offset(split_root: Path, person_id: str) -> Tuple[int, Dict[str, Any]]:
    record_path = split_root / f"person_{person_id}" / f"alignment_record_{person_id}.json"
    if not record_path.exists():
        raise FileNotFoundError(f"Missing split alignment record: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    metadata = dict(record.get("metadata", {}))
    if "offset_side_to_face" not in metadata:
        raise KeyError(f"Missing offset_side_to_face in split alignment record: {record_path}")
    metadata["alignment_record"] = str(record_path)
    return int(metadata["offset_side_to_face"]), metadata


def build_aligned_timeline(
    face_by_frame: Mapping[int, np.ndarray],
    side_by_frame: Mapping[int, np.ndarray],
    offset_override: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Estimate time offset from SAM3D 3D sequences and return overlap in original frame ids."""
    face_ids = np.asarray(sorted(face_by_frame), dtype=np.int32)
    side_ids = np.asarray(sorted(side_by_frame), dtype=np.int32)
    if len(face_ids) == 0 or len(side_ids) == 0:
        raise ValueError("Cannot align empty face/side sequences")

    face_seq = np.stack([face_by_frame[int(frame_id)] for frame_id in face_ids], axis=0).astype(np.float32)
    side_seq = np.stack([side_by_frame[int(frame_id)] for frame_id in side_ids], axis=0).astype(np.float32)

    if offset_override is None:
        theta_face = compute_theta_unwrap_from_world(face_seq, IDX)
        theta_side = compute_theta_unwrap_from_world(side_seq, IDX)
        offset = estimate_offset_by_dtw(theta_face, theta_side)
    else:
        offset = int(offset_override)

    face_u, side_u, face_pos_u, side_pos_u = align_to_common_timeline(
        face_seq, side_seq, offset, pad_value=np.nan
    )
    face_aligned, side_aligned, face_pos, side_pos, _, _ = crop_to_overlap(
        face_u, side_u, face_pos_u, side_pos_u
    )
    if len(face_aligned) == 0:
        raise ValueError("No overlap segment found after temporal alignment")

    face_map = np.where(face_pos >= 0, face_ids[face_pos], -1).astype(np.int32)
    side_map = np.where(side_pos >= 0, side_ids[side_pos], -1).astype(np.int32)
    return (
        face_aligned.astype(np.float32),
        side_aligned.astype(np.float32),
        face_map,
        side_map,
        int(offset),
    )


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


def build_pair_index(face_map: np.ndarray, side_map: np.ndarray) -> Dict[Tuple[int, int], int]:
    index: Dict[Tuple[int, int], int] = {}
    for timeline_idx, (face_idx, side_idx) in enumerate(zip(face_map, side_map)):
        if int(face_idx) >= 0 and int(side_idx) >= 0:
            index[(int(face_idx), int(side_idx))] = timeline_idx
    return index


def frame_pairs_from_summary(summary: Mapping[str, Any]) -> List[Tuple[int, int]]:
    face_start = int(summary["face_video_frames"]["start"])
    side_start = int(summary["side_video_frames"]["start"])
    return [(face_start + i, side_start + i) for i in range(int(summary["processed_frames"]))]


def load_triangulated_sequence(cycle_root: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    seq_path = cycle_root / "joints_3d_sequence.npz"
    summary_path = cycle_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(seq_path, allow_pickle=True) as data:
        joints = np.asarray(data["joints_3d"], dtype=np.float32)
        if "frame_records" in data and len(data["frame_records"]) == len(joints):
            pairs = [
                (int(record["face_frame_index"]), int(record["side_frame_index"]))
                for record in data["frame_records"]
            ]
        else:
            pairs = frame_pairs_from_summary(summary)
    return joints, pairs


def root_normalize(kpts: np.ndarray) -> np.ndarray:
    root = np.nanmean(kpts[:, PELVIS_INDICES, :], axis=1, keepdims=True)
    return kpts - root


def joint_errors(candidate: np.ndarray, triangulated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    candidate = root_normalize(candidate)
    triangulated = root_normalize(triangulated)
    valid = np.isfinite(candidate).all(axis=-1) & np.isfinite(triangulated).all(axis=-1)
    errors = np.linalg.norm(candidate - triangulated, axis=-1)
    return errors, valid


def summarize_values(values: np.ndarray) -> Dict[str, float | int]:
    if values.size == 0:
        return {
            "valid_points": 0,
            "mpjpe": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max_error": float("nan"),
        }
    return {
        "valid_points": int(values.size),
        "mpjpe": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max_error": float(np.max(values)),
    }


def evaluate_sequence(
    person_id: str,
    method: str,
    fused_world: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
    triangulated_person_root: Path,
) -> Tuple[PersonMetric, List[JointMetric]]:
    pair_index = build_pair_index(face_map, side_map)
    all_values: List[np.ndarray] = []
    joint_values: List[List[np.ndarray]] = []
    eval_frames = 0
    n_joints = fused_world.shape[1]
    joint_values = [[] for _ in range(n_joints)]

    for cycle_root in sorted(triangulated_person_root.glob("cycle_*")):
        triangulated, pairs = load_triangulated_sequence(cycle_root)
        candidate_frames = []
        tri_frames = []
        for tri_frame, pair in zip(triangulated, pairs):
            fused_idx = pair_index.get(pair)
            if fused_idx is None:
                continue
            candidate_frames.append(fused_world[fused_idx])
            tri_frames.append(tri_frame)
        if not candidate_frames:
            continue
        eval_frames += len(candidate_frames)
        errors, valid = joint_errors(np.stack(candidate_frames), np.stack(tri_frames))
        all_values.append(errors[valid])
        for joint_idx in range(n_joints):
            joint_values[joint_idx].append(errors[:, joint_idx][valid[:, joint_idx]])

    values = np.concatenate(all_values) if all_values else np.asarray([], dtype=np.float32)
    stats = summarize_values(values)
    person_metric = PersonMetric(
        person_id=person_id,
        method=method,
        eval_frames=eval_frames,
        valid_points=int(stats["valid_points"]),
        mpjpe=float(stats["mpjpe"]),
        median=float(stats["median"]),
        p95=float(stats["p95"]),
        max_error=float(stats["max_error"]),
    )

    joint_metrics: List[JointMetric] = []
    for joint_idx, chunks in enumerate(joint_values):
        joint_error_values = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        joint_stats = summarize_values(joint_error_values)
        joint_metrics.append(
            JointMetric(
                person_id=person_id,
                method=method,
                joint=joint_idx,
                valid_points=int(joint_stats["valid_points"]),
                mpjpe=float(joint_stats["mpjpe"]),
                median=float(joint_stats["median"]),
                p95=float(joint_stats["p95"]),
                max_error=float(joint_stats["max_error"]),
            )
        )
    return person_metric, joint_metrics


def estimate_weights_from_triangulated(
    face: np.ndarray,
    side_aligned: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
    triangulated_person_root: Path,
) -> np.ndarray:
    pair_index = build_pair_index(face_map, side_map)
    face_errors_by_joint: List[List[np.ndarray]] = [[] for _ in range(face.shape[1])]
    side_errors_by_joint: List[List[np.ndarray]] = [[] for _ in range(face.shape[1])]
    for cycle_root in sorted(triangulated_person_root.glob("cycle_*")):
        triangulated, pairs = load_triangulated_sequence(cycle_root)
        face_frames = []
        side_frames = []
        tri_frames = []
        for tri_frame, pair in zip(triangulated, pairs):
            idx = pair_index.get(pair)
            if idx is None:
                continue
            face_frames.append(face[idx])
            side_frames.append(side_aligned[idx])
            tri_frames.append(tri_frame)
        if not face_frames:
            continue
        tri_seq = np.stack(tri_frames)
        face_errors, face_valid = joint_errors(np.stack(face_frames), tri_seq)
        side_errors, side_valid = joint_errors(np.stack(side_frames), tri_seq)
        for joint_idx in range(face.shape[1]):
            face_errors_by_joint[joint_idx].append(face_errors[:, joint_idx][face_valid[:, joint_idx]])
            side_errors_by_joint[joint_idx].append(side_errors[:, joint_idx][side_valid[:, joint_idx]])

    face_joint_errors = np.empty((face.shape[1],), dtype=np.float32)
    side_joint_errors = np.empty((face.shape[1],), dtype=np.float32)
    for joint_idx in range(face.shape[1]):
        face_values = np.concatenate(face_errors_by_joint[joint_idx]) if face_errors_by_joint[joint_idx] else np.asarray([1.0])
        side_values = np.concatenate(side_errors_by_joint[joint_idx]) if side_errors_by_joint[joint_idx] else np.asarray([1.0])
        face_joint_errors[joint_idx] = float(np.mean(face_values))
        side_joint_errors[joint_idx] = float(np.mean(side_values))
    return estimate_joint_weights(face_joint_errors, side_joint_errors)


def save_compact_sequence(
    out_root: Path,
    person_id: str,
    method: str,
    fused_world: np.ndarray,
    fused_body: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
    extra: Mapping[str, Any],
) -> None:
    person_root = out_root / method / f"person_{person_id}"
    person_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        person_root / "fused_sequence.npz",
        kpts_world=fused_world,
        kpts_body=fused_body,
        face_map=face_map,
        side_map=side_map,
    )
    metadata = {
        "person_id": person_id,
        "method": method,
        "n_frames": int(fused_world.shape[0]),
        "n_joints": int(fused_world.shape[1]),
        **extra,
    }
    (person_root / "config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def process_person(
    person_id: str,
    methods: Sequence[str],
    sam3d_root: Path,
    triangulated_root: Path,
    split_root: Path,
    out_root: Path,
    save_frame_npz: bool,
) -> Tuple[List[PersonMetric], List[JointMetric]]:
    face_by_frame = load_sam3d_world_by_frame(sam3d_root, person_id, "face")
    side_by_frame = load_sam3d_world_by_frame(sam3d_root, person_id, "side")
    split_offset, split_metadata = load_split_alignment_offset(split_root, person_id)
    face, side, face_map, side_map, offset = build_aligned_timeline(
        face_by_frame,
        side_by_frame,
        offset_override=split_offset,
    )
    triangulated_person_root = triangulated_root / f"person_{person_id}"
    has_triangulated = triangulated_person_root.exists() and any(triangulated_person_root.glob("cycle_*"))

    sim3_all = None
    sim3_stable = None
    sim3_stable_scales = None
    sim3_stable_smooth = None
    metrics: List[PersonMetric] = []
    joint_metrics: List[JointMetric] = []

    for method in methods:
        extra: Dict[str, Any] = {
            "time_alignment": "split_alignment_record",
            "offset_side_to_face": int(offset),
            "split_alignment": split_metadata,
            "fusion_method": method,
        }
        if method == "avg_body_current":
            fused_world = current_body_average(face, side)
        elif method == "avg_world_face_ref":
            fused_world = 0.5 * (face + side)
        elif method == "root_face_stable":
            side_aligned = root_align_to_reference(side, face)
            fused_world = 0.5 * (face + side_aligned)
        elif method == "sim3_face_all":
            if sim3_all is None:
                sim3_all, scales = sim3_align_to_reference(side, face, tuple(range(face.shape[1])))
                extra["scale_mean"] = float(np.mean(scales))
            fused_world = 0.5 * (face + sim3_all)
        elif method == "sim3_face_stable":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = 0.5 * (face + sim3_stable)
        elif method == "sim3_face_stable_joint_weight":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            if has_triangulated:
                weights = estimate_weights_from_triangulated(
                    face, sim3_stable, face_map, side_map, triangulated_person_root
                )
                extra["joint_weight_source"] = "triangulated"
            else:
                weights = np.full((face.shape[1], 2), 0.5, dtype=np.float32)
                extra["joint_weight_source"] = "missing_triangulated_equal_fallback"
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["joint_weights"] = weights.tolist()
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = fuse_weighted(face, sim3_stable, weights)
        elif method == "sim3_face_stable_bodypart_weight":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            weights = bodypart_weights(face.shape[1])
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["joint_weights"] = weights.tolist()
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = fuse_weighted(face, sim3_stable, weights)
        elif method == "sim3_face_stable_smooth_transform":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            if sim3_stable_smooth is None:
                sim3_stable_smooth = smooth_sequence(sim3_stable, win=5)
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["smooth_target"] = "side_after_sim3"
            extra["smooth_window"] = 5
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = 0.5 * (face + sim3_stable_smooth)
        elif method == "sim3_face_stable_smooth_kpt":
            if sim3_stable is None:
                sim3_stable, sim3_stable_scales = sim3_align_to_reference(side, face, STABLE_SIM3_JOINTS)
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["smooth_target"] = "fused_world"
            extra["smooth_window"] = 5
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = smooth_sequence(0.5 * (face + sim3_stable), win=5)
        else:
            raise ValueError(f"Unsupported method: {method}")

        fused_world = fused_world.astype(np.float32)
        fused_body = kpts_world_to_body(fused_world)
        save_compact_sequence(out_root, person_id, method, fused_world, fused_body, face_map, side_map, extra)
        if save_frame_npz:
            save_fused_kpts(
                fused_world=fused_world,
                fused_body=fused_body,
                face_map=face_map,
                side_map=side_map,
                person_id=person_id,
                out_root=out_root / method / f"person_{person_id}" / "frames_format",
                fps=60.0,
            )

        if has_triangulated:
            person_metric, per_joint = evaluate_sequence(
                person_id=person_id,
                method=method,
                fused_world=fused_world,
                face_map=face_map,
                side_map=side_map,
                triangulated_person_root=triangulated_person_root,
            )
            metrics.append(person_metric)
            joint_metrics.extend(per_joint)
            print(f"[metric] person_{person_id} {method}: mpjpe={person_metric.mpjpe:.6g}")
        else:
            print(f"[metric] person_{person_id} {method}: skipped missing triangulated GT")

    return metrics, joint_metrics


def iter_person_ids(sam3d_root: Path, wanted: Sequence[str] | None) -> Iterable[str]:
    wanted_set = {str(item) for item in wanted} if wanted else None
    person_dirs = [
        person_dir
        for person_dir in sam3d_person_root(sam3d_root).iterdir()
        if person_dir.is_dir() and person_dir.name.isdigit()
    ]
    for person_dir in sorted(person_dirs, key=lambda p: int(p.name)):
        person_id = person_dir.name
        if wanted_set is None or person_id in wanted_set:
            yield person_id


def write_csv(path: Path, rows: Sequence[Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face/side/fuse experiment matrix.")
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument("--triangulated-root", type=Path, default=DEFAULT_TRIANGULATED_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--person", nargs="*", default=None, help="Optional person ids, e.g. 27 29")
    parser.add_argument("--methods", nargs="*", default=list(ALL_METHODS), choices=ALL_METHODS)
    parser.add_argument("--save-frame-npz", action="store_true", help="Also save old per-frame npz format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_person_metrics: List[PersonMetric] = []
    all_joint_metrics: List[JointMetric] = []
    config = {
        "sam3d_root": str(args.sam3d_root),
        "triangulated_root": str(args.triangulated_root),
        "split_root": str(args.split_root),
        "methods": list(args.methods),
        "stable_sim3_joints": list(STABLE_SIM3_JOINTS),
        "save_frame_npz": bool(args.save_frame_npz),
    }
    (args.out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for person_id in iter_person_ids(args.sam3d_root, args.person):
        print(f"[person] {person_id}")
        person_metrics, joint_metrics = process_person(
            person_id=person_id,
            methods=args.methods,
            sam3d_root=args.sam3d_root,
            triangulated_root=args.triangulated_root,
            split_root=args.split_root,
            out_root=args.out_dir,
            save_frame_npz=args.save_frame_npz,
        )
        all_person_metrics.extend(person_metrics)
        all_joint_metrics.extend(joint_metrics)

    write_csv(
        args.out_dir / "metrics_by_person.csv",
        all_person_metrics,
        ("person_id", "method", "eval_frames", "valid_points", "mpjpe", "median", "p95", "max_error"),
    )
    write_csv(
        args.out_dir / "metrics_by_joint.csv",
        all_joint_metrics,
        ("person_id", "method", "joint", "valid_points", "mpjpe", "median", "p95", "max_error"),
    )

    for method in args.methods:
        values = [row.mpjpe for row in all_person_metrics if row.method == method]
        if values:
            print(f"[summary] {method}: mean_person_mpjpe={np.nanmean(values):.6g}")
        else:
            print(f"[summary] {method}: no triangulated GT metrics")
    print(f"[save] {args.out_dir / 'metrics_by_person.csv'}")
    print(f"[save] {args.out_dir / 'metrics_by_joint.csv'}")


if __name__ == "__main__":
    main()
