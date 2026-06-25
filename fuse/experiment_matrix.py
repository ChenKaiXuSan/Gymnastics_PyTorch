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
DEFAULT_FUSED_ROOT = Path("/home/data/xchen/gymnastics/fused_kpt")
DEFAULT_TRIANGULATED_ROOT = Path("/home/data/xchen/gymnastics/sam3d_triangulated/person")
DEFAULT_OUT_DIR = Path("logs/fuse_experiments")

PELVIS_INDICES = (9, 10)
BODY_IDX = {"lhip": 9, "rhip": 10, "lsho": 5, "rsho": 6}
STABLE_SIM3_JOINTS = (5, 6, 9, 10, 11, 12, 13, 14, 15, 16)
ALL_METHODS = (
    "avg_body_current",
    "avg_world_face_ref",
    "root_face_stable",
    "sim3_face_all",
    "sim3_face_stable",
    "sim3_face_stable_joint_weight",
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


def person_id_from_dir(path: Path) -> str:
    return path.name.removeprefix("person_")


def load_fused_metadata(fused_root: Path, person_id: str) -> Dict[str, Any]:
    metadata_path = fused_root / f"person_{person_id}" / f"fused_kpts_metadata_{person_id}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing fused metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_existing_fused_world(fused_root: Path, person_id: str, metadata: Mapping[str, Any]) -> np.ndarray:
    person_root = fused_root / f"person_{person_id}"
    frames_dir = person_root / str(metadata["frames_dir"])
    frames = []
    for idx in range(int(metadata["n_frames"])):
        frame_path = frames_dir / f"frame_{idx:06d}.npz"
        with np.load(frame_path) as data:
            frames.append(np.asarray(data["kpts_world"], dtype=np.float32))
    return np.stack(frames, axis=0)


def sam3d_view_dir(sam3d_root: Path, person_id: str, view: str) -> Path:
    if sam3d_root.name == "sam3d_body_results":
        return sam3d_root / "person" / person_id / view
    return sam3d_root / "sam3d_body_results" / "person" / person_id / view


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


def align_sam3d_to_fused_timeline(
    face_by_frame: Mapping[int, np.ndarray],
    side_by_frame: Mapping[int, np.ndarray],
    face_map: np.ndarray,
    side_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_seq = []
    side_seq = []
    keep = []
    for idx, (face_idx, side_idx) in enumerate(zip(face_map, side_map)):
        face = face_by_frame.get(int(face_idx))
        side = side_by_frame.get(int(side_idx))
        if face is None or side is None:
            keep.append(False)
            continue
        face_seq.append(face)
        side_seq.append(side)
        keep.append(True)
    if not face_seq:
        raise ValueError("No overlapping face/side SAM3D frames matched fused metadata")
    return (
        np.stack(face_seq, axis=0).astype(np.float32),
        np.stack(side_seq, axis=0).astype(np.float32),
        np.asarray(keep, dtype=bool),
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


def current_body_average(face: np.ndarray, side: np.ndarray) -> np.ndarray:
    face_body = kpts_world_to_body(face)
    side_body = kpts_world_to_body(side)
    fused_body = 0.5 * (face_body + side_body)
    pelvis_ref, rotation_ref = build_body_frame(face)
    return kpts_body_to_world(fused_body, pelvis_ref, rotation_ref)


def build_pair_index(face_map: np.ndarray, side_map: np.ndarray, keep_mask: np.ndarray | None = None) -> Dict[Tuple[int, int], int]:
    index: Dict[Tuple[int, int], int] = {}
    timeline_idx = 0
    for original_idx, (face_idx, side_idx) in enumerate(zip(face_map, side_map)):
        if keep_mask is not None and not bool(keep_mask[original_idx]):
            continue
        if int(face_idx) >= 0 and int(side_idx) >= 0:
            index[(int(face_idx), int(side_idx))] = timeline_idx
        timeline_idx += 1 if keep_mask is None or bool(keep_mask[original_idx]) else 0
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
    keep_mask: np.ndarray,
    triangulated_person_root: Path,
) -> Tuple[PersonMetric, List[JointMetric]]:
    pair_index = build_pair_index(face_map, side_map, keep_mask)
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
    keep_mask: np.ndarray,
    triangulated_person_root: Path,
) -> np.ndarray:
    pair_index = build_pair_index(face_map, side_map, keep_mask)
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
    fused_root: Path,
    triangulated_root: Path,
    out_root: Path,
    save_frame_npz: bool,
) -> Tuple[List[PersonMetric], List[JointMetric]]:
    metadata = load_fused_metadata(fused_root, person_id)
    face_map_full = np.asarray(metadata["face_map"], dtype=np.int32)
    side_map_full = np.asarray(metadata["side_map"], dtype=np.int32)
    face_by_frame = load_sam3d_world_by_frame(sam3d_root, person_id, "face")
    side_by_frame = load_sam3d_world_by_frame(sam3d_root, person_id, "side")
    face, side, keep_mask = align_sam3d_to_fused_timeline(
        face_by_frame, side_by_frame, face_map_full, side_map_full
    )
    face_map = face_map_full[keep_mask]
    side_map = side_map_full[keep_mask]
    triangulated_person_root = triangulated_root / f"person_{person_id}"

    existing_fused = None
    sim3_all = None
    sim3_stable = None
    sim3_stable_scales = None
    metrics: List[PersonMetric] = []
    joint_metrics: List[JointMetric] = []

    for method in methods:
        extra: Dict[str, Any] = {"face_map_source": str(fused_root), "fusion_method": method}
        if method == "avg_body_current":
            if existing_fused is None:
                existing_fused = load_existing_fused_world(fused_root, person_id, metadata)[keep_mask]
            fused_world = existing_fused
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
            weights = estimate_weights_from_triangulated(
                face, sim3_stable, face_map_full, side_map_full, keep_mask, triangulated_person_root
            )
            extra["sim3_joints"] = list(STABLE_SIM3_JOINTS)
            extra["joint_weights"] = weights.tolist()
            extra["scale_mean"] = float(np.mean(sim3_stable_scales))
            fused_world = fuse_weighted(face, sim3_stable, weights)
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

        person_metric, per_joint = evaluate_sequence(
            person_id=person_id,
            method=method,
            fused_world=fused_world,
            face_map=face_map_full,
            side_map=side_map_full,
            keep_mask=keep_mask,
            triangulated_person_root=triangulated_person_root,
        )
        metrics.append(person_metric)
        joint_metrics.extend(per_joint)
        print(f"[metric] person_{person_id} {method}: mpjpe={person_metric.mpjpe:.6g}")

    return metrics, joint_metrics


def iter_person_ids(triangulated_root: Path, wanted: Sequence[str] | None) -> Iterable[str]:
    wanted_set = {str(item) for item in wanted} if wanted else None
    for person_dir in sorted(triangulated_root.glob("person_*"), key=lambda p: int(person_id_from_dir(p))):
        person_id = person_id_from_dir(person_dir)
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
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--triangulated-root", type=Path, default=DEFAULT_TRIANGULATED_ROOT)
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
        "fused_root": str(args.fused_root),
        "triangulated_root": str(args.triangulated_root),
        "methods": list(args.methods),
        "stable_sim3_joints": list(STABLE_SIM3_JOINTS),
        "save_frame_npz": bool(args.save_frame_npz),
    }
    (args.out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for person_id in iter_person_ids(args.triangulated_root, args.person):
        print(f"[person] {person_id}")
        person_metrics, joint_metrics = process_person(
            person_id=person_id,
            methods=args.methods,
            sam3d_root=args.sam3d_root,
            fused_root=args.fused_root,
            triangulated_root=args.triangulated_root,
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
        print(f"[summary] {method}: mean_person_mpjpe={np.nanmean(values):.6g}")
    print(f"[save] {args.out_dir / 'metrics_by_person.csv'}")
    print(f"[save] {args.out_dir / 'metrics_by_joint.csv'}")


if __name__ == "__main__":
    main()
