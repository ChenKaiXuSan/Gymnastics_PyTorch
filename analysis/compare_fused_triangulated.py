#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare face/side/fused SAM3D-Body 3D keypoints with triangulated 3D keypoints."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

DEFAULT_SAM3D_ROOT = Path("/home/data/xchen/gymnastics/sam3d_body_results")
DEFAULT_FUSED_ROOT = Path("/home/data/xchen/gymnastics/fused_kpt")
DEFAULT_TRIANGULATED_ROOT = Path("/home/data/xchen/gymnastics/sam3d_triangulated/person")
DEFAULT_OUT_DIR = Path("logs/analysis/source_vs_triangulated_by_person")
PELVIS_INDICES = (9, 10)
SOURCES = ("face", "side", "fuse")


@dataclass(frozen=True)
class PersonSourceComparison:
    person_id: str
    source: str
    matched_frames: int
    missing_frames: int
    valid_points: int
    mpjpe: float
    median: float
    p95: float
    max_error: float
    scale: float


def person_id_from_dir(path: Path) -> str:
    return path.name.removeprefix("person_")


def build_fused_pair_index(face_map: np.ndarray, side_map: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Map original (face_frame, side_frame) pairs to fused timeline indices."""
    index: Dict[Tuple[int, int], int] = {}
    for fused_idx, (face_idx, side_idx) in enumerate(zip(face_map, side_map)):
        face_i = int(face_idx)
        side_i = int(side_idx)
        if face_i >= 0 and side_i >= 0:
            index[(face_i, side_i)] = int(fused_idx)
    return index


def frame_pairs_from_summary(summary: Mapping[str, Any]) -> List[Tuple[int, int]]:
    """Build frame pairs from cycle summary ranges."""
    face_range = summary["face_video_frames"]
    side_range = summary["side_video_frames"]
    face_start = int(face_range["start"])
    side_start = int(side_range["start"])
    processed = int(summary["processed_frames"])
    return [(face_start + i, side_start + i) for i in range(processed)]


def load_fused_metadata(person_root: Path, person_id: str) -> Dict[str, Any]:
    metadata_path = person_root / f"fused_kpts_metadata_{person_id}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing fused metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_fused_world_sequence(person_root: Path, metadata: Mapping[str, Any]) -> np.ndarray:
    frames_dir = person_root / str(metadata["frames_dir"])
    n_frames = int(metadata["n_frames"])
    frames: List[np.ndarray] = []
    for idx in range(n_frames):
        frame_path = frames_dir / f"frame_{idx:06d}.npz"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing fused frame: {frame_path}")
        with np.load(frame_path) as data:
            frames.append(np.asarray(data["kpts_world"], dtype=np.float32))
    return np.stack(frames, axis=0)


def sam3d_view_dir(sam3d_root: Path, person_id: str, view: str) -> Path:
    if sam3d_root.name == "sam3d_body_results":
        return sam3d_root / "person" / person_id / view
    return sam3d_root / "sam3d_body_results" / "person" / person_id / view


def load_sam3d_world_by_frame(sam3d_root: Path, person_id: str, view: str) -> Dict[int, np.ndarray]:
    """Load SAM3D-Body pred_keypoints_3d keyed by original frame_idx."""
    view_dir = sam3d_view_dir(sam3d_root, person_id, view)
    if not view_dir.exists():
        raise FileNotFoundError(f"Missing SAM3D {view} directory: {view_dir}")

    frames: Dict[int, np.ndarray] = {}
    for frame_path in sorted(view_dir.glob("*_sam3d_body.npz")):
        with np.load(frame_path, allow_pickle=True) as data:
            output = data["output"].item()
        frame_idx = int(output["frame_idx"])
        frames[frame_idx] = np.asarray(output["pred_keypoints_3d"], dtype=np.float32)
    if not frames:
        raise FileNotFoundError(f"No SAM3D npz files found: {view_dir}")
    return frames


def load_triangulated_sequence(cycle_root: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    seq_path = cycle_root / "joints_3d_sequence.npz"
    summary_path = cycle_root / "summary.json"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing triangulated sequence: {seq_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing cycle summary: {summary_path}")

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

    if len(pairs) != len(joints):
        raise ValueError(
            f"Frame pair count ({len(pairs)}) does not match joints length ({len(joints)}): {cycle_root}"
        )
    return joints, pairs


def pelvis_center(kpts: np.ndarray) -> np.ndarray:
    return np.nanmean(kpts[:, PELVIS_INDICES, :], axis=1, keepdims=True)


def align_sequences(
    candidate: np.ndarray,
    triangulated: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Align sequences before error computation."""
    if mode == "none":
        return candidate, triangulated, 1.0
    if mode == "root":
        return candidate - pelvis_center(candidate), triangulated - pelvis_center(triangulated), 1.0
    if mode == "procrustes":
        aligned = np.empty_like(candidate, dtype=np.float32)
        scales: List[float] = []
        for idx, (src, dst) in enumerate(zip(candidate, triangulated)):
            aligned[idx], scale = procrustes_align(src, dst)
            scales.append(scale)
        return aligned, triangulated, float(np.nanmean(scales)) if scales else 1.0
    raise ValueError(f"Unsupported alignment mode: {mode}")


def procrustes_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Align source joints to target with similarity Procrustes transform."""
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    if valid.sum() < 3:
        return source.astype(np.float32), 1.0

    src = source[valid].astype(np.float64)
    dst = target[valid].astype(np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean

    norm_src = np.linalg.norm(src0)
    norm_dst = np.linalg.norm(dst0)
    if norm_src < 1e-8 or norm_dst < 1e-8:
        return source.astype(np.float32), 1.0

    src0 /= norm_src
    dst0 /= norm_dst
    u, _, vt = np.linalg.svd(src0.T @ dst0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt

    scale = norm_dst / norm_src
    aligned = (source - src_mean) @ rotation * scale + dst_mean
    return aligned.astype(np.float32), float(scale)


def compute_joint_errors(candidate: np.ndarray, triangulated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-joint Euclidean errors and a finite-value mask."""
    valid = np.isfinite(candidate).all(axis=-1) & np.isfinite(triangulated).all(axis=-1)
    errors = np.linalg.norm(candidate - triangulated, axis=-1)
    return errors, valid


def finite_error_values(
    candidate: np.ndarray,
    triangulated: np.ndarray,
    align: str,
) -> Tuple[np.ndarray, float]:
    candidate, triangulated, scale = align_sequences(candidate, triangulated, align)
    errors, valid = compute_joint_errors(candidate, triangulated)
    return errors[valid], scale


def summarize_source_errors(
    person_id: str,
    source_errors: Mapping[str, Sequence[np.ndarray]],
    matched_frames: Mapping[str, int],
    missing_frames: Mapping[str, int],
    scales: Mapping[str, Sequence[float]],
) -> List[PersonSourceComparison]:
    rows: List[PersonSourceComparison] = []
    for source in SOURCES:
        chunks = list(source_errors.get(source, []))
        values = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        source_scales = list(scales.get(source, []))
        if values.size:
            stats = {
                "valid_points": int(values.size),
                "mpjpe": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max_error": float(np.max(values)),
            }
        else:
            stats = {
                "valid_points": 0,
                "mpjpe": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "max_error": float("nan"),
            }
        rows.append(
            PersonSourceComparison(
                person_id=person_id,
                source=source,
                matched_frames=int(matched_frames.get(source, 0)),
                missing_frames=int(missing_frames.get(source, 0)),
                valid_points=int(stats["valid_points"]),
                mpjpe=float(stats["mpjpe"]),
                median=float(stats["median"]),
                p95=float(stats["p95"]),
                max_error=float(stats["max_error"]),
                scale=float(np.mean(source_scales)) if source_scales else 1.0,
            )
        )
    return rows


def iter_person_dirs(root: Path, wanted: Sequence[str] | None) -> Iterable[Path]:
    wanted_set = {str(p) for p in wanted} if wanted else None
    for person_dir in sorted(root.glob("person_*"), key=lambda p: int(person_id_from_dir(p))):
        person_id = person_id_from_dir(person_dir)
        if wanted_set is None or person_id in wanted_set:
            yield person_dir


def compare_person(
    person_id: str,
    tri_person_root: Path,
    sam3d_root: Path,
    fused_root: Path,
    align: str,
) -> List[PersonSourceComparison]:
    fused_person_root = fused_root / f"person_{person_id}"
    metadata = load_fused_metadata(fused_person_root, person_id)
    fused_world = load_fused_world_sequence(fused_person_root, metadata)
    fused_pair_index = build_fused_pair_index(
        np.asarray(metadata["face_map"], dtype=np.int32),
        np.asarray(metadata["side_map"], dtype=np.int32),
    )

    face_world = load_sam3d_world_by_frame(sam3d_root, person_id, "face")
    side_world = load_sam3d_world_by_frame(sam3d_root, person_id, "side")

    source_errors: Dict[str, List[np.ndarray]] = {source: [] for source in SOURCES}
    matched_frames = {source: 0 for source in SOURCES}
    missing_frames = {source: 0 for source in SOURCES}
    scales: Dict[str, List[float]] = {source: [] for source in SOURCES}

    for cycle_root in sorted(tri_person_root.glob("cycle_*")):
        triangulated, pairs = load_triangulated_sequence(cycle_root)
        for tri_frame, (face_idx, side_idx) in zip(triangulated, pairs):
            candidates = {
                "face": face_world.get(face_idx),
                "side": side_world.get(side_idx),
            }
            fused_idx = fused_pair_index.get((face_idx, side_idx))
            candidates["fuse"] = None if fused_idx is None else fused_world[fused_idx]

            for source, candidate in candidates.items():
                if candidate is None:
                    missing_frames[source] += 1
                    continue
                candidate_seq = np.asarray(candidate, dtype=np.float32)[None, ...]
                triangulated_seq = np.asarray(tri_frame, dtype=np.float32)[None, ...]
                values, scale = finite_error_values(candidate_seq, triangulated_seq, align)
                source_errors[source].append(values)
                matched_frames[source] += 1
                scales[source].append(scale)

    return summarize_source_errors(
        person_id=person_id,
        source_errors=source_errors,
        matched_frames=matched_frames,
        missing_frames=missing_frames,
        scales=scales,
    )


def compare_all_by_person(
    sam3d_root: Path,
    fused_root: Path,
    triangulated_root: Path,
    person_ids: Sequence[str] | None,
    align: str,
    num_workers: int,
) -> Tuple[List[PersonSourceComparison], List[str]]:
    rows: List[PersonSourceComparison] = []
    warnings: List[str] = []

    person_dirs = list(iter_person_dirs(triangulated_root, person_ids))
    if num_workers <= 1:
        for tri_person_root in person_dirs:
            person_id = person_id_from_dir(tri_person_root)
            print(f"[person] {person_id}")
            try:
                rows.extend(
                    compare_person(
                        person_id=person_id,
                        tri_person_root=tri_person_root,
                        sam3d_root=sam3d_root,
                        fused_root=fused_root,
                        align=align,
                    )
                )
            except Exception as exc:
                warnings.append(f"person_{person_id}: skipped ({exc})")
        return rows, warnings

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                compare_person,
                person_id_from_dir(tri_person_root),
                tri_person_root,
                sam3d_root,
                fused_root,
                align,
            ): person_id_from_dir(tri_person_root)
            for tri_person_root in person_dirs
        }
        for future in as_completed(futures):
            person_id = futures[future]
            try:
                person_rows = future.result()
                rows.extend(person_rows)
                print(f"[person] {person_id} done")
            except Exception as exc:
                warnings.append(f"person_{person_id}: skipped ({exc})")

    rows.sort(
        key=lambda row: (
            int(row.person_id),
            SOURCES.index(row.source) if row.source in SOURCES else len(SOURCES),
        )
    )
    return rows, warnings


def overall_summary(rows: Sequence[PersonSourceComparison]) -> Dict[str, Dict[str, float | int]]:
    summary: Dict[str, Dict[str, float | int]] = {}
    for source in SOURCES:
        source_rows = [row for row in rows if row.source == source]
        if not source_rows:
            summary[source] = {"persons": 0, "matched_frames": 0, "valid_points": 0}
            continue
        weights = np.asarray([row.valid_points for row in source_rows], dtype=np.float64)
        mpjpe = np.asarray([row.mpjpe for row in source_rows], dtype=np.float64)
        valid = weights > 0
        summary[source] = {
            "persons": int(len(source_rows)),
            "matched_frames": int(sum(row.matched_frames for row in source_rows)),
            "missing_frames": int(sum(row.missing_frames for row in source_rows)),
            "valid_points": int(sum(row.valid_points for row in source_rows)),
            "weighted_mpjpe": float(np.average(mpjpe[valid], weights=weights[valid]))
            if np.any(valid)
            else float("nan"),
            "mean_person_mpjpe": float(np.nanmean(mpjpe)) if mpjpe.size else float("nan"),
            "median_person_mpjpe": float(np.nanmedian(mpjpe)) if mpjpe.size else float("nan"),
        }
    return summary


def write_outputs(
    rows: Sequence[PersonSourceComparison],
    warnings: Sequence[str],
    out_dir: Path,
    align: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"source_vs_triangulated_by_person_{align}.csv"
    json_path = out_dir / f"source_vs_triangulated_by_person_{align}.json"

    fieldnames = [
        "person_id",
        "source",
        "matched_frames",
        "missing_frames",
        "valid_points",
        "mpjpe",
        "median",
        "p95",
        "max_error",
        "scale",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    payload = {
        "alignment": align,
        "overall": overall_summary(rows),
        "by_person": [row.__dict__ for row in rows],
        "warnings": list(warnings),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {csv_path}")
    print(f"[save] {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare face/side/fused SAM3D 3D keypoints with triangulated 3D keypoints by person."
    )
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--triangulated-root", type=Path, default=DEFAULT_TRIANGULATED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--person", nargs="*", default=None, help="Optional person ids, e.g. 27 29")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of persons to process in parallel.")
    parser.add_argument(
        "--align",
        choices=("none", "root", "procrustes"),
        default="root",
        help="Alignment before computing Euclidean joint errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, warnings = compare_all_by_person(
        sam3d_root=args.sam3d_root,
        fused_root=args.fused_root,
        triangulated_root=args.triangulated_root,
        person_ids=args.person,
        align=args.align,
        num_workers=max(1, int(args.num_workers)),
    )
    write_outputs(rows, warnings, args.out_dir, args.align)
    summary = overall_summary(rows)
    for source in SOURCES:
        item = summary.get(source, {})
        print(
            "[summary] "
            f"source={source} "
            f"persons={item.get('persons', 0)} "
            f"matched_frames={item.get('matched_frames', 0)} "
            f"valid_points={item.get('valid_points', 0)} "
            f"weighted_mpjpe={item.get('weighted_mpjpe', float('nan')):.6g}"
        )
    if warnings:
        print(f"[warn] {len(warnings)} skipped items; see JSON for details")


if __name__ == "__main__":
    main()
