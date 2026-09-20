"""Inputs and outputs of the deterministic experiment matrix.

SAM3D per-view keypoints, split-cycle alignment offsets, estimated extrinsics,
the aligned-cycle cache, triangulated reference sequences and the compact
fused-sequence files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from common.paths import SAM3D_RESULTS_ROOT, TRIANGULATED_ROOT
from fusion.baselines.methods import IDX, compute_theta_unwrap_from_world


DEFAULT_SAM3D_ROOT = SAM3D_RESULTS_ROOT


DEFAULT_TRIANGULATED_ROOT = TRIANGULATED_ROOT


DEFAULT_SPLIT_ROOT = Path("local/runs/split_cycle")


DEFAULT_OUT_DIR = Path("local/runs/fuse_experiments")


DEFAULT_EXTRINSICS_PATH = Path(
    "local/runs/analysis/extrinsics/estimated_extrinsics.json"
)


DEFAULT_SKELETON_PATH = Path("src/configs/shared/skeleton_mhr70.yaml")


DEFAULT_ALIGNED_CACHE_ROOT = Path("local/runs/fuse_rotation_aware/cache")


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


def load_extrinsic_rotation(
    path: Path,
    person_id: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and validate the calibrated face-to-side rotation for one person.

    The extrinsics file follows the triangulation convention
    ``X_side = R X_face + t`` for column vectors. Root-relative SAM3D poses only
    use ``R``; the camera translation is deliberately excluded.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing estimated extrinsics: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    persons = document.get("persons")
    if not isinstance(persons, Mapping):
        raise ValueError(f"Estimated extrinsics has no persons mapping: {path}")
    entry = persons.get(str(person_id))
    if not isinstance(entry, Mapping):
        raise KeyError(f"Estimated extrinsics has no entry for person {person_id}: {path}")
    rotation = np.asarray(entry.get("R"), dtype=np.float64)
    is_rotation = (
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
    )
    if not is_rotation:
        raise ValueError(
            f"Estimated extrinsics for person {person_id} does not contain a valid rotation"
        )
    metadata = dict(entry)
    metadata["person_id"] = str(person_id)
    metadata["source_path"] = str(path)
    return rotation.astype(np.float32), metadata


def load_aligned_cycle_cache(
    cache_root: Path,
    person_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load immutable split-cycle face/side arrays from the aligned cache."""
    person_root = cache_root / f"person_{person_id}"
    pointer_path = person_root / "manifest.json"
    if not pointer_path.exists():
        raise FileNotFoundError(f"Missing aligned-cache manifest: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    generation = pointer.get("generation")
    if generation is None:
        data_root = person_root
        manifest = pointer
    else:
        if not isinstance(generation, str) or not generation:
            raise ValueError(f"Invalid aligned-cache generation: {pointer_path}")
        data_root = person_root / ".generations" / generation
        manifest_path = data_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing aligned-cache generation manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation") != generation:
            raise ValueError(
                f"Aligned-cache generation mismatch for person {person_id}"
            )
    trials = manifest.get("trials")
    if (
        not isinstance(trials, list)
        or not trials
        or not all(isinstance(trial, str) and trial for trial in trials)
    ):
        raise ValueError(
            f"Aligned-cache manifest has no valid trials for person {person_id}"
        )

    face_chunks: List[np.ndarray] = []
    side_chunks: List[np.ndarray] = []
    face_map_chunks: List[np.ndarray] = []
    side_map_chunks: List[np.ndarray] = []
    for trial in trials:
        path = data_root / f"{trial}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing aligned-cache trial: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"face", "side", "face_map", "side_map"}
            if not required.issubset(data.files):
                raise ValueError(
                    f"Aligned-cache trial is missing {sorted(required - set(data.files))}: {path}"
                )
            face = np.asarray(data["face"], dtype=np.float32)
            side = np.asarray(data["side"], dtype=np.float32)
            face_map = np.asarray(data["face_map"], dtype=np.int32)
            side_map = np.asarray(data["side_map"], dtype=np.int32)
        if (
            face.shape != side.shape
            or face.ndim != 3
            or face.shape[-1] != 3
            or face_map.shape != (len(face),)
            or side_map.shape != (len(face),)
        ):
            raise ValueError(f"Invalid aligned-cache trial shapes: {path}")
        face_chunks.append(face)
        side_chunks.append(side)
        face_map_chunks.append(face_map)
        side_map_chunks.append(side_map)

    source = manifest.get("source")
    if not isinstance(source, Mapping) or "offset_side_to_face" not in source:
        raise ValueError(
            f"Aligned-cache manifest has no split offset for person {person_id}"
        )
    metadata = {
        "cache_root": str(cache_root),
        "person_id": str(person_id),
        "generation": generation,
        "source_hash": manifest.get("source_hash"),
        "config_hash": manifest.get("config_hash"),
        "offset_side_to_face": int(source["offset_side_to_face"]),
        "trials": list(trials),
        "trial_lengths": [int(len(chunk)) for chunk in face_chunks],
        "sequence_scope": "split_cycles_concatenated",
    }
    return (
        np.concatenate(face_chunks),
        np.concatenate(side_chunks),
        np.concatenate(face_map_chunks),
        np.concatenate(side_map_chunks),
        metadata,
    )


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

