"""Split-cycle adapter and compact cache for rotation-aware fusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fuse.experiment_matrix import (
    build_aligned_timeline,
    load_sam3d_world_by_frame,
    load_split_alignment_offset,
)

from .config import SkeletonSpec
from .schema import PosePairTrial, valid_from_points

DEFAULT_FPS = 60.0


def _load_split_record(split_root: Path, person_id: str) -> tuple[Path, dict[str, Any]]:
    path = split_root / f"person_{person_id}" / f"alignment_record_{person_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split alignment record for person {person_id}: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("cycles"), list):
        raise ValueError(f"Split alignment record has no cycles list for person {person_id}: {path}")
    if not record["cycles"]:
        raise ValueError(f"Split alignment record has no cycles for person {person_id}: {path}")
    return path, record


def _cycle_mask(frame_map: np.ndarray, bounds: Mapping[str, Any], label: str) -> np.ndarray:
    try:
        start, end = int(bounds["start"]), int(bounds["end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label} cycle bounds") from error
    if end <= start:
        raise ValueError(f"Invalid {label} cycle bounds: [{start}, {end})")
    return (frame_map >= start) & (frame_map < end)


def load_person_trials(
    person_id: str,
    sam3d_root: str | Path,
    split_root: str | Path,
    skeleton: SkeletonSpec,
) -> list[PosePairTrial]:
    """Load split-defined, overlap-only trials with the recorded time offset."""
    sam3d_path, split_path = Path(sam3d_root), Path(split_root)
    record_path, record = _load_split_record(split_path, person_id)
    offset, alignment_metadata = load_split_alignment_offset(split_path, person_id)
    face_by_frame = load_sam3d_world_by_frame(sam3d_path, person_id, "face")
    side_by_frame = load_sam3d_world_by_frame(sam3d_path, person_id, "side")
    face, side, face_map, side_map, used_offset = build_aligned_timeline(
        face_by_frame, side_by_frame, offset_override=offset
    )
    if used_offset != offset:
        raise RuntimeError(f"Split offset changed while aligning person {person_id}: {record_path}")
    if face.shape[1] != len(skeleton.joint_names):
        raise ValueError(
            f"Person {person_id} has {face.shape[1]} joints but skeleton {skeleton.name} expects "
            f"{len(skeleton.joint_names)}"
        )

    metadata = record.get("metadata", {})
    fps = float(metadata.get("fps", DEFAULT_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps in split record for person {person_id}: {record_path}")
    trials: list[PosePairTrial] = []
    for position, cycle in enumerate(record["cycles"]):
        if not isinstance(cycle, dict):
            raise ValueError(f"Invalid cycle entry {position} for person {person_id}: {record_path}")
        mask = _cycle_mask(face_map, cycle.get("face_video_frames", {}), "face")
        mask &= _cycle_mask(side_map, cycle.get("side_video_frames", {}), "side")
        if not np.any(mask):
            raise ValueError(f"Cycle {position} has no aligned frames for person {person_id}: {record_path}")
        cycle_index = int(cycle.get("cycle_index", position))
        cycle_face, cycle_side = face[mask], side[mask]
        cycle_face_map, cycle_side_map = face_map[mask], side_map[mask]
        timestamps = np.arange(len(cycle_face), dtype=np.float64) / fps
        source_metadata = {
            "alignment_record": str(record_path),
            "offset_side_to_face": offset,
            "fps": fps,
            "person_id": str(person_id),
            "cycle_index": cycle_index,
            "face_video_frames": dict(cycle["face_video_frames"]),
            "side_video_frames": dict(cycle["side_video_frames"]),
        }
        trials.append(
            PosePairTrial(
                face=cycle_face,
                side=cycle_side,
                valid_face=valid_from_points(cycle_face),
                valid_side=valid_from_points(cycle_side),
                timestamps=timestamps,
                face_map=cycle_face_map,
                side_map=cycle_side_map,
                joint_names=skeleton.joint_names,
                person_id=str(person_id),
                trial_id=f"cycle_{cycle_index:03d}",
                fps=fps,
                source_metadata=source_metadata,
            )
        )
    return trials


def _metadata_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    return value


def _validate_metadata(metadata: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    normalized = _plain_metadata(metadata)
    assert isinstance(normalized, dict)
    try:
        json.dumps(normalized, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be JSON serializable") from error
    return normalized


def write_person_cache(
    trials: Sequence[PosePairTrial],
    cache_root: str | Path = Path("logs/fuse_rotation_aware/cache"),
    *,
    source_metadata: Mapping[str, Any],
    config_metadata: Mapping[str, Any],
) -> Path:
    """Write compact per-cycle arrays and traceable source/config metadata."""
    if not trials:
        raise ValueError("Cannot cache an empty trial list")
    person_id = trials[0].person_id
    if any(trial.person_id != person_id for trial in trials):
        raise ValueError("All cached trials must belong to one person")
    person_cache = Path(cache_root) / f"person_{person_id}"
    person_cache.mkdir(parents=True, exist_ok=True)
    source = _validate_metadata(source_metadata, "source_metadata")
    config = _validate_metadata(config_metadata, "config_metadata")
    required_source_fields = {"alignment_record", "offset_side_to_face", "fps", "person_id"}
    missing_source_fields = sorted(required_source_fields - set(source))
    if missing_source_fields:
        raise ValueError(f"source_metadata is missing required fields: {missing_source_fields}")
    source["trial_sources"] = {
        trial.trial_id: _plain_metadata(trial.source_metadata) for trial in trials
    }
    for trial in trials:
        np.savez_compressed(
            person_cache / f"{trial.trial_id}.npz",
            face=trial.face,
            side=trial.side,
            valid_face=trial.valid_face,
            valid_side=trial.valid_side,
            timestamps=trial.timestamps,
            face_map=trial.face_map,
            side_map=trial.side_map,
            joint_names=np.asarray(trial.joint_names),
            person_id=np.asarray(trial.person_id),
            trial_id=np.asarray(trial.trial_id),
            fps=np.asarray(trial.fps, dtype=np.float64),
        )
    manifest = {
        "person_id": person_id,
        "trials": [trial.trial_id for trial in trials],
        "source": source,
        "config": config,
        "source_hash": _metadata_hash(source),
        "config_hash": _metadata_hash(config),
    }
    (person_cache / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return person_cache


def load_cached_trial(cache_path: str | Path, trial_id: str | None = None) -> tuple[PosePairTrial, dict[str, Any]]:
    """Load one cache entry and its person manifest metadata."""
    path = Path(cache_path)
    trial_path = path if path.suffix == ".npz" else path / f"{trial_id}.npz"
    if trial_id is None and path.suffix != ".npz":
        raise ValueError("trial_id is required when loading from a person cache directory")
    if not trial_path.exists():
        raise FileNotFoundError(f"Missing cached trial: {trial_path}")
    manifest_path = trial_path.parent / "manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with np.load(trial_path, allow_pickle=False) as data:
        trial = PosePairTrial(
            face=data["face"],
            side=data["side"],
            valid_face=data["valid_face"],
            valid_side=data["valid_side"],
            timestamps=data["timestamps"],
            face_map=data["face_map"],
            side_map=data["side_map"],
            joint_names=tuple(str(name) for name in data["joint_names"].tolist()),
            person_id=str(data["person_id"].item()),
            trial_id=str(data["trial_id"].item()),
            fps=float(data["fps"].item()),
        )
    return trial, metadata
