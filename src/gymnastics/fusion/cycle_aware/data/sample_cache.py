"""Optional on-disk cache of converted :class:`DualViewSample` lists.

Converting raw material into the common representation canonicalises every
view frame by frame (a Python loop in the shared body-frame code), which
costs about half a second per 1,000 frames and view.  For the private data
this is negligible, but a full FreeMan load (40 subjects, ~3,700 sessions)
takes several minutes.  When ``data.cache_dir`` is set, the base DataModule
stores the converted samples once and reloads them on later runs.

Layout:
    <cache_dir>/<dataset>_<key>/index.json      list of sample files + config
    <cache_dir>/<dataset>_<key>/<n>.npz         one sample per file

The key is a SHA-256 of the dataset name, skeleton, adapter options and the
``attach_reference`` flag, so a changed configuration never reads a stale
cache.  Metadata is stored as JSON inside the npz.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..sample import CanonicalTransformRecord, DualViewSample

INDEX_FILENAME = "index.json"


def cache_key(name: str, skeleton: str, options: Mapping[str, Any], attach_reference: bool) -> str:
    """Deterministic key for one adapter configuration."""
    payload = json.dumps({"name": name, "skeleton": skeleton, "options": dict(options), "attach_reference": bool(attach_reference)}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_samples(directory: Path, samples: Sequence[DualViewSample], *, config: Mapping[str, Any]) -> Path:
    """Write ``samples`` below ``directory`` (created if needed)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for index, sample in enumerate(samples):
        arrays: dict[str, Any] = {
            "view_a": sample.view_a,
            "view_b": sample.view_b,
            "valid_a": sample.valid_a,
            "valid_b": sample.valid_b,
            "timestamps": sample.timestamps,
            "cycle_bounds": np.asarray(sample.cycle_bounds, dtype=np.int64).reshape(-1, 2),
            "cycle_mids": np.asarray(sample.cycle_mids, dtype=np.int64),
            "header": np.asarray(
                json.dumps(
                    {
                        "dataset": sample.dataset,
                        "subject_id": sample.subject_id,
                        "sequence_id": sample.sequence_id,
                        "joint_names": list(sample.joint_names),
                        "metadata": _plain(sample.metadata),
                        "reference_canonical": bool(sample.reference_canonical),
                    }
                )
            ),
        }
        if sample.reference is not None and sample.reference_valid is not None:
            arrays["reference"] = sample.reference
            arrays["reference_valid"] = sample.reference_valid
        if sample.transform_a is not None:
            arrays["transform_rotation"] = sample.transform_a.rotation
            arrays["transform_origin"] = sample.transform_a.origin
            arrays["transform_scale"] = np.asarray(sample.transform_a.scale, dtype=np.float64)
            arrays["transform_valid"] = sample.transform_a.valid
        filename = f"{index:06d}.npz"
        np.savez_compressed(directory / filename, **arrays)
        files.append(filename)
    (directory / INDEX_FILENAME).write_text(json.dumps({"files": files, "config": _plain(config)}, indent=2, default=str), encoding="utf-8")
    return directory


def load_samples(directory: Path) -> list[DualViewSample] | None:
    """Read samples written by :func:`save_samples`; ``None`` when absent."""
    directory = Path(directory)
    index_path = directory / INDEX_FILENAME
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    samples: list[DualViewSample] = []
    for filename in index["files"]:
        with np.load(directory / filename, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            transform = None
            if "transform_rotation" in data:
                transform = CanonicalTransformRecord(
                    rotation=data["transform_rotation"],
                    origin=data["transform_origin"],
                    scale=float(data["transform_scale"]),
                    valid=data["transform_valid"],
                )
            samples.append(
                DualViewSample(
                    dataset=header["dataset"],
                    subject_id=header["subject_id"],
                    sequence_id=header["sequence_id"],
                    view_a=data["view_a"],
                    view_b=data["view_b"],
                    valid_a=data["valid_a"],
                    valid_b=data["valid_b"],
                    timestamps=data["timestamps"],
                    joint_names=tuple(header["joint_names"]),
                    cycle_bounds=tuple((int(s), int(e)) for s, e in data["cycle_bounds"].tolist()),
                    cycle_mids=tuple(int(m) for m in data["cycle_mids"].tolist()) if "cycle_mids" in data else (),
                    reference=data["reference"] if "reference" in data else None,
                    reference_valid=data["reference_valid"] if "reference_valid" in data else None,
                    reference_canonical=bool(header.get("reference_canonical", False)),
                    transform_a=transform,
                    metadata=header["metadata"],
                )
            )
    return samples


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value
