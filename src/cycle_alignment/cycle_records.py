"""Cycle record files: the hand-off from cycle detection to training.

Cycle detection runs once, offline, in :mod:`cycle_alignment` and
writes one JSON record per sequence.  Training code
(:mod:`fusion`) only reads these files through
:func:`read_cycle_record`; it contains no detection logic.

Two file layouts are understood.

Private recordings (``local/runs/split_cycle/person_<id>/alignment_record_<id>.json``)::

    {
      "metadata": {"person_id": "1", "offset_side_to_face": -3, "fps": 60.0, ...,
                   "cycle_detection": {...DetectionSettings...}},
      "cycles": [
        {"cycle_index": 0,
         "face_video_frames": {"start": 131, "mid": 209, "end": 287},
         "side_video_frames": {"start": 128, "mid": 206, "end": 284}},
        ...]
    }

    ``start``/``end`` are written by ``python -m cycle_alignment align``; ``mid`` and
    ``metadata.cycle_detection`` are added by ``python -m cycle_alignment cycles
    private`` (or by a fresh ``python -m cycle_alignment align`` run).  Frame numbers refer
    to each view's own video; the two views differ by the alignment offset.

Public datasets (``local/runs/cycle_records/<dataset>/<subject>/<sequence>.json``)::

    {
      "format": "cycle_record_v1",
      "metadata": {"dataset": "freeman", "subject_id": "12",
                   "sequence_id": "20220618_..._subj12", "fps": 25.0, "frames": 925,
                   "views": ["c04", "c07"], "cycle_detection": {...}},
      "cycles": [{"cycle_index": 0, "frames": {"start": 12, "mid": 40, "end": 70}}, ...]
    }

    Frame numbers refer to the shared (synchronised) frame index of the
    sequence, i.e. the ``frame_ids`` of the SAM3D predictions.

Both layouts are normalised into :class:`CycleRecord`, whose ``cycles`` are
on the view-A (face) frame timeline and whose optional ``side_cycles`` carry
the view-B video frames of the private layout.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .cycles import CycleSpan, DetectionSettings

FORMAT_V1 = "cycle_record_v1"


@dataclass(frozen=True)
class CycleRecord:
    """Normalised content of one cycle record file.

    Attributes:
        dataset: Dataset identifier (``"gymnastics"``, ``"freeman"``, ``"unity"``).
        subject_id: Subject identifier used for splitting.
        sequence_id: Sequence identifier (``"all"`` for the private layout).
        fps: Frames per second of the timeline.
        cycles: Cycles on the view-A frame timeline (with or without middles:
            ``mid`` equals ``-1`` when the record predates middle detection).
        side_cycles: View-B video frames of the private layout, else ``None``.
        frames: Number of frames of the sequence when known.
        detection: Detection settings stored with the record (may be empty).
        metadata: Every other metadata field of the file.
    """

    dataset: str
    subject_id: str
    sequence_id: str
    fps: float
    cycles: Tuple[Tuple[int, int, int], ...]
    side_cycles: Optional[Tuple[Tuple[int, int, int], ...]] = None
    frames: Optional[int] = None
    detection: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_mids(self) -> bool:
        """True when every cycle carries a middle frame."""
        return bool(self.cycles) and all(mid >= 0 for _, mid, _ in self.cycles)

    def spans(self) -> Tuple[CycleSpan, ...]:
        """Cycles as :class:`CycleSpan` objects (requires middles)."""
        if not self.has_mids:
            raise ValueError("record has no middle frames; run `python -m cycle_alignment cycles` first")
        return tuple(CycleSpan(s, m, e) for s, m, e in self.cycles)


def _span_dict(span: CycleSpan) -> Dict[str, int]:
    return {"start": int(span.start), "mid": int(span.mid), "end": int(span.end)}


def _triple(frames: Mapping[str, Any]) -> Tuple[int, int, int]:
    start, end = int(frames["start"]), int(frames["end"])
    mid = int(frames.get("mid", -1))
    if end <= start or (mid >= 0 and not start < mid < end):
        raise ValueError(f"invalid cycle frames {dict(frames)}")
    return start, mid, end


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def cycle_record_path(root: Path, subject_id: str, sequence_id: str) -> Path:
    """Canonical location of a public-dataset record below ``root``."""
    return Path(root) / f"subject_{subject_id}" / f"{sequence_id}.json"


def write_cycle_record(
    path: Path,
    *,
    dataset: str,
    subject_id: str,
    sequence_id: str,
    fps: float,
    frames: int,
    spans: Sequence[CycleSpan],
    detection: DetectionSettings,
    views: Sequence[str] = (),
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a ``cycle_record_v1`` file (public-dataset layout)."""
    payload: Dict[str, Any] = {
        "format": FORMAT_V1,
        "metadata": {
            "dataset": dataset,
            "subject_id": str(subject_id),
            "sequence_id": str(sequence_id),
            "fps": float(fps),
            "frames": int(frames),
            "views": list(views),
            "cycle_detection": detection.to_dict(),
            **dict(extra_metadata or {}),
        },
        "cycles": [{"cycle_index": i, "frames": _span_dict(span)} for i, span in enumerate(spans)],
    }
    _atomic_write_json(Path(path), payload)
    return Path(path)


def augment_alignment_record(
    path: Path,
    face_spans: Sequence[CycleSpan],
    side_spans: Sequence[CycleSpan],
    detection: DetectionSettings,
) -> Path:
    """Add ``mid`` frames to an existing private alignment record in place.

    Every existing key is preserved; ``start``/``end`` must match the record.

    Raises:
        ValueError: If the cycle count or boundaries differ from the file.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cycles = payload.get("cycles", [])
    if len(cycles) != len(face_spans) or len(cycles) != len(side_spans):
        raise ValueError(f"{path}: expected {len(cycles)} cycles, got {len(face_spans)}/{len(side_spans)}")
    for entry, face, side in zip(cycles, face_spans, side_spans):
        for key, span in (("face_video_frames", face), ("side_video_frames", side)):
            frames = entry[key]
            if int(frames["start"]) != span.start or int(frames["end"]) != span.end:
                raise ValueError(f"{path}: cycle {entry.get('cycle_index')} {key} boundaries changed")
            frames["mid"] = int(span.mid)
    payload.setdefault("metadata", {})["cycle_detection"] = detection.to_dict()
    _atomic_write_json(path, payload)
    return path


def read_cycle_record(path: Path) -> CycleRecord:
    """Read either layout into a :class:`CycleRecord`.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the content is malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cycles"), list):
        raise ValueError(f"cycle record must contain a cycles list: {path}")
    metadata = dict(payload.get("metadata", {}))
    detection = dict(metadata.pop("cycle_detection", {}) or {})
    if payload.get("format") == FORMAT_V1:
        cycles = tuple(_triple(entry["frames"]) for entry in payload["cycles"])
        return CycleRecord(
            dataset=str(metadata.pop("dataset")),
            subject_id=str(metadata.pop("subject_id")),
            sequence_id=str(metadata.pop("sequence_id")),
            fps=float(metadata.pop("fps")),
            cycles=cycles,
            frames=int(frames_value) if (frames_value := metadata.pop("frames", None)) is not None else None,
            detection=detection,
            metadata=metadata,
        )
    # Private alignment_record layout.
    face = tuple(_triple(entry["face_video_frames"]) for entry in payload["cycles"])
    side = tuple(_triple(entry["side_video_frames"]) for entry in payload["cycles"])
    return CycleRecord(
        dataset="gymnastics",
        subject_id=str(metadata.get("person_id", path.stem.split("_")[-1])),
        sequence_id="all",
        fps=float(metadata.get("fps", 60.0)),
        cycles=face,
        side_cycles=side,
        frames=None,
        detection=detection,
        metadata=metadata,
    )
