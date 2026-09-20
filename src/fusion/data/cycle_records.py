"""Read-only access to precomputed cycle records for the DataModules.

Cycle detection lives in :mod:`cycle_alignment`; this module only maps
its record files onto the frame timeline of a :class:`DualViewSample`.  It
is the single import through which the training package touches cycle
information, so the boundary "training never detects cycles" is easy to
audit.

Two helpers are provided:

* :func:`private_cycles_for_trials` maps the ``alignment_record_<id>.json``
  cycles (face video frames, with middles) onto the concatenated
  all-cycles timeline built by :func:`...data.gymnastics.concatenate_cycles`.
* :func:`public_cycles_for_sequence` reads a ``cycle_record_v1`` file and
  maps its frame ids onto the sample timeline through the trial's
  ``face_map`` (the frame ids of view A).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from cycle_alignment.cycle_records import CycleRecord, cycle_record_path, read_cycle_record
from fusion.keypoints.schema import PosePairTrial


def private_cycles_for_trials(
    record_path: Path,
    ordered_trials: Sequence[PosePairTrial],
    *,
    require_mids: bool,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Cycle bounds and middles on the concatenated timeline of ``ordered_trials``.

    Each trial is one cycle (``cycle_NNN``) whose ``face_map`` holds the face
    video frames it covers.  The record's ``mid`` for that cycle is located
    in ``face_map`` and shifted by the trial's offset in the concatenation.

    Args:
        record_path: ``alignment_record_<id>.json`` of the person.
        ordered_trials: Cycle trials in concatenation order.
        require_mids: Raise when the record has no middles.

    Returns:
        Tuple ``(cycle_bounds, cycle_mids)``; ``cycle_mids`` is empty when
        the record has no middles and ``require_mids`` is false.

    Raises:
        ValueError: If a trial's face frames do not match a record cycle, or
            middles are required but absent.
    """
    record = read_cycle_record(record_path)
    by_start = {start: (start, mid, end) for start, mid, end in record.cycles}
    bounds: list[tuple[int, int]] = []
    mids: list[int] = []
    offset = 0
    for trial in ordered_trials:
        first = int(trial.face_map[0])
        entry = by_start.get(first)
        length = int(trial.face.shape[0])
        if entry is None or entry[2] != int(trial.face_map[-1]) + 1:
            raise ValueError(f"{record_path}: trial {trial.trial_id} face frames [{first}, {int(trial.face_map[-1]) + 1}) do not match a recorded cycle")
        bounds.append((offset, offset + length))
        if entry[1] >= 0:
            position = int(np.searchsorted(trial.face_map, entry[1]))
            if position <= 0 or position >= length or int(trial.face_map[position]) != entry[1]:
                raise ValueError(f"{record_path}: middle frame {entry[1]} of {trial.trial_id} is not inside the cycle")
            mids.append(offset + position)
        offset += length
    if len(mids) != len(bounds):
        if require_mids:
            raise ValueError(f"{record_path} has no cycle middles; run `python -m cycle_alignment cycles private` first")
        mids = []
    return tuple(bounds), tuple(mids)


def public_cycles_for_sequence(
    records_root: Path,
    subject_id: str,
    sequence_id: str,
    trial: PosePairTrial,
    *,
    require_record: bool,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], CycleRecord | None]:
    """Cycle bounds and middles of one public-dataset sequence on the trial timeline.

    Args:
        records_root: Root written by ``python -m cycle_alignment cycles <dataset>``.
        subject_id: Subject identifier used in the record path.
        sequence_id: Sequence identifier used in the record path.
        trial: The sequence trial (its ``face_map`` holds the frame ids).
        require_record: Raise when the record file is missing.

    Returns:
        Tuple ``(cycle_bounds, cycle_mids, record)``; empty tuples and
        ``None`` when the record is missing and not required.
    """
    path = cycle_record_path(records_root, subject_id, sequence_id)
    if not path.is_file():
        if require_record:
            raise FileNotFoundError(f"missing cycle record {path}; run `python -m cycle_alignment cycles` first")
        return (), (), None
    record = read_cycle_record(path)
    frame_ids = np.asarray(trial.face_map, dtype=np.int64)
    bounds: list[tuple[int, int]] = []
    mids: list[int] = []
    for start, mid, end in record.cycles:
        s = int(np.searchsorted(frame_ids, start))
        e = int(np.searchsorted(frame_ids, end))
        if s >= e or s >= len(frame_ids):
            continue
        e = min(e, len(frame_ids))
        if mid >= 0:
            m = int(np.searchsorted(frame_ids, mid))
            if not s < m < e:
                continue
            mids.append(m)
        bounds.append((s, e))
    if len(mids) != len(bounds):
        mids = []
    return tuple(bounds), tuple(mids), record
