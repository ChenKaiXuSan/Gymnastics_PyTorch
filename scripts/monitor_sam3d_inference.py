#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ProgressSnapshot:
    completed_ids: tuple[int, ...]
    incomplete_ids: tuple[int, ...]
    npz_count: int
    latest_npz_mtime: float


def parse_person_ids(spec: str) -> list[int]:
    person_ids: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"descending range is not allowed: {part}")
            person_ids.update(range(start, end + 1))
        else:
            person_ids.add(int(part))
    if not person_ids:
        raise ValueError("person ID specification is empty")
    return sorted(person_ids)


def _view_npz_files(result_root: Path, person_id: int, view: str) -> list[Path]:
    return list((result_root / str(person_id) / view).glob("*_sam3d_body.npz"))


def collect_progress(
    person_ids: Sequence[int], result_root: Path, person_log_root: Path
) -> ProgressSnapshot:
    completed: list[int] = []
    all_npz: list[Path] = []
    for person_id in person_ids:
        face_files = _view_npz_files(result_root, person_id, "face")
        side_files = _view_npz_files(result_root, person_id, "side")
        all_npz.extend(face_files)
        all_npz.extend(side_files)
        person_log = person_log_root / f"{person_id}.log"
        finish_marker = f"==== Finished Person: {person_id} ===="
        finished = person_log.exists() and finish_marker in person_log.read_text(
            encoding="utf-8", errors="replace"
        )
        if finished and face_files and side_files:
            completed.append(person_id)
    completed_set = set(completed)
    incomplete = [person_id for person_id in person_ids if person_id not in completed_set]
    latest_mtime = max((path.stat().st_mtime for path in all_npz), default=0.0)
    return ProgressSnapshot(
        completed_ids=tuple(completed),
        incomplete_ids=tuple(incomplete),
        npz_count=len(all_npz),
        latest_npz_mtime=latest_mtime,
    )
