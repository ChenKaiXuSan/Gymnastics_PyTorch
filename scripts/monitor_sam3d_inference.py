#!/usr/bin/env python3

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


FATAL_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "Traceback",
    "\u65f6\u51fa\u9519",
    "CUDA error",
    "Killed",
    "No module named",
)


@dataclass(frozen=True)
class ProgressSnapshot:
    completed_ids: tuple[int, ...]
    incomplete_ids: tuple[int, ...]
    npz_count: int
    latest_npz_mtime: float


@dataclass
class MonitorState:
    started_at: float
    last_progress_at: float
    last_npz_count: int = 0
    latest_npz_mtime: float = 0.0
    log_offset: int = 0
    sent_fingerprints: list[str] = field(default_factory=list)
    stalled: bool = False

    @classmethod
    def new(cls, now: float) -> "MonitorState":
        return cls(started_at=now, last_progress_at=now)


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


def load_state(path: Path, now: float) -> MonitorState:
    if not path.exists():
        return MonitorState.new(now)
    return MonitorState(**json.loads(path.read_text(encoding="utf-8")))


def save_state(path: Path, state: MonitorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def scan_new_fatal_errors(run_log: Path, offset: int) -> tuple[list[str], int]:
    if not run_log.exists():
        return [], 0
    size = run_log.stat().st_size
    if offset > size:
        offset = 0
    with run_log.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode("utf-8", errors="replace")
        new_offset = handle.tell()
    errors = [
        line.strip()
        for line in text.splitlines()
        if any(marker in line for marker in FATAL_MARKERS)
    ]
    return errors, new_offset


def process_is_active(commands: Sequence[str], process_match: str) -> bool:
    return any(
        "python -m SAM3Dbody.main" in command and process_match in command
        for command in commands
    )


def update_progress_state(
    state: MonitorState, snapshot: ProgressSnapshot, now: float
) -> bool:
    progressed = (
        snapshot.npz_count > state.last_npz_count
        or snapshot.latest_npz_mtime > state.latest_npz_mtime
    )
    if progressed:
        state.last_progress_at = now
    state.last_npz_count = snapshot.npz_count
    state.latest_npz_mtime = snapshot.latest_npz_mtime
    return progressed


def is_stalled(state: MonitorState, now: float, stall_seconds: int) -> bool:
    return now - state.last_progress_at >= stall_seconds
