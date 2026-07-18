#!/usr/bin/env python3

import hashlib
import json
import os
import smtplib
import stat
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from email.message import EmailMessage
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


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    user: str
    password: str = field(repr=False)
    recipient: str


@dataclass(frozen=True)
class Notification:
    kind: str
    fingerprint: str
    subject: str
    body: str


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


def load_smtp_settings(path: Path) -> SmtpSettings:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"SMTP config must have mode 0600: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    required = (
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_APP_PASSWORD",
        "EMAIL_TO",
    )
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"missing SMTP settings: {', '.join(missing)}")
    return SmtpSettings(
        host=values["SMTP_HOST"],
        port=int(values["SMTP_PORT"]),
        user=values["SMTP_USER"],
        password=values["SMTP_APP_PASSWORD"].replace(" ", ""),
        recipient=values["EMAIL_TO"],
    )


def make_notification(
    kind: str, detail: str, snapshot: ProgressSnapshot
) -> Notification:
    fingerprint = hashlib.sha256(f"{kind}\n{detail}".encode("utf-8")).hexdigest()
    subject = f"[Gymnastics][SAM3D-Body] {kind}"
    body = (
        f"Status: {kind}\n"
        f"Completed persons: {len(snapshot.completed_ids)}\n"
        f"Incomplete persons: {','.join(map(str, snapshot.incomplete_ids))}\n"
        f"NPZ files: {snapshot.npz_count}\n\n"
        f"Detail:\n{detail}\n"
    )
    return Notification(kind, fingerprint, subject, body)


def send_email(
    settings: SmtpSettings,
    notification: Notification,
    smtp_factory=smtplib.SMTP_SSL,
    sleep_fn=time.sleep,
) -> None:
    message = EmailMessage()
    message["From"] = settings.user
    message["To"] = settings.recipient
    message["Subject"] = notification.subject
    message.set_content(notification.body)
    last_error: OSError | smtplib.SMTPException | None = None
    for attempt in range(3):
        try:
            with smtp_factory(settings.host, settings.port, timeout=30) as smtp:
                smtp.login(settings.user, settings.password)
                smtp.send_message(message)
            return
        except (OSError, smtplib.SMTPException) as error:
            last_error = error
            if attempt < 2:
                sleep_fn(2**attempt)
    if last_error is None:
        raise RuntimeError("SMTP delivery failed without an exception")
    raise last_error


def send_once(
    notification: Notification,
    state: MonitorState,
    sender: Callable[[Notification], bool],
) -> bool:
    if notification.fingerprint in state.sent_fingerprints:
        return True
    if not sender(notification):
        return False
    state.sent_fingerprints.append(notification.fingerprint)
    return True
