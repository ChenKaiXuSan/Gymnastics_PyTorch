# SAM3D-Body Inference Email Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a tmux-hosted monitor that emails `chenkaixusan@gmail.com` when the active 69-person SAM3D-Body inference completes, stops, stalls, or emits a fatal error.

**Architecture:** A single standard-library Python script separates pure status collection and decision functions from SMTP and CLI side effects. It persists JSON state atomically so restarts do not resend alerts, reads credentials from a mode-`0600` file outside Git, and polls the active inference process, stable log, person logs, and NPZ outputs every 60 seconds.

**Tech Stack:** Python 3.10+, `argparse`, `dataclasses`, `email.message`, `hashlib`, `json`, `pathlib`, `smtplib`, `subprocess`, `pytest`, tmux.

## Global Constraints

- Run repository Python and tests with `conda run -n gymnastic` unless a SAM3D model process itself requires the existing `sam_3d_body` environment.
- Use only the Python standard library in the monitor.
- Monitor exactly `69-134,136-138`; ID135 remains excluded.
- Keep Gmail credentials outside the repository in `/home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env` with mode `0600`.
- Never print, persist in monitor state, or commit `SMTP_APP_PASSWORD`.
- Read inference outputs from `/home/data/xchen/gymnastics/sam3d_body_results/person`.
- Read the stable run log `/home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_new_20260718_w2.stdout.log`.
- Treat `No person detected` as a nonfatal frame warning.
- Send each notification fingerprint at most once across monitor restarts.
- Do not send real email from tests.

## File Structure

- Create `scripts/monitor_sam3d_inference.py`: ID parsing, output/log/process inspection, state persistence, SMTP delivery, polling decisions, and CLI entry point.
- Create `tests/test_sam3d_inference_monitor.py`: focused unit and orchestration tests with temporary directories and fake SMTP/process inputs.
- Create outside Git during deployment: `/home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env`.
- Create at runtime: `/home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_state.json`.
- Create at deployment: `/home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_20260718.log`.

---

### Task 1: Person IDs And Progress Collection

**Files:**
- Create: `scripts/monitor_sam3d_inference.py`
- Create: `tests/test_sam3d_inference_monitor.py`

**Interfaces:**
- Produces: `parse_person_ids(spec: str) -> list[int]`
- Produces: `ProgressSnapshot`
- Produces: `collect_progress(person_ids: Sequence[int], result_root: Path, person_log_root: Path) -> ProgressSnapshot`

- [ ] **Step 1: Write failing tests for ID parsing and completion rules**

```python
from pathlib import Path

import pytest

from scripts import monitor_sam3d_inference as monitor


def test_parse_person_ids_supports_ranges_and_gap():
    ids = monitor.parse_person_ids("69-134,136-138")
    assert len(ids) == 69
    assert ids[0] == 69
    assert ids[-1] == 138
    assert 135 not in ids


def test_parse_person_ids_rejects_descending_range():
    with pytest.raises(ValueError, match="descending range"):
        monitor.parse_person_ids("10-8")


def test_collect_progress_requires_finished_log_and_both_views(tmp_path):
    result_root = tmp_path / "person"
    log_root = tmp_path / "person_logs"
    (result_root / "69" / "face").mkdir(parents=True)
    (result_root / "69" / "side").mkdir(parents=True)
    log_root.mkdir()
    (result_root / "69" / "face" / "000000_sam3d_body.npz").write_bytes(b"face")
    (result_root / "69" / "side" / "000000_sam3d_body.npz").write_bytes(b"side")
    (log_root / "69.log").write_text(
        "==== Finished Person: 69 ====\n", encoding="utf-8"
    )

    snapshot = monitor.collect_progress([69, 70], result_root, log_root)

    assert snapshot.completed_ids == (69,)
    assert snapshot.incomplete_ids == (70,)
    assert snapshot.npz_count == 2
    assert snapshot.latest_npz_mtime > 0


def test_collect_progress_keeps_partial_person_incomplete(tmp_path):
    result_root = tmp_path / "person"
    log_root = tmp_path / "person_logs"
    (result_root / "69" / "face").mkdir(parents=True)
    log_root.mkdir()
    (result_root / "69" / "face" / "000000_sam3d_body.npz").write_bytes(b"face")
    (log_root / "69.log").write_text(
        "==== Finished Person: 69 ====\n", encoding="utf-8"
    )

    snapshot = monitor.collect_progress([69], result_root, log_root)

    assert snapshot.completed_ids == ()
    assert snapshot.incomplete_ids == (69,)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
```

Expected: collection fails with `ImportError` because `scripts.monitor_sam3d_inference` does not exist.

- [ ] **Step 3: Implement ID parsing and progress collection**

```python
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
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit progress collection**

```bash
git add scripts/monitor_sam3d_inference.py tests/test_sam3d_inference_monitor.py
git commit -m "feat: collect SAM3D inference progress"
```

---

### Task 2: Persistent State, Log Errors, Process Exit, And Stall Decisions

**Files:**
- Modify: `scripts/monitor_sam3d_inference.py`
- Modify: `tests/test_sam3d_inference_monitor.py`

**Interfaces:**
- Consumes: `ProgressSnapshot`
- Produces: `MonitorState`, `load_state`, `save_state`
- Produces: `scan_new_fatal_errors(run_log: Path, offset: int) -> tuple[list[str], int]`
- Produces: `process_is_active(commands: Sequence[str], process_match: str) -> bool`
- Produces: `update_progress_state(state: MonitorState, snapshot: ProgressSnapshot, now: float) -> bool`

- [ ] **Step 1: Write failing tests for state, incremental log scans, process matching, and stalls**

```python
def test_scan_new_fatal_errors_uses_offset_and_ignores_frame_warning(tmp_path):
    run_log = tmp_path / "run.log"
    run_log.write_text("No person detected in frame 3\n", encoding="utf-8")
    errors, offset = monitor.scan_new_fatal_errors(run_log, 0)
    assert errors == []

    with run_log.open("a", encoding="utf-8") as handle:
        handle.write("GPU 0 处理 83 时出错: CUDA out of memory\n")
    errors, new_offset = monitor.scan_new_fatal_errors(run_log, offset)

    assert errors == ["GPU 0 处理 83 时出错: CUDA out of memory"]
    assert new_offset > offset


def test_scan_resets_offset_when_log_is_truncated(tmp_path):
    run_log = tmp_path / "run.log"
    run_log.write_text("CUDA error\n", encoding="utf-8")

    errors, offset = monitor.scan_new_fatal_errors(run_log, 10_000)

    assert errors == ["CUDA error"]
    assert offset == run_log.stat().st_size


def test_process_is_active_requires_main_and_run_match():
    commands = [
        "python -m SAM3Dbody.main infer.workers_per_gpu=2",
        "python scripts/monitor_sam3d_inference.py",
    ]
    assert monitor.process_is_active(commands, "infer.workers_per_gpu=2")
    assert not monitor.process_is_active(commands, "infer.workers_per_gpu=3")


def test_state_round_trip_and_atomic_save(tmp_path):
    state_path = tmp_path / "state.json"
    state = monitor.MonitorState(
        started_at=100.0,
        last_progress_at=120.0,
        last_npz_count=10,
        latest_npz_mtime=110.0,
        log_offset=42,
        sent_fingerprints=["abc"],
        stalled=True,
    )
    monitor.save_state(state_path, state)
    assert monitor.load_state(state_path, now=999.0) == state
    assert not state_path.with_suffix(".json.tmp").exists()


def test_update_progress_state_detects_progress_and_stall_age():
    state = monitor.MonitorState.new(now=100.0)
    snapshot = monitor.ProgressSnapshot((), (69,), 3, 90.0)

    progressed = monitor.update_progress_state(state, snapshot, now=120.0)

    assert progressed
    assert state.last_progress_at == 120.0
    assert not monitor.is_stalled(state, now=1_919.0, stall_seconds=1_800)
    assert monitor.is_stalled(state, now=1_920.0, stall_seconds=1_800)
```

- [ ] **Step 2: Run the new tests and verify missing symbols fail**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
```

Expected: failures name `scan_new_fatal_errors`, `process_is_active`, and `MonitorState`.

- [ ] **Step 3: Implement persistent state and decisions**

```python
import json
import os
from dataclasses import asdict, field


FATAL_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "Traceback",
    "时出错",
    "CUDA error",
    "Killed",
    "No module named",
)


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
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
```

Expected: all monitor tests pass.

- [ ] **Step 5: Commit status decisions**

```bash
git add scripts/monitor_sam3d_inference.py tests/test_sam3d_inference_monitor.py
git commit -m "feat: detect SAM3D monitor failures and stalls"
```

---

### Task 3: Secure Gmail SMTP And Notification Deduplication

**Files:**
- Modify: `scripts/monitor_sam3d_inference.py`
- Modify: `tests/test_sam3d_inference_monitor.py`

**Interfaces:**
- Produces: `SmtpSettings`, `Notification`
- Produces: `load_smtp_settings(path: Path) -> SmtpSettings`
- Produces: `make_notification(kind: str, detail: str, snapshot: ProgressSnapshot) -> Notification`
- Produces: `send_email(settings: SmtpSettings, notification: Notification, smtp_factory, sleep_fn) -> None`
- Produces: `send_once(notification: Notification, state: MonitorState, sender: Callable[[Notification], bool]) -> bool`

- [ ] **Step 1: Write failing tests for secure config, SMTP retries, and deduplication**

```python
def test_load_smtp_settings_requires_private_file(tmp_path):
    config = tmp_path / "smtp.env"
    config.write_text(
        "SMTP_HOST=smtp.gmail.com\n"
        "SMTP_PORT=465\n"
        "SMTP_USER=chenkaixusan@gmail.com\n"
        "SMTP_APP_PASSWORD=abcdefghijklmnop\n"
        "EMAIL_TO=chenkaixusan@gmail.com\n",
        encoding="utf-8",
    )
    config.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        monitor.load_smtp_settings(config)

    config.chmod(0o600)
    settings = monitor.load_smtp_settings(config)
    assert settings.user == "chenkaixusan@gmail.com"
    assert settings.port == 465


def test_send_email_retries_without_exposing_password():
    attempts = []

    class FakeSmtp:
        def __init__(self, host, port, timeout):
            attempts.append((host, port, timeout))

        def __enter__(self):
            if len(attempts) < 3:
                raise OSError("temporary network failure")
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def login(self, user, password):
            assert password == "secret-app-password"

        def send_message(self, message):
            assert message["To"] == "chenkaixusan@gmail.com"

    settings = monitor.SmtpSettings(
        "smtp.gmail.com", 465, "chenkaixusan@gmail.com",
        "secret-app-password", "chenkaixusan@gmail.com"
    )
    notification = monitor.Notification("ERROR", "fp", "subject", "body")
    monitor.send_email(
        settings, notification, smtp_factory=FakeSmtp, sleep_fn=lambda _: None
    )
    assert len(attempts) == 3


def test_send_once_records_only_successful_delivery():
    state = monitor.MonitorState.new(now=100.0)
    notification = monitor.Notification("ERROR", "same", "subject", "body")
    deliveries = []

    def sender(item):
        deliveries.append(item)
        return True

    assert monitor.send_once(notification, state, sender)
    assert monitor.send_once(notification, state, sender)
    assert deliveries == [notification]
    assert state.sent_fingerprints == ["same"]


def test_send_once_does_not_record_failed_delivery():
    state = monitor.MonitorState.new(now=100.0)
    notification = monitor.Notification("ERROR", "retry-me", "subject", "body")

    assert not monitor.send_once(notification, state, lambda _: False)
    assert state.sent_fingerprints == []
```

- [ ] **Step 2: Run tests and verify SMTP symbols are missing**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
```

Expected: failures name `SmtpSettings`, `Notification`, and `load_smtp_settings`.

- [ ] **Step 3: Implement secure SMTP settings and delivery**

```python
import hashlib
import smtplib
import stat
import time
from collections.abc import Callable
from email.message import EmailMessage


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    user: str
    password: str
    recipient: str


@dataclass(frozen=True)
class Notification:
    kind: str
    fingerprint: str
    subject: str
    body: str


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
    required = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_APP_PASSWORD", "EMAIL_TO")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"missing SMTP settings: {', '.join(missing)}")
    return SmtpSettings(
        host=values["SMTP_HOST"],
        port=int(values["SMTP_PORT"]),
        user=values["SMTP_USER"],
        password=values["SMTP_APP_PASSWORD"],
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
                sleep_fn(2 ** attempt)
    assert last_error is not None
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
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
```

Expected: all monitor tests pass without network access.

- [ ] **Step 5: Commit SMTP support**

```bash
git add scripts/monitor_sam3d_inference.py tests/test_sam3d_inference_monitor.py
git commit -m "feat: notify SAM3D monitor through Gmail"
```

---

### Task 4: Poll Orchestration, CLI, And Deployment Verification

**Files:**
- Modify: `scripts/monitor_sam3d_inference.py`
- Modify: `tests/test_sam3d_inference_monitor.py`
- Create outside Git: `/home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env`

**Interfaces:**
- Consumes: all Task 1-3 interfaces.
- Produces: `parse_args(argv: Sequence[str] | None) -> argparse.Namespace`
- Produces: `poll_once(args, state, sender, now, process_commands) -> int | None`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing tests for CLI defaults and terminal decisions**

```python
def test_parse_args_has_current_run_defaults():
    args = monitor.parse_args([])
    assert args.person_ids == "69-134,136-138"
    assert args.poll_seconds == 60
    assert args.stall_seconds == 1800
    assert args.process_match == "infer.workers_per_gpu=2"


def test_poll_once_sends_completion_and_returns_zero(tmp_path):
    result_root = tmp_path / "person"
    person_logs = tmp_path / "person_logs"
    run_log = tmp_path / "run.log"
    state_path = tmp_path / "state.json"
    for view in ("face", "side"):
        view_dir = result_root / "69" / view
        view_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "000000_sam3d_body.npz").write_bytes(view.encode())
    person_logs.mkdir()
    (person_logs / "69.log").write_text(
        "==== Finished Person: 69 ====\n", encoding="utf-8"
    )
    run_log.write_text("", encoding="utf-8")
    args = monitor.parse_args([
        "--person-ids", "69",
        "--result-root", str(result_root),
        "--person-log-root", str(person_logs),
        "--run-log", str(run_log),
        "--state-file", str(state_path),
        "--once",
    ])
    state = monitor.MonitorState.new(now=100.0)
    sent = []

    def sender(notification):
        sent.append(notification)
        return True

    exit_code = monitor.poll_once(args, state, sender, 120.0, [])

    assert exit_code == 0
    assert [notification.kind for notification in sent] == ["COMPLETED"]


def test_poll_once_reports_stopped_incomplete_run(tmp_path):
    args = monitor.parse_args([
        "--person-ids", "69",
        "--result-root", str(tmp_path / "person"),
        "--person-log-root", str(tmp_path / "person_logs"),
        "--run-log", str(tmp_path / "run.log"),
        "--state-file", str(tmp_path / "state.json"),
        "--once",
    ])
    state = monitor.MonitorState.new(now=100.0)
    sent = []

    def sender(notification):
        sent.append(notification)
        return True

    exit_code = monitor.poll_once(args, state, sender, 120.0, [])

    assert exit_code == 1
    assert [notification.kind for notification in sent] == ["STOPPED"]
```

- [ ] **Step 2: Run tests and verify orchestration symbols are missing**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
```

Expected: failures name `parse_args` and `poll_once`.

- [ ] **Step 3: Implement CLI, polling order, and main loop**

```python
import argparse
import logging
import subprocess
import sys


DEFAULT_RESULT_ROOT = Path("/home/data/xchen/gymnastics/sam3d_body_results/person")
DEFAULT_LOG_ROOT = Path("/home/data/xchen/gymnastics/sam3d_body_results/logs")
DEFAULT_SMTP_CONFIG = Path(
    "/home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor SAM3D-Body inference")
    parser.add_argument("--person-ids", default="69-134,136-138")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--person-log-root", type=Path, default=DEFAULT_LOG_ROOT / "person_logs"
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=DEFAULT_LOG_ROOT / "sam3dbody_new_20260718_w2.stdout.log",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_LOG_ROOT / "sam3dbody_monitor_state.json",
    )
    parser.add_argument("--smtp-config", type=Path, default=DEFAULT_SMTP_CONFIG)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    parser.add_argument("--process-match", default="infer.workers_per_gpu=2")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def list_process_commands() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "args="], check=True, capture_output=True, text=True
    )
    return result.stdout.splitlines()


def poll_once(
    args: argparse.Namespace,
    state: MonitorState,
    sender: Callable[[Notification], bool],
    now: float,
    process_commands: Sequence[str],
) -> int | None:
    person_ids = parse_person_ids(args.person_ids)
    snapshot = collect_progress(person_ids, args.result_root, args.person_log_root)
    progressed = update_progress_state(state, snapshot, now)
    previous_log_offset = state.log_offset
    errors, new_log_offset = scan_new_fatal_errors(args.run_log, state.log_offset)
    errors_delivered = True
    for error in errors:
        if not send_once(make_notification("ERROR", error, snapshot), state, sender):
            errors_delivered = False
            break
    state.log_offset = new_log_offset if errors_delivered else previous_log_offset

    if not snapshot.incomplete_ids:
        detail = f"All {len(person_ids)} target persons completed."
        delivered = send_once(
            make_notification("COMPLETED", detail, snapshot), state, sender
        )
        save_state(args.state_file, state)
        return 0 if delivered else None

    active = process_is_active(process_commands, args.process_match)
    if not active:
        detail = "Inference exited with incomplete persons: " + ",".join(
            map(str, snapshot.incomplete_ids)
        )
        delivered = send_once(make_notification("STOPPED", detail, snapshot), state, sender)
        save_state(args.state_file, state)
        return 1 if delivered else None

    if progressed and state.stalled:
        delivered = send_once(
            make_notification("RECOVERED", "Output resumed after a stall.", snapshot),
            state,
            sender,
        )
        if delivered:
            state.stalled = False
    elif is_stalled(state, now, args.stall_seconds) and not state.stalled:
        delivered = send_once(
            make_notification(
                "STALLED",
                f"No NPZ progress for at least {args.stall_seconds} seconds.",
                snapshot,
            ),
            state,
            sender,
        )
        if delivered:
            state.stalled = True

    save_state(args.state_file, state)
    logging.info(
        "completed=%d/%d npz=%d active=%s",
        len(snapshot.completed_ids),
        len(person_ids),
        snapshot.npz_count,
        active,
    )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_smtp_settings(args.smtp_config)
    state = load_state(args.state_file, now=time.time())

    def sender(notification: Notification) -> bool:
        try:
            send_email(settings, notification)
        except (OSError, smtplib.SMTPException):
            logging.exception("email delivery failed; the next poll will retry")
            return False
        return True

    while True:
        exit_code = poll_once(args, state, sender, time.time(), list_process_commands())
        if exit_code is not None:
            return exit_code
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run focused and full relevant tests**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_sam3d_inference_monitor.py -q
conda run -n sam_3d_body python -m pytest tests/test_sam3d_body_rotation.py -q
```

Expected: all tests pass, with no real SMTP connection.

- [ ] **Step 5: Run static and one-shot diagnostics without credentials**

Run:

```bash
conda run -n gymnastic python -m py_compile scripts/monitor_sam3d_inference.py
conda run -n gymnastic python scripts/monitor_sam3d_inference.py --help
```

Expected: compilation succeeds and help lists every documented argument.

- [ ] **Step 6: Commit the completed monitor**

```bash
git add scripts/monitor_sam3d_inference.py tests/test_sam3d_inference_monitor.py
git commit -m "feat: monitor SAM3D inference by email"
```

- [ ] **Step 7: Configure Gmail credentials securely**

In an interactive local shell, run:

```bash
install -d -m 700 /home/workspace/kaixu/.config/gymnastics
read -r -s -p 'Gmail application password: ' GYMNASTICS_GMAIL_APP_PASSWORD
printf '\n'
printf 'SMTP_HOST=smtp.gmail.com\nSMTP_PORT=465\nSMTP_USER=chenkaixusan@gmail.com\nSMTP_APP_PASSWORD=%s\nEMAIL_TO=chenkaixusan@gmail.com\n' "${GYMNASTICS_GMAIL_APP_PASSWORD// /}" | install -m 600 /dev/stdin /home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env
unset GYMNASTICS_GMAIL_APP_PASSWORD
```

The silent `read` keeps the application password out of shell history and the
space removal accepts the grouped form displayed by Google. Verify:

```bash
stat -c '%a %n' /home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env
```

Expected:

```text
600 /home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env
```

- [ ] **Step 8: Send a real SMTP smoke-test notification**

Run:

```bash
conda run -n gymnastic python -c 'from pathlib import Path; from scripts.monitor_sam3d_inference import Notification, load_smtp_settings, send_email; settings = load_smtp_settings(Path("/home/workspace/kaixu/.config/gymnastics/sam3d_monitor.env")); send_email(settings, Notification("TEST", "smtp-smoke-test", "[Gymnastics][SAM3D-Body] TEST", "SMTP configuration is working."))'
```

Expected: command exits zero and exactly one `TEST` email arrives at `chenkaixusan@gmail.com`. Notification deduplication remains covered by the mocked state tests, so this command is not repeated.

- [ ] **Step 9: Launch the production monitor in tmux**

Run:

```bash
tmux new-session -d -s sam3dbody_monitor_20260718 -c /home/workspace/kaixu/code/Gymnastics_PyTorch
tmux send-keys -t sam3dbody_monitor_20260718:0.0 'conda run --no-capture-output -n gymnastic python scripts/monitor_sam3d_inference.py 2>&1 | tee -a /home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_20260718.log' C-m
```

Expected: the first poll logs current completed-person and NPZ counts, the process remains alive, and the state JSON is created without containing SMTP credentials.

- [ ] **Step 10: Verify production monitoring state**

Run:

```bash
tmux has-session -t sam3dbody_monitor_20260718
tail -n 20 /home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_20260718.log
rg -n "SMTP_APP_PASSWORD|secret-app-password" /home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_state.json /home/data/xchen/gymnastics/sam3d_body_results/logs/sam3dbody_monitor_20260718.log
```

Expected: tmux exits zero; the monitor log shows an active poll; the secret scan returns no matches.
