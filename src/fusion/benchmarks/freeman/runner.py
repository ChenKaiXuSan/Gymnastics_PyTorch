"""Stage orchestration for the FreeMan benchmark: the ``StageOperations``
protocol, the persisted run state, per-subject sequencing, multi-GPU workers
and forced-stage resets."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from copy import deepcopy
from pathlib import Path
from typing import Any

from .download import cleanup_subject_workspace
from fusion.benchmarks.freeman.stages import (
    _atomic_json,
    _load_state,
)


def partition_subjects(
    subjects: Sequence[int],
    worker_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition unique sorted subjects round-robin across workers."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    partitions: list[list[int]] = [[] for _ in range(worker_count)]
    for index, subject in enumerate(sorted({int(value) for value in subjects})):
        if subject < 1 or subject > 40:
            raise ValueError("FreeMan subjects must be within 1..40")
        partitions[index % worker_count].append(subject)
    return tuple(tuple(values) for values in partitions)


class StageOperations:
    """Replaceable stage boundary used by the CLI and its tests."""

    def inspect(self, config: Mapping[str, Any], *, dry_run: bool = False) -> Any:
        raise NotImplementedError

    def download(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def infer(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def fuse(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def evaluate(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def report(self, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def run(
        self,
        config: Mapping[str, Any],
        *,
        force_stage: str | None = None,
        keep_workspace: bool = False,
        dry_run: bool = False,
        devices: Sequence[int] | None = None,
    ) -> Any:
        raise NotImplementedError


def run_subjects(
    subjects: Sequence[int],
    *,
    state_path: Path,
    process: Callable[[int], Any],
    cleanup: Callable[[int], Any],
    keep_workspace: bool,
) -> None:
    """Run subjects in numeric order, publishing state after every transition.

    Every write re-loads the state file and merges only this subject's entry,
    so concurrent per-subject jobs (one qsub request per subject) cannot
    resurrect stale sibling statuses from a snapshot held across hours.
    """
    state_file = Path(state_path)

    def publish(key: str, value: Mapping[str, Any]) -> None:
        state = _load_state(state_file)
        state["subjects"][key] = dict(value)
        _atomic_json(state_file, state)

    for subject in sorted({int(value) for value in subjects}):
        if subject < 1 or subject > 40:
            raise ValueError("FreeMan subjects must be within 1..40")
        key = str(subject)
        current = _load_state(state_file)["subjects"].get(key, {})
        if current.get("status") == "complete":
            continue
        publish(key, {"status": "running"})
        try:
            artifacts = process(subject)
            if not keep_workspace:
                cleanup(subject)
        except Exception as error:
            publish(
                key,
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            raise
        completed: dict[str, Any] = {"status": "complete"}
        if isinstance(artifacts, Mapping):
            completed["artifacts"] = dict(artifacts)
        publish(key, completed)


def _worker_state_path(output_root: Path, device: int) -> Path:
    return (
        Path(output_root)
        / "workers"
        / f"device_{int(device)}"
        / "run_state.json"
    )


def _run_device_worker(
    config: Mapping[str, Any],
    device: int,
    subjects: Sequence[int],
    state_path: Path,
    keep_workspace: bool,
) -> None:
    """Run one disjoint subject shard with one visible physical GPU."""
    worker_config = deepcopy(dict(config))
    worker_config["sam3d"]["device"] = 0
    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(device))
    from fusion.benchmarks.freeman.cli import DefaultStageOperations  # spawned worker; avoids a cycle

    operations = DefaultStageOperations()
    work_root = Path(worker_config["paths"]["work_root"])
    try:
        run_subjects(
            subjects,
            state_path=Path(state_path),
            process=lambda subject: operations._process_subject(
                worker_config,
                subject,
            ),
            cleanup=lambda subject: cleanup_subject_workspace(
                subject,
                work_root / f"subject_{subject:02d}",
                work_root,
            ),
            keep_workspace=keep_workspace,
        )
    finally:
        if previous_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices


def _merge_worker_states(
    canonical_path: Path,
    worker_paths: Sequence[Path],
) -> dict[str, Any]:
    """Atomically merge worker terminal states without downgrading completion."""
    canonical = _load_state(Path(canonical_path))
    canonical_subjects = canonical["subjects"]
    for worker_path in worker_paths:
        worker = _load_state(Path(worker_path))
        for subject, details in worker["subjects"].items():
            if canonical_subjects.get(subject, {}).get("status") == "complete":
                continue
            if (
                isinstance(details, Mapping)
                and details.get("status") in {"complete", "failed"}
            ):
                canonical_subjects[str(subject)] = dict(details)
    _atomic_json(Path(canonical_path), canonical)
    return canonical


def _run_parallel_subjects(
    config: Mapping[str, Any],
    canonical_state_path: Path,
    devices: Sequence[int],
    keep_workspace: bool,
) -> None:
    """Run outstanding subjects in isolated spawned GPU workers."""
    device_ids = tuple(int(value) for value in devices)
    if (
        not device_ids
        or len(set(device_ids)) != len(device_ids)
        or any(value < 0 for value in device_ids)
    ):
        raise ValueError("devices must contain unique non-negative integers")
    canonical = _load_state(Path(canonical_state_path))
    outstanding = [
        int(subject)
        for subject in config["dataset"]["subjects"]
        if canonical["subjects"].get(str(int(subject)), {}).get("status")
        != "complete"
    ]
    assignments = partition_subjects(outstanding, len(device_ids))
    context = multiprocessing.get_context("spawn")
    workers: list[tuple[int, tuple[int, ...], Path, Any]] = []
    output_root = Path(config["paths"]["output_root"])
    for device, subjects in zip(device_ids, assignments):
        if not subjects:
            continue
        state_path = _worker_state_path(output_root, device)
        seed = {
            "stages": {},
            "subjects": {
                str(subject): dict(canonical["subjects"][str(subject)])
                for subject in subjects
                if str(subject) in canonical["subjects"]
                and canonical["subjects"][str(subject)].get("status")
                != "complete"
            },
        }
        _atomic_json(state_path, seed)
        process = context.Process(
            target=_run_device_worker,
            args=(
                config,
                device,
                subjects,
                state_path,
                keep_workspace,
            ),
        )
        workers.append((device, subjects, state_path, process))
        previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        try:
            process.start()
        finally:
            if previous_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices

    for _, _, _, process in workers:
        process.join()

    failures: list[tuple[int, int]] = []
    for device, subjects, state_path, process in workers:
        exitcode = int(process.exitcode or 0)
        if exitcode == 0:
            continue
        failures.append((device, exitcode))
        worker_state = _load_state(state_path)
        for subject in subjects:
            key = str(subject)
            status = worker_state["subjects"].get(key, {}).get("status")
            if status in {"complete", "failed"}:
                continue
            worker_state["subjects"][key] = {
                "status": "failed",
                "error_type": "WorkerProcessError",
                "error_message": (
                    f"device {device} worker exited with code {exitcode}"
                ),
            }
        _atomic_json(state_path, worker_state)

    _merge_worker_states(
        Path(canonical_state_path),
        [state_path for _, _, state_path, _ in workers],
    )
    if failures:
        details = ", ".join(
            f"device {device} (exit code {exitcode})"
            for device, exitcode in failures
        )
        raise RuntimeError(f"FreeMan GPU workers failed: {details}")

