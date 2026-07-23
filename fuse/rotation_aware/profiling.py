"""Optional CPU wall-clock and deferred CUDA stage timing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from threading import Lock
from typing import Iterator

import torch


class StageProfiler:
    """Collect stage timings without synchronizing CUDA during the hot path."""

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self._wall: dict[str, list[float]] = {}
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._lock = Lock()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        started = time.perf_counter()
        events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        if self.device.type == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            begin.record(stream)
            events = (begin, end)
        try:
            yield
        finally:
            if events is not None:
                events[1].record(stream)
            with self._lock:
                if events is not None:
                    self._events.setdefault(name, []).append(events)
                self._wall.setdefault(name, []).append(time.perf_counter() - started)

    @contextmanager
    def cpu_stage(self, name: str) -> Iterator[None]:
        """Collect CPU wall time without creating CUDA events, including in workers."""
        if not self.enabled:
            yield
            return

        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_cpu_duration(name, time.perf_counter() - started)

    def record_cpu_duration(self, name: str, seconds: float) -> None:
        """Record a completed CPU duration without creating CUDA events."""
        if not self.enabled:
            return
        with self._lock:
            self._wall.setdefault(name, []).append(float(seconds))

    def summary(self) -> dict[str, dict[str, float | int]]:
        if not self.enabled:
            return {}
        with self._lock:
            wall = {name: list(values) for name, values in self._wall.items()}
            events_by_name = {
                name: list(events) for name, events in self._events.items()
            }
        if events_by_name:
            torch.cuda.synchronize(self.device)

        names = set(wall) | set(events_by_name)
        summary: dict[str, dict[str, float | int]] = {}
        for name in sorted(names):
            wall_values = wall.get(name, [])
            values: dict[str, float | int] = {
                "calls": len(wall_values),
                "wall_seconds": float(sum(wall_values)),
            }
            events = events_by_name.get(name, [])
            if events:
                values["cuda_seconds"] = float(
                    sum(begin.elapsed_time(end) for begin, end in events) / 1000.0
                )
            summary[name] = values
        return summary
