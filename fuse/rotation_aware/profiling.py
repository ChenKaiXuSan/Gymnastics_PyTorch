"""Optional CPU wall-clock and deferred CUDA stage timing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


class StageProfiler:
    """Collect stage timings without synchronizing CUDA during the hot path."""

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self._wall: dict[str, list[float]] = {}
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

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
            begin.record()
            events = (begin, end)
        try:
            yield
        finally:
            if events is not None:
                events[1].record()
                self._events.setdefault(name, []).append(events)
            self._wall.setdefault(name, []).append(time.perf_counter() - started)

    def summary(self) -> dict[str, dict[str, float | int]]:
        if not self.enabled:
            return {}
        if self._events:
            torch.cuda.synchronize(self.device)

        names = set(self._wall) | set(self._events)
        summary: dict[str, dict[str, float | int]] = {}
        for name in sorted(names):
            wall = self._wall.get(name, [])
            values: dict[str, float | int] = {
                "calls": len(wall),
                "wall_seconds": float(sum(wall)),
            }
            events = self._events.get(name, [])
            if events:
                values["cuda_seconds"] = float(
                    sum(begin.elapsed_time(end) for begin, end in events) / 1000.0
                )
            summary[name] = values
        return summary
