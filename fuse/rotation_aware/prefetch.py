"""Bounded ordered CPU preparation helpers for rotation-aware training."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, TypeVar

import torch
from torch import Tensor


Source = TypeVar("Source")
Prepared = TypeVar("Prepared")


@dataclass(frozen=True)
class ThroughputConfig:
    prefetch_batches: int = 0
    pin_memory: bool = False
    non_blocking_transfer: bool = False
    cache_validation_batches: bool = False
    profile_stages: bool = False


def ordered_prefetch(
    source: Iterable[Source], prepare: Callable[[Source], Prepared], depth: int
) -> Iterator[Prepared]:
    """Prepare a bounded lookahead on one worker without changing source order."""
    if depth < 0:
        raise ValueError("prefetch depth must be non-negative")
    if depth == 0:
        yield from (prepare(value) for value in source)
        return

    executor = ThreadPoolExecutor(max_workers=1)
    futures: deque[Future[Prepared]] = deque()
    iterator = iter(source)
    try:
        for _ in range(depth):
            try:
                futures.append(executor.submit(prepare, next(iterator)))
            except StopIteration:
                break
        while futures:
            result = futures.popleft().result()
            try:
                futures.append(executor.submit(prepare, next(iterator)))
            except StopIteration:
                pass
            yield result
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def pin_tensor_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """Return a pinned tensor copy when CUDA supports asynchronous transfers."""
    if not torch.cuda.is_available():
        return dict(batch)
    return {
        name: value.pin_memory() if isinstance(value, Tensor) and value.device.type == "cpu" else value
        for name, value in batch.items()
    }
