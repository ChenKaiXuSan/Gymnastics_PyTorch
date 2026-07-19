"""Subject-safe fixed windows over cached rotation-aware pose trials."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import PosePairTrial


@dataclass(frozen=True)
class WindowConfig:
    length: int = 128
    train_stride: int = 32
    eval_stride: int = 64

    def __post_init__(self) -> None:
        if self.length < 1 or self.train_stride < 1 or self.eval_stride < 1:
            raise ValueError("window length and strides must be positive")


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"split person overlap between {left} and {right}: {sorted(overlap)}")


def _person_id(sample: Any) -> str:
    if isinstance(sample, str | int):
        return str(sample)
    if isinstance(sample, Mapping):
        for key in ("person_id", "person", "subject_id"):
            if key in sample:
                return str(sample[key])
    raise ValueError("Fold entries must contain a person_id, person, or subject_id")


def build_split_manifest(fold_json: str | Path | Mapping[str, Any]) -> SplitManifest:
    """Extract only person membership from an existing fold JSON, never labels."""
    if isinstance(fold_json, (str, Path)):
        payload = json.loads(Path(fold_json).read_text(encoding="utf-8"))
    else:
        payload = fold_json
    if not isinstance(payload, Mapping):
        raise ValueError("fold JSON must be a mapping")

    def members(name: str) -> tuple[str, ...]:
        entries = payload.get(name, [])
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"fold {name} entries must be a sequence")
        return tuple(sorted({_person_id(entry) for entry in entries}))

    return SplitManifest(train=members("train"), val=members("val"), test=members("test"))


@dataclass(frozen=True)
class _WindowIndex:
    trial: PosePairTrial
    start: int


class PosePairWindowDataset(Dataset[dict[str, Any]]):
    """Manifest-bound CPU windows from cached ``PosePairTrial`` records."""

    def __init__(
        self,
        trials: Sequence[PosePairTrial],
        *,
        manifest: SplitManifest,
        split: str,
        config: WindowConfig | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.config = config or WindowConfig()
        included = set(getattr(manifest, split))
        unexpected = sorted({trial.person_id for trial in trials} - included)
        if unexpected:
            raise ValueError(f"trials are not members of the {split} split: {unexpected}")
        self.split = split
        self._windows: list[_WindowIndex] = []
        stride = self.config.train_stride if split == "train" else self.config.eval_stride
        for trial in trials:
            if trial.person_id not in included:
                continue
            starts = self._starts(len(trial.timestamps), self.config.length, stride)
            self._windows.extend(_WindowIndex(trial, start) for start in starts)

    @staticmethod
    def _starts(frames: int, length: int, stride: int) -> tuple[int, ...]:
        if frames <= length:
            return (0,)
        starts = list(range(0, frames - length + 1, stride))
        final = frames - length
        if starts[-1] != final:
            starts.append(final)
        return tuple(starts)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        descriptor = self._windows[index]
        trial, start, length = descriptor.trial, descriptor.start, self.config.length
        available = min(length, len(trial.timestamps) - start)
        joints = trial.face.shape[1]
        face = torch.zeros((length, joints, 3), dtype=torch.float32)
        side = torch.zeros_like(face)
        valid_face = torch.zeros((length, joints), dtype=torch.bool)
        valid_side = torch.zeros_like(valid_face)
        padding_mask = torch.zeros(length, dtype=torch.bool)
        timestamps = torch.zeros(length, dtype=torch.float64)
        dt = torch.zeros(length, dtype=torch.float32)
        face[:available] = torch.from_numpy(np.array(trial.face[start : start + available], copy=True))
        side[:available] = torch.from_numpy(np.array(trial.side[start : start + available], copy=True))
        valid_face[:available] = torch.from_numpy(np.array(trial.valid_face[start : start + available], copy=True))
        valid_side[:available] = torch.from_numpy(np.array(trial.valid_side[start : start + available], copy=True))
        padding_mask[:available] = True
        timestamps[:available] = torch.from_numpy(np.array(trial.timestamps[start : start + available], copy=True))
        if available:
            dt[0] = 1.0 / trial.fps
        if available > 1:
            dt[1:available] = torch.from_numpy(np.diff(trial.timestamps[start : start + available]).astype(np.float32))
        return {
            "face": face,
            "side": side,
            "valid_face": valid_face,
            "valid_side": valid_side,
            "padding_mask": padding_mask,
            "loss_mask": padding_mask.unsqueeze(-1) & valid_face & valid_side,
            "timestamps": timestamps,
            "fps": torch.tensor(trial.fps, dtype=torch.float32),
            "dt": dt,
            "complete_cycle": start == 0 and available == len(trial.timestamps),
            "person_id": trial.person_id,
            "trial_id": trial.trial_id,
            "window_start": start,
            "window_id": f"person_{trial.person_id}/{trial.trial_id}/{start}",
        }


def collate_pose_pair_windows(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields while retaining sample identifiers as ordered lists."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return batch
