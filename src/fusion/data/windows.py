"""Fixed-length training windows over phase-normalised dual-view samples.

This is the single place where a :class:`DualViewSample` becomes the tensor
dictionary consumed by the model.  Every DataModule uses it, so dataset
adapters never build tensors themselves.

Windowing rule:
    * If phase normalisation is enabled and the sample has cycles, the
      sample is first resampled to ``samples_per_cycle`` samples per cycle
      (:func:`~fusion.phase.normalize_sample_to_phase`).
    * Windows of ``length = round(num_cycles * samples_per_cycle)`` samples
      are cut with a stride; the last window is aligned to the end so no
      sample is lost.  Shorter sequences produce one zero-padded window
      (``frame_mask`` false on padding).
    * Physical timestamps are kept; ``delta_t[t]`` is the interval to the
      previous sample (``delta_t[0]`` copies ``delta_t[1]``).

Corruption:
    When a :class:`CorruptionConfig` is supplied the window additionally
    carries the clean copy of both views (``clean_*``) and the corruption
    masks; the seed is a deterministic function of ``(seed, epoch, window_id)``
    so training is reproducible and a validation suite replays the same
    damage every epoch (``set_epoch`` changes it for training).

Output dictionary (``T = length``, ``J`` joints):
    pose_a, pose_b [T, J, 3]      valid_a, valid_b [T, J]
    frame_mask [T]                delta_t [T]           timestamps [T]
    phase [T]  phase_valid [T]    cycle_index [T]   half_index [T]
    reference [T, J, 3]           reference_valid [T, J]   (zeros/false if absent)
    cycle_target [T, J, 3]        cycle_confidence [T, J]  cycle_dispersion [T, J]
                                  (leave-one-cycle-out target and its statistics, see cycle_target.py)
    clean_a, clean_b, clean_valid_a, clean_valid_b, corruption_mask_a/b (if corrupted)
    window_start (scalar)  dataset, subject_id, sequence_id, window_id (str)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..corruptions import CorruptionConfig, corrupt_window, stable_seed
from ..cycle_target import CrossCycleTarget, CycleTargetConfig, cross_cycle_target
from ..phase import normalize_sample_to_phase, phase_from_cycle_bounds
from ..sample import DualViewSample
from ..skeleton import CommonSkeleton


@dataclass(frozen=True)
class WindowConfig:
    """Window geometry shared by the model and the DataModules.

    Attributes:
        num_cycles: Cycles per window (long-term context); may be fractional
            (0.5, 1, 2, ...).  ``None`` means *full context*: one window per
            sequence, padded to the longest loaded sequence.
        samples_per_cycle: Samples per normalised cycle ``S``.
        train_stride: Stride between training windows in samples
            (``None`` = ``S // 2``).
        eval_stride: Stride between evaluation windows (``None`` = window length).
    """

    num_cycles: float | None = 2.0
    samples_per_cycle: int = 64
    train_stride: int | None = None
    eval_stride: int | None = None

    def __post_init__(self) -> None:
        if (self.num_cycles is not None and self.num_cycles <= 0) or self.samples_per_cycle < 2:
            raise ValueError("num_cycles must be positive (or None) and samples_per_cycle >= 2")
        if (self.train_stride is not None and self.train_stride < 1) or (self.eval_stride is not None and self.eval_stride < 1):
            raise ValueError("strides must be positive")

    @property
    def full_context(self) -> bool:
        """Whether every sequence becomes exactly one window."""
        return self.num_cycles is None

    @property
    def length(self) -> int | None:
        """Window length in samples (``None`` for full context)."""
        if self.num_cycles is None:
            return None
        return max(2, int(round(self.num_cycles * self.samples_per_cycle)))

    def stride(self, split: str, length: int) -> int:
        """Stride for ``split`` given the effective window ``length``."""
        if self.full_context:
            return length
        if split == "train":
            return self.train_stride or max(1, self.samples_per_cycle // 2)
        return self.eval_stride or length


def window_starts(frames: int, length: int, stride: int) -> tuple[int, ...]:
    """Start indices covering ``frames`` with windows of ``length`` and ``stride``."""
    if frames <= length:
        return (0,)
    starts = list(range(0, frames - length + 1, stride))
    if starts[-1] != frames - length:
        starts.append(frames - length)
    return tuple(starts)


@dataclass(frozen=True)
class _Window:
    sample: DualViewSample
    start: int
    phase: np.ndarray
    phase_valid: np.ndarray
    cycle_index: np.ndarray
    half_index: np.ndarray
    cycle_target: np.ndarray
    cycle_confidence: np.ndarray
    cycle_dispersion: np.ndarray


class CycleWindowDataset(Dataset[dict[str, Any]]):
    """Windows over (phase-normalised) samples with optional corruption.

    Attributes:
        samples: The (possibly resampled) samples backing the windows.
        length: Window length in samples.
    """

    def __init__(
        self,
        samples: Sequence[DualViewSample],
        *,
        skeleton: CommonSkeleton,
        window: WindowConfig,
        split: str,
        phase_normalize: bool = True,
        corruption: CorruptionConfig | None = None,
        seed: int = 0,
        epoch: int = 0,
        cycle_target: CycleTargetConfig | None = None,
    ) -> None:
        if split not in {"train", "val", "test", "predict"}:
            raise ValueError("split must be train, val, test or predict")
        self.skeleton = skeleton
        self.window = window
        self.split = split
        self.corruption = corruption if (corruption is not None and corruption.enabled) else None
        self.cycle_target = cycle_target or CycleTargetConfig(enabled=False)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.samples: list[DualViewSample] = []
        self._windows: list[_Window] = []
        for sample in samples:
            if sample.num_joints != skeleton.num_joints or sample.joint_names != skeleton.joint_names:
                raise ValueError(f"sample {sample.key} does not use skeleton {skeleton.name}")
            self.samples.append(normalize_sample_to_phase(sample, window.samples_per_cycle) if phase_normalize else sample)
        # Full context: one window per sequence, padded to the longest sequence.
        self.length = window.length if window.length is not None else max((s.num_frames for s in self.samples), default=2)
        stride = window.stride(split, self.length)
        for prepared in self.samples:
            phase, phase_valid, cycle_index, half_index = phase_from_cycle_bounds(prepared.num_frames, prepared.cycle_bounds, prepared.cycle_mids)
            if self.cycle_target.enabled and phase_normalize and len(prepared.cycle_bounds) >= 2:
                reference = cross_cycle_target(prepared.view_a, prepared.view_b, prepared.valid_a, prepared.valid_b, prepared.cycle_bounds, samples_per_cycle=window.samples_per_cycle, config=self.cycle_target)
            else:
                reference = CrossCycleTarget.empty(prepared.num_frames, prepared.num_joints)
            for start in window_starts(prepared.num_frames, self.length, stride):
                self._windows.append(_Window(prepared, start, phase, phase_valid, cycle_index, half_index, reference.target, reference.confidence, reference.dispersion))

    def set_epoch(self, epoch: int) -> None:
        """Change the corruption seed for a new training epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._windows)

    def window_id(self, index: int) -> str:
        """Stable identifier of window ``index``."""
        descriptor = self._windows[index]
        return f"{descriptor.sample.key}/{descriptor.start}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        descriptor = self._windows[index]
        sample, start, length = descriptor.sample, descriptor.start, self.length
        available = min(length, sample.num_frames - start)
        joints = sample.num_joints
        stop = start + available

        def pad_points(values: np.ndarray) -> torch.Tensor:
            out = torch.zeros((length, joints, 3), dtype=torch.float32)
            out[:available] = torch.from_numpy(np.array(values[start:stop], dtype=np.float32, copy=True))
            return out

        def pad_mask(values: np.ndarray) -> torch.Tensor:
            out = torch.zeros((length, joints), dtype=torch.bool)
            out[:available] = torch.from_numpy(np.array(values[start:stop], dtype=bool, copy=True))
            return out

        timestamps = torch.zeros(length, dtype=torch.float64)
        timestamps[:available] = torch.from_numpy(np.array(sample.timestamps[start:stop], copy=True))
        delta_t = torch.zeros(length, dtype=torch.float32)
        if available > 1:
            delta_t[1:available] = torch.from_numpy(np.diff(sample.timestamps[start:stop]).astype(np.float32))
            delta_t[0] = delta_t[1]
        elif available == 1:
            fps = float(sample.metadata.get("fps", 0.0)) if "fps" in sample.metadata else 0.0
            delta_t[0] = 1.0 / fps if fps > 0 else 1.0
        frame_mask = torch.zeros(length, dtype=torch.bool)
        frame_mask[:available] = True
        phase = torch.zeros(length, dtype=torch.float32)
        phase[:available] = torch.from_numpy(descriptor.phase[start:stop].astype(np.float32))
        phase_valid = torch.zeros(length, dtype=torch.bool)
        phase_valid[:available] = torch.from_numpy(descriptor.phase_valid[start:stop])
        cycle_index = torch.full((length,), -1, dtype=torch.int64)
        cycle_index[:available] = torch.from_numpy(descriptor.cycle_index[start:stop])
        half_index = torch.full((length,), -1, dtype=torch.int64)
        half_index[:available] = torch.from_numpy(descriptor.half_index[start:stop])

        item: dict[str, Any] = {
            "pose_a": pad_points(sample.view_a),
            "pose_b": pad_points(sample.view_b),
            "valid_a": pad_mask(sample.valid_a),
            "valid_b": pad_mask(sample.valid_b),
            "frame_mask": frame_mask,
            "delta_t": delta_t,
            "timestamps": timestamps,
            "phase": phase,
            "phase_valid": phase_valid,
            "cycle_index": cycle_index,
            "half_index": half_index,
            "reference": pad_points(sample.reference) if sample.reference is not None else torch.zeros((length, joints, 3)),
            "reference_valid": pad_mask(sample.reference_valid) if sample.reference_valid is not None else torch.zeros((length, joints), dtype=torch.bool),
            "reference_canonical": torch.tensor(bool(sample.reference_canonical)),
            "cycle_target": pad_points(descriptor.cycle_target),
            "cycle_confidence": torch.from_numpy(np.pad(descriptor.cycle_confidence[start:stop], ((0, length - available), (0, 0))).astype(np.float32)),
            # Dispersion is padded with inf (undefined) so the dead zone never activates on padding.
            "cycle_dispersion": torch.from_numpy(np.pad(descriptor.cycle_dispersion[start:stop], ((0, length - available), (0, 0)), constant_values=np.inf).astype(np.float32)),
            "window_start": torch.tensor(start, dtype=torch.int64),
            "dataset": sample.dataset,
            "subject_id": sample.subject_id,
            "sequence_id": sample.sequence_id,
            "window_id": self.window_id(index),
        }
        if self.corruption is not None:
            corrupted = corrupt_window(
                item["pose_a"],
                item["pose_b"],
                item["valid_a"],
                item["valid_b"],
                seed=stable_seed(self.seed, item["window_id"], self.epoch),
                config=self.corruption,
                skeleton=self.skeleton,
            )
            item["clean_a"], item["clean_b"] = item["pose_a"], item["pose_b"]
            item["clean_valid_a"], item["clean_valid_b"] = item["valid_a"], item["valid_b"]
            item["pose_a"], item["pose_b"] = corrupted["pose_a"], corrupted["pose_b"]
            item["valid_a"], item["valid_b"] = corrupted["valid_a"], corrupted["valid_b"]
            item["corruption_mask_a"], item["corruption_mask_b"] = corrupted["corruption_mask_a"], corrupted["corruption_mask_b"]
        return item
