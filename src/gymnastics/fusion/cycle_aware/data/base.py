"""Abstract LightningDataModule shared by every dataset adapter.

Responsibilities kept here so adapters stay tiny and dataset-specific:

* subject-disjoint splitting (``SplitSpec``) with an overlap check,
* phase normalisation and windowing through :class:`CycleWindowDataset`,
* training-time corruption (and a fixed-seed replay for validation),
* DataLoader construction with the project collate function,
* optional reference attachment restricted to evaluation splits.

An adapter implements :meth:`DualViewDataModule.load_samples` returning
*all* samples and :meth:`DualViewDataModule.default_split` returning the
subject lists.  Nothing else is dataset specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..corruptions import CorruptionConfig
from ..sample import DualViewSample, collate_fusion_batch
from ..skeleton import CommonSkeleton, build_common_skeleton
from .sample_cache import cache_key, load_samples, save_samples
from .windows import CycleWindowDataset, WindowConfig


@dataclass(frozen=True)
class SplitSpec:
    """Subject-disjoint split membership.

    Attributes:
        train: Subject ids used for fitting.
        val: Subject ids used for model selection.
        test: Held-out subject ids.
    """

    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        groups = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"subjects appear in both {left} and {right}: {sorted(overlap)}")
        object.__setattr__(self, "train", tuple(str(s) for s in self.train))
        object.__setattr__(self, "val", tuple(str(s) for s in self.val))
        object.__setattr__(self, "test", tuple(str(s) for s in self.test))

    def members(self, split: str) -> tuple[str, ...]:
        """Subject ids of ``split``."""
        return getattr(self, split)


@dataclass(frozen=True)
class DataConfig:
    """Fields common to every DataModule (``configs/cycle_aware/data/*.yaml``).

    Attributes:
        name: Dataset identifier.
        skeleton: Common skeleton variant.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        seed: Seed for corruption and shuffling.
        window: Window geometry.
        phase_normalize: Resample cycles to ``samples_per_cycle`` samples.
        attach_reference: Attach reference poses to val/test samples when the
            adapter can provide them (never used in training).
        corruption: Training corruption parameters.
        validate_with_corruption: Replay a fixed corruption on validation
            windows so the recovery loss is comparable across epochs.
        split: Explicit subject split; empty entries fall back to the adapter
            default.
        max_subjects_per_split: Optional cap for quick experiments.
        cache_dir: Optional directory for the converted-sample cache
            (:mod:`.sample_cache`); ``None`` disables caching.
        options: Adapter-specific settings (documented by each adapter).
    """

    name: str
    skeleton: str = "mhr70_major"
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 0
    window: WindowConfig = field(default_factory=WindowConfig)
    phase_normalize: bool = True
    attach_reference: bool = True
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    validate_with_corruption: bool = True
    split: SplitSpec = field(default_factory=SplitSpec)
    max_subjects_per_split: int | None = None
    cache_dir: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.max_subjects_per_split is not None and self.max_subjects_per_split < 1:
            raise ValueError("max_subjects_per_split must be positive or None")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "DataConfig") -> "DataConfig":
        """Build from the Hydra ``data`` node; unknown top-level keys go to ``options``."""
        if isinstance(value, cls):
            return value
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)  # type: ignore[assignment]
        except ImportError:  # pragma: no cover
            pass
        payload = dict(value)
        window = WindowConfig(**dict(payload.pop("window", {}) or {}))
        corruption = CorruptionConfig.from_mapping(payload.pop("corruption", None))
        split_payload = dict(payload.pop("split", {}) or {})
        split = SplitSpec(**{k: tuple(v or ()) for k, v in split_payload.items()})
        options = dict(payload.pop("options", {}) or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for key in list(payload):
            if key not in known:
                options[key] = payload.pop(key)
        return cls(window=window, corruption=corruption, split=split, options=options, **payload)

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary representation."""
        payload = asdict(self)
        payload["options"] = dict(self.options)
        return payload


class DualViewDataModule(pl.LightningDataModule, ABC):
    """Base DataModule: adapters implement ``load_samples`` and ``default_split``.

    Attributes:
        config: The :class:`DataConfig`.
        skeleton: Common skeleton used for every sample.
        samples: All loaded samples (after ``setup``).
        split: Effective subject split (after ``setup``).
    """

    def __init__(self, config: DataConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = DataConfig.from_mapping(config)
        self.skeleton: CommonSkeleton = build_common_skeleton(self.config.skeleton)
        self.samples: list[DualViewSample] = []
        self.split: SplitSpec = SplitSpec()
        self._datasets: dict[str, CycleWindowDataset] = {}

    @abstractmethod
    def load_samples(self) -> Sequence[DualViewSample]:
        """Load every sample of the dataset in the common representation."""

    @abstractmethod
    def default_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        """Return the dataset's default subject-disjoint split."""

    def _effective_split(self, samples: Sequence[DualViewSample]) -> SplitSpec:
        default = self.default_split(samples)
        requested = self.config.split
        split = SplitSpec(
            train=requested.train or default.train,
            val=requested.val or default.val,
            test=requested.test or default.test,
        )
        cap = self.config.max_subjects_per_split
        if cap is not None:
            split = SplitSpec(train=split.train[:cap], val=split.val[:cap], test=split.test[:cap])
        known = {sample.subject_id for sample in samples}
        unknown = sorted((set(split.train) | set(split.val) | set(split.test)) - known)
        if unknown:
            raise ValueError(f"split references unknown subjects: {unknown}")
        return split

    def _samples_for(self, split: str) -> list[DualViewSample]:
        members = set(self.split.members(split))
        selected = [sample for sample in self.samples if sample.subject_id in members]
        if split == "train" or not self.config.attach_reference:
            # References are evaluation-only: strip them from training samples.
            selected = [replace(sample, reference=None, reference_valid=None) if sample.reference is not None else sample for sample in selected]
        return selected

    def _cache_directory(self) -> Path | None:
        if not self.config.cache_dir:
            return None
        from gymnastics.common.paths import PROJECT_ROOT

        root = Path(self.config.cache_dir)
        root = root if root.is_absolute() else PROJECT_ROOT / root
        key = cache_key(self.config.name, self.config.skeleton, self.config.options, self.config.attach_reference)
        return root / f"{self.config.name}_{key}"

    def _load_or_convert(self) -> list[DualViewSample]:
        directory = self._cache_directory()
        if directory is not None:
            cached = load_samples(directory)
            if cached:
                return cached
        samples = list(self.load_samples())
        if samples and directory is not None:
            save_samples(directory, samples, config=self.config.to_dict())
        return samples

    def setup(self, stage: str | None = None) -> None:
        if not self.samples:
            self.samples = self._load_or_convert()
            if not self.samples:
                raise ValueError(f"{type(self).__name__} loaded no samples")
            self.split = self._effective_split(self.samples)
        wanted = {"fit": ("train", "val"), "validate": ("val",), "test": ("test",), "predict": ("test",)}.get(stage or "fit", ("train", "val", "test"))
        for split in wanted:
            if split in self._datasets:
                continue
            corruption = None
            if split == "train":
                corruption = self.config.corruption
            elif split == "val" and self.config.validate_with_corruption:
                corruption = self.config.corruption
            self._datasets[split] = CycleWindowDataset(
                self._samples_for(split),
                skeleton=self.skeleton,
                window=self.config.window,
                split=split,
                phase_normalize=self.config.phase_normalize,
                corruption=corruption,
                seed=self.config.seed,
            )

    def dataset(self, split: str) -> CycleWindowDataset:
        """Return the window dataset of ``split`` (after ``setup``)."""
        if split not in self._datasets:
            self.setup({"train": "fit", "val": "fit", "test": "test"}[split])
        return self._datasets[split]

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.dataset(split),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_fusion_batch,
            drop_last=False,
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch to the training dataset (re-seeds corruption)."""
        if "train" in self._datasets:
            self._datasets["train"].set_epoch(epoch)

    def summary(self) -> dict[str, Any]:
        """Human-readable description of the loaded data."""
        return {
            "dataset": self.config.name,
            "skeleton": self.skeleton.name,
            "subjects": {split: len(self.split.members(split)) for split in ("train", "val", "test")},
            "samples": len(self.samples),
            "windows": {split: len(ds) for split, ds in self._datasets.items()},
            "window_length": {split: ds.length for split, ds in self._datasets.items()} if self.config.window.full_context else self.config.window.length,
        }
