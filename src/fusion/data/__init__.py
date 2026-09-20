"""Dataset adapters and shared windowing for cycle-aware fusion.

Three independent dataset entry points convert their raw material into the
unified :class:`~fusion.sample.DualViewSample`:

    ``gymnastics``  private two-camera gymnastics recordings (cycles annotated)
    ``freeman``     public FreeMan release, two selected views per session
    ``unity``       synthetic Unity benchmark, two virtual cameras

plus a ``synthetic`` DataModule that generates periodic motion for smoke
tests and continuous integration.  Every DataModule derives from
:class:`~fusion.data.base.DualViewDataModule`, which
owns subject-disjoint splitting, phase normalisation, windowing, corruption
and the DataLoaders; the adapters only implement ``load_samples``.

Dataset-specific logic never leaks into the shared code: each adapter's
module docstring documents its source layout, skeleton, coordinate frame,
synchronisation, ground truth and cycle availability.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import DataConfig, DualViewDataModule, SplitSpec
from .windows import CycleWindowDataset, WindowConfig


def build_datamodule(config: Mapping[str, Any]) -> DualViewDataModule:
    """Instantiate the DataModule named by ``config["name"]``.

    Args:
        config: The ``data`` section of the Hydra configuration.

    Returns:
        The DataModule for that dataset.

    Raises:
        ValueError: If the name is unknown.
    """
    name = str(config["name"])
    if name == "synthetic":
        from .synthetic import SyntheticDataModule

        return SyntheticDataModule(config)
    if name == "gymnastics":
        from .gymnastics import GymnasticsDataModule

        return GymnasticsDataModule(config)
    if name == "freeman":
        from .freeman import FreeManDataModule

        return FreeManDataModule(config)
    if name == "unity":
        from .unity import UnityDataModule

        return UnityDataModule(config)
    raise ValueError(f"unknown dataset {name!r}; expected synthetic, gymnastics, freeman or unity")


__all__ = [
    "CycleWindowDataset",
    "DataConfig",
    "DualViewDataModule",
    "SplitSpec",
    "WindowConfig",
    "build_datamodule",
]
