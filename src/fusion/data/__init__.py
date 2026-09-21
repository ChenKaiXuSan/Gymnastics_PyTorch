"""Dataset adapters and shared windowing for cycle-aware fusion.

Four independent dataset entry points convert their raw material into the
unified :class:`~fusion.sample.DualViewSample`:

    ``gymnastics``  private two-camera gymnastics recordings (cycles annotated)
    ``freeman``     public FreeMan release, two selected views per session
    ``unity``       synthetic Unity benchmark, two virtual cameras
    ``sportspose``  public SportsPose release, trials of one action as cycles

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


def build_datamodule(config: Mapping[str, Any], *, trial_transform: Any = None) -> DualViewDataModule:
    """Instantiate the DataModule named by ``config["name"]``.

    Args:
        config: The ``data`` section of the Hydra configuration.
        trial_transform: Optional hook replacing each loaded trial's views
            (published external baselines); ignored by the synthetic data.

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

        return GymnasticsDataModule(config, trial_transform=trial_transform)
    if name == "freeman":
        from .freeman import FreeManDataModule

        return FreeManDataModule(config, trial_transform=trial_transform)
    if name == "unity":
        from .unity import UnityDataModule

        return UnityDataModule(config, trial_transform=trial_transform)
    if name == "sportspose":
        from .sportspose import SportsPoseDataModule

        return SportsPoseDataModule(config, trial_transform=trial_transform)
    raise ValueError(f"unknown dataset {name!r}; expected synthetic, gymnastics, freeman, unity or sportspose")


__all__ = [
    "CycleWindowDataset",
    "DataConfig",
    "DualViewDataModule",
    "SplitSpec",
    "WindowConfig",
    "build_datamodule",
]
