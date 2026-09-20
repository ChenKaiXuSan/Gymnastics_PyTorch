"""Configuration loading for the cohort-cycle pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve OmegaConf environment/path interpolations."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"cohort-cycle config does not exist: {source}")
    loaded = OmegaConf.load(source)
    resolved = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("cohort-cycle config must be a mapping")
    return resolved
