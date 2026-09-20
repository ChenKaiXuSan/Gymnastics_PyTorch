"""Repository-root YAML configuration with OmegaConf dot-list overrides."""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: Path, overrides: Sequence[str] = ()) -> DictConfig:
    unsupported = [value for value in overrides if value.startswith("-")]
    if unsupported:
        raise ValueError(
            "configuration overrides use key=value syntax; unsupported options: "
            + ", ".join(unsupported)
        )
    base = OmegaConf.load(path)
    if not overrides:
        return base
    return OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
