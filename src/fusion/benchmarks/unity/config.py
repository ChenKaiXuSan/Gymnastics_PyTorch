"""Configuration and run-context helpers shared by the Unity benchmark stages."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import (
    Mapping,
    Sequence,
)


DEFAULT_CONFIG = Path("src/configs/benchmarks/unity.yaml")


DEFAULT_SUPERVISED_CONFIG = Path("src/configs/benchmarks/unity_supervised.yaml")


def _load_config(path: Path) -> Mapping[str, object]:
    """Load the benchmark YAML, resolving ``${oc.env:...}`` interpolations."""
    from omegaconf import OmegaConf

    import common.paths  # noqa: F401  (exports GYMNASTICS_DATA_ROOT when unset)

    payload = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Unity benchmark config must be a mapping")
    return payload


def _paths(config: Mapping[str, object]) -> tuple[Path, Path]:
    raw = config.get("paths")
    if not isinstance(raw, Mapping):
        raise ValueError("Unity benchmark config requires paths")
    return Path(str(raw["dataset_root"])), Path(str(raw["output_root"]))


def _path_value(config: Mapping[str, object], name: str) -> Path:
    raw = config.get("paths")
    if not isinstance(raw, Mapping) or name not in raw:
        raise ValueError(f"Unity benchmark config requires paths.{name}")
    return Path(str(raw[name]))


def _checkpoints(
    config: Mapping[str, object], wanted: Sequence[str] | None
) -> dict[str, Path]:
    raw = config.get("checkpoints", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Unity benchmark checkpoints must be a mapping")
    selected = tuple(wanted) if wanted else tuple(str(key) for key in raw)
    missing = [name for name in selected if name not in raw]
    if missing:
        raise ValueError(f"unknown rotation-aware ablations: {missing}")
    return {name: Path(str(raw[name])) for name in selected}


def _data_fps(config: Mapping[str, object]) -> float:
    raw = config.get("data", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Unity benchmark data config must be a mapping")
    return float(raw.get("fps", 60.0))


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _required_mapping(
    config: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Unity supervised config requires {name}")
    return value

