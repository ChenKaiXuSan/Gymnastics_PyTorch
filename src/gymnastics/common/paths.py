"""Canonical repository and runtime paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
LOCAL_ROOT = PROJECT_ROOT / "local"
RUN_ROOT = LOCAL_ROOT / "runs"
CHECKPOINT_ROOT = LOCAL_ROOT / "checkpoints"
CALIBRATION_INPUT_ROOT = LOCAL_ROOT / "calibration_inputs"
