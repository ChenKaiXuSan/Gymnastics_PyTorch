"""Canonical repository, configuration, runtime and data paths.

Every absolute path the project depends on is defined here once.

* Repository-relative roots are derived from this file's location.
* The gymnastics data root comes from ``GYMNASTICS_DATA_ROOT``. When the
  variable is unset, the first existing directory among the known machines is
  used and exported, so YAML files can interpolate
  ``${oc.env:GYMNASTICS_DATA_ROOT}`` without repeating a default path.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = SRC_ROOT / "configs"
LOCAL_ROOT = PROJECT_ROOT / "local"
RUN_ROOT = LOCAL_ROOT / "runs"
CHECKPOINT_ROOT = LOCAL_ROOT / "checkpoints"
CALIBRATION_INPUT_ROOT = LOCAL_ROOT / "calibration_inputs"

DATA_ROOT_ENV = "GYMNASTICS_DATA_ROOT"
# Known locations of the private dataset, in order of preference.
_KNOWN_DATA_ROOTS = (
    Path("/work/1/HP260146/chenkaixu/gymnastics"),  # HP260146 / Pegasus
    Path("/home/data/xchen/gymnastics"),  # lab workstation
)


def _resolve_data_root() -> Path:
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured:
        return Path(configured)
    for candidate in _KNOWN_DATA_ROOTS:
        if candidate.is_dir():
            return candidate
    return _KNOWN_DATA_ROOTS[-1]


DATA_ROOT = _resolve_data_root()
os.environ.setdefault(DATA_ROOT_ENV, str(DATA_ROOT))

RAW_VIDEO_ROOT = DATA_ROOT / "raw"
SAM3D_RESULTS_ROOT = DATA_ROOT / "sam3d_body_results"
SAM3D_PERSON_ROOT = SAM3D_RESULTS_ROOT / "person"
SAM3D_LOG_ROOT = SAM3D_RESULTS_ROOT / "logs"
TRIANGULATED_ROOT = DATA_ROOT / "sam3d_triangulated" / "person"
UNITY_BENCHMARK_ROOT = DATA_ROOT / "unity_benchmark"
