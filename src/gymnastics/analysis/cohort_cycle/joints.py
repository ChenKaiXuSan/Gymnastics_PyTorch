"""Canonical joint selections shared with the classification pipeline."""

from __future__ import annotations

from gymnastics.classification.map_config import INDICES
from gymnastics.common.skeletons import MHR70_NAMES


MAJOR_JOINT_INDICES = tuple(int(index) for index in INDICES)
MAJOR_JOINT_NAMES = tuple(MHR70_NAMES[index] for index in MAJOR_JOINT_INDICES)

if len(MAJOR_JOINT_INDICES) != 20 or len(set(MAJOR_JOINT_INDICES)) != 20:
    raise RuntimeError("classification major-joint mapping must contain 20 joints")
