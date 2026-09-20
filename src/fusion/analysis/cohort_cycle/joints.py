"""Canonical major-joint selection for whole-body descriptors."""

from __future__ import annotations

from common.skeletons import MHR70_MAJOR_JOINT_INDICES, MHR70_NAMES


MAJOR_JOINT_INDICES = tuple(int(index) for index in MHR70_MAJOR_JOINT_INDICES)
MAJOR_JOINT_NAMES = tuple(MHR70_NAMES[index] for index in MAJOR_JOINT_INDICES)

if len(MAJOR_JOINT_INDICES) != 20 or len(set(MAJOR_JOINT_INDICES)) != 20:
    raise RuntimeError("major-joint mapping must contain 20 distinct joints")
