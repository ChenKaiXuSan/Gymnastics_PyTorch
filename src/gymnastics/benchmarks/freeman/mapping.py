"""Versioned MHR70-to-FreeMan COCO17 joint mapping."""

from __future__ import annotations

import numpy as np

from gymnastics.common.skeletons.mhr70 import MHR70_INDEX

from .schema import MappedPose


FREEMAN_COCO17_NAMES = (
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-wrist",
    "right-wrist",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
)
MAPPING_VERSION = "mhr70_to_freeman_coco17_v1"
_MHR_INDICES = tuple(MHR70_INDEX[name] for name in FREEMAN_COCO17_NAMES)


def map_mhr70_to_freeman(
    points: np.ndarray,
    valid: np.ndarray | None = None,
) -> MappedPose:
    """Select exact homologous MHR70 joints in official FreeMan order."""
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (70, 3):
        raise ValueError("MHR70 points must have shape [T,70,3]")
    finite = np.isfinite(source).all(axis=-1)
    nonzero = np.any(source != 0, axis=-1)
    source_valid = finite & nonzero
    if valid is not None:
        supplied = np.asarray(valid, dtype=bool)
        if supplied.shape != source.shape[:2]:
            raise ValueError("MHR70 valid mask must have shape [T,70]")
        source_valid &= supplied
    mapped_valid = source_valid[:, _MHR_INDICES]
    mapped = np.array(source[:, _MHR_INDICES], copy=True)
    mapped[~mapped_valid] = 0
    return MappedPose(mapped, mapped_valid, FREEMAN_COCO17_NAMES)
