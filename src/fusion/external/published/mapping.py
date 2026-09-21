"""Joint-layout conversions between MHR70, COCO17 and Human3.6M-17."""

from __future__ import annotations

import numpy as np

from common.skeletons.mhr70 import MHR70_INDEX

# COCO17 order (VideoPose3D ``detectron_coco`` input layout).
COCO17_NAMES: tuple[str, ...] = (
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
)
COCO17_FROM_MHR70: tuple[int, ...] = tuple(MHR70_INDEX[name] for name in COCO17_NAMES)
COCO17_LEFT: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13, 15)
COCO17_RIGHT: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16)

# Human3.6M 17-joint layout (Martinez et al. / VideoPose3D output).
H36M17_NAMES: tuple[str, ...] = (
    "pelvis", "right-hip", "right-knee", "right-ankle",
    "left-hip", "left-knee", "left-ankle",
    "spine", "thorax", "neck", "head",
    "left-shoulder", "left-elbow", "left-wrist",
    "right-shoulder", "right-elbow", "right-wrist",
)
H36M17_LEFT: tuple[int, ...] = (4, 5, 6, 11, 12, 13)
H36M17_RIGHT: tuple[int, ...] = (1, 2, 3, 14, 15, 16)
# H36M joints with a homologous MHR70 joint ("neck" is H36M's "Neck/Nose"
# point at the base of the head, matched to MHR70's neck; pelvis, spine,
# thorax and head have no MHR70 counterpart and are not evaluated).
H36M17_TO_MHR70: dict[int, int] = {
    1: MHR70_INDEX["right-hip"], 2: MHR70_INDEX["right-knee"], 3: MHR70_INDEX["right-ankle"],
    4: MHR70_INDEX["left-hip"], 5: MHR70_INDEX["left-knee"], 6: MHR70_INDEX["left-ankle"],
    9: MHR70_INDEX["neck"],
    11: MHR70_INDEX["left-shoulder"], 12: MHR70_INDEX["left-elbow"], 13: MHR70_INDEX["left-wrist"],
    14: MHR70_INDEX["right-shoulder"], 15: MHR70_INDEX["right-elbow"], 16: MHR70_INDEX["right-wrist"],
}


def mhr70_to_coco17_2d(points: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``[T, 70, 2]`` -> ``[T, 17, 2]`` COCO17 image keypoints (+ validity)."""
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (70, 2):
        raise ValueError("expected [T, 70, 2] MHR70 image keypoints")
    index = list(COCO17_FROM_MHR70)
    mapped = source[:, index]
    ok = np.isfinite(mapped).all(axis=-1)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)[:, index]
    return mapped, ok


def h36m17_to_mhr70(points: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scatter ``[T, 17, 3]`` H36M joints into ``[T, 70, 3]`` MHR70 positions with a validity mask."""
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (17, 3):
        raise ValueError("expected [T, 17, 3] H36M joints")
    target = np.zeros((source.shape[0], 70, 3), dtype=np.float32)
    target_valid = np.zeros((source.shape[0], 70), dtype=bool)
    for h36m_index, mhr_index in H36M17_TO_MHR70.items():
        ok = np.isfinite(source[:, h36m_index]).all(axis=-1)
        if valid is not None:
            ok &= np.asarray(valid, dtype=bool)[:, h36m_index]
        target[:, mhr_index] = np.where(ok[:, None], source[:, h36m_index], 0.0)
        target_valid[:, mhr_index] = ok
    return target, target_valid
