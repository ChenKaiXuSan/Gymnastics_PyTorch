"""Combining two monocular 3D sequences without learning or a depth prior.

Published monocular lifters give one root-relative 3D pose per view; the
strict two-view row combines them with the simplest calibration-free rule:
per frame, view B is aligned onto view A by a similarity Procrustes fit over
the joints valid in both views, and the two are averaged (a joint valid in
only one view keeps that view). This is the published methods' own fusion
baseline (plain averaging after alignment), not the depth-aware rule.
"""

from __future__ import annotations

import numpy as np


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform ``s R x + t`` mapping ``source`` onto ``target`` (``[N, 3]`` each, N >= 3)."""
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    xs, xt = source - mean_s, target - mean_t
    covariance = xt.T @ xs / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = (xs ** 2).sum() / len(source)
    scale = float((singular * sign).sum() / max(variance, 1e-12))
    translation = mean_t - scale * rotation @ mean_s
    return scale, rotation, translation


def procrustes_average(pose_a: np.ndarray, valid_a: np.ndarray, pose_b: np.ndarray, valid_b: np.ndarray, *, min_joints: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame Procrustes-aligned average of two ``[T, J, 3]`` sequences.

    Returns:
        ``(fused, valid)`` with ``valid = valid_a | valid_b``; frames where the
        alignment cannot be fitted (fewer than ``min_joints`` shared joints)
        fall back to view A where valid, else view B.
    """
    pose_a = np.asarray(pose_a, dtype=np.float32)
    pose_b = np.asarray(pose_b, dtype=np.float32)
    valid_a = np.asarray(valid_a, dtype=bool)
    valid_b = np.asarray(valid_b, dtype=bool)
    if pose_a.shape != pose_b.shape or pose_a.ndim != 3 or pose_a.shape[-1] != 3:
        raise ValueError("both sequences must have shape [T, J, 3]")
    fused = np.zeros_like(pose_a)
    both = valid_a & valid_b
    for t in range(pose_a.shape[0]):
        shared = both[t]
        aligned_b = pose_b[t]
        if shared.sum() >= min_joints:
            scale, rotation, translation = umeyama(pose_b[t][shared].astype(np.float64), pose_a[t][shared].astype(np.float64))
            aligned_b = (scale * (rotation @ pose_b[t].astype(np.float64).T).T + translation).astype(np.float32)
        elif not shared.any():
            fused[t] = np.where(valid_a[t][:, None], pose_a[t], pose_b[t])
            continue
        fused[t] = np.where(shared[:, None], 0.5 * (pose_a[t] + aligned_b), np.where(valid_a[t][:, None], pose_a[t], aligned_b))
    valid = valid_a | valid_b
    fused = np.where(valid[..., None], fused, 0.0)
    return fused, valid
