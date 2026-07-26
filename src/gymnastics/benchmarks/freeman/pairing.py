"""Geometry-only FreeMan camera pair selection."""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from .schema import FreeManCamera, FreeManSession, SelectedPair


def _camera_geometry(
    camera: FreeManCamera,
    up: np.ndarray,
    minimum_axis_norm: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    world_to_camera = cv2.Rodrigues(camera.rotation)[0]
    if not np.isfinite(world_to_camera).all() or not np.isclose(
        np.linalg.det(world_to_camera),
        1.0,
        atol=1e-5,
    ):
        return None
    center = -world_to_camera.T @ camera.translation
    optical_axis = world_to_camera.T @ np.array([0.0, 0.0, 1.0])
    horizontal = optical_axis - float(np.dot(optical_axis, up)) * up
    norm = float(np.linalg.norm(horizontal))
    if norm <= minimum_axis_norm:
        return None
    return center, horizontal / norm


def select_camera_pair(
    session: FreeManSession,
    *,
    target_angle_deg: float,
    world_up: np.ndarray,
    minimum_axis_norm: float = 1e-8,
) -> SelectedPair:
    """Choose the reproducible pair closest to horizontal orthogonality."""
    up = np.asarray(world_up, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) <= 1e-12:
        raise ValueError("world_up must be a finite non-zero 3-vector")
    if (
        not np.isfinite(target_angle_deg)
        or target_angle_deg < 0
        or target_angle_deg > 90
    ):
        raise ValueError("target_angle_deg must be within 0..90")
    if minimum_axis_norm <= 0:
        raise ValueError("minimum_axis_norm must be positive")
    up = up / np.linalg.norm(up)
    geometry = {
        name: resolved
        for name, camera in session.cameras.items()
        if (resolved := _camera_geometry(camera, up, minimum_axis_norm)) is not None
    }
    ranked: list[
        tuple[
            tuple[float, float, str, str],
            float,
            float,
            float,
            str,
            str,
        ]
    ] = []
    for left, right in combinations(sorted(geometry), 2):
        left_center, left_axis = geometry[left]
        right_center, right_axis = geometry[right]
        cosine = float(np.clip(abs(np.dot(left_axis, right_axis)), 0.0, 1.0))
        separation = float(np.degrees(np.arccos(cosine)))
        height_difference = float(
            abs(np.dot(left_center - right_center, up))
        )
        target_error = abs(separation - target_angle_deg)
        rank = (
            round(target_error, 12),
            round(height_difference, 12),
            left,
            right,
        )
        ranked.append(
            (rank, separation, target_error, height_difference, left, right)
        )
    if not ranked:
        raise ValueError(
            f"session {session.session_id} has fewer than two valid horizontal cameras"
        )
    _, separation, target_error, height_difference, view_a, view_b = min(
        ranked,
        key=lambda item: item[0],
    )
    return SelectedPair(
        session_id=session.session_id,
        view_a=view_a,
        view_b=view_b,
        reference_view=view_a,
        separation_deg=separation,
        target_error_deg=target_error,
        height_difference=height_difference,
    )
