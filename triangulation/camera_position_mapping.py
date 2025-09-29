#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/camera_position_mapping.py
Project: /workspace/code/triangulation
Created Date: Monday September 29th 2025
Author: Kaixu Chen
-----
Comment:
# Create a reusable Python module that maps camera IDs to extrinsic matrices (R, t).
# The module supports multiple ways to define orientation:
# - look-at target + up
# - yaw/pitch/roll (degrees, ZYX order)
# - rotation vector (Rodrigues) in radians
# It returns both world->camera and camera->world forms, plus projection matrices if K is provided.

Have a good code time :)
-----
Last Modified: Monday September 29th 2025 10:43:13 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Union
import numpy as np


@dataclass
class Extrinsics:
    """Extrinsic parameters for a single camera.

    R_wc, t_wc: world -> camera (cv2/colmap style: X_cam = R*X_world + t)
    R_cw, t_cw: camera -> world (X_world = R^T * (X_cam - t))
    C: camera center in world coordinates (== t_cw)
    """

    R_wc: np.ndarray  # (3,3)
    t_wc: np.ndarray  # (3,)
    R_cw: np.ndarray  # (3,3)
    t_cw: np.ndarray  # (3,)
    C: np.ndarray  # (3,)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / n


def rodrigues_to_R(rvec: Iterable[float]) -> np.ndarray:
    """Rodrigues rotation vector (radians) -> rotation matrix (3x3)."""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def ypr_deg_to_R(
    yaw_deg: float, pitch_deg: float, roll_deg: float, order: str = "ZYX"
) -> np.ndarray:
    """Convert yaw/pitch/roll in degrees to rotation matrix.
    Default order ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)

    # multiply in the provided order
    mapping = {"X": Rx, "Y": Ry, "Z": Rz}
    R = np.eye(3)
    for ax in order:
        R = mapping[ax] @ R
    return R


def lookat_to_Rt(
    cam_pos: Iterable[float],
    target: Iterable[float],
    up: Iterable[float] = (0, 0, 1),
    cv_convention: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build world->camera (R, t) from a camera position, look-at target and up vector.

    If cv_convention=True: camera's +Z looks forward, +X right, +Y down (OpenCV pinhole).
    If False (OpenGL style): camera's -Z looks forward, +X right, +Y up.
    """
    C = np.asarray(cam_pos, dtype=float).reshape(3)
    T = np.asarray(target, dtype=float).reshape(3)
    up = _normalize(np.asarray(up, dtype=float).reshape(3))

    if cv_convention:
        z_cam = _normalize(T - C)  # forward (+Z)
        x_cam = _normalize(np.cross(z_cam, up))  # right
        y_cam = np.cross(x_cam, z_cam)  # down-ish
    else:
        z_cam = _normalize(C - T)  # forward (-Z in OpenGL)
        x_cam = _normalize(np.cross(up, z_cam))
        y_cam = np.cross(z_cam, x_cam)

    R_cw = np.stack([x_cam, y_cam, z_cam], axis=0)  # camera axes expressed in world
    R_wc = R_cw.T
    t_wc = -R_wc @ C
    t_cw = C
    return R_wc, t_wc


def compose_extrinsics_from(
    cam_pos: Iterable[float],
    orientation: Dict[str, Union[Tuple[float, float, float], float]],
    orientation_mode: str = "lookat",
    up: Iterable[float] = (0, 0, 1),
) -> Extrinsics:
    """Create Extrinsics from different orientation specs.

    orientation_mode:
      - 'lookat': orientation must contain key 'target' -> (x,y,z)
      - 'ypr': orientation must contain ('yaw','pitch','roll') in degrees (ZYX)
      - 'rodrigues': orientation must contain 'rvec' (rx,ry,rz) radians, rotation from world to camera
    """
    C = np.asarray(cam_pos, dtype=float).reshape(3)
    if orientation_mode == "lookat":
        R_wc, t_wc = lookat_to_Rt(C, orientation["target"], up=up, cv_convention=True)
    elif orientation_mode == "ypr":
        R_cw = ypr_deg_to_R(
            orientation["yaw"], orientation["pitch"], orientation["roll"], order="ZYX"
        )
        R_wc = R_cw.T
        t_wc = -R_wc @ C
    elif orientation_mode == "rodrigues":
        R_wc = rodrigues_to_R(orientation["rvec"])
        t_wc = -R_wc @ C
    else:
        raise ValueError(f"Unknown orientation_mode: {orientation_mode}")

    R_cw = R_wc.T
    t_cw = C
    return Extrinsics(R_wc=R_wc, t_wc=t_wc, R_cw=R_cw, t_cw=t_cw, C=C)


def build_extrinsics_map(
    camera_layout: Dict[int, Dict],
    default_mode: str = "lookat",
    default_up: Iterable[float] = (0, 0, 1),
) -> Dict[int, Extrinsics]:
    """Given a layout dict (id -> spec), produce a map id -> Extrinsics.

    Each spec must contain:
    - 'pos': (x,y,z) camera center in world
    - and one of:
        * 'target': (x,y,z)  # used with lookat
        * 'ypr': (yaw_deg, pitch_deg, roll_deg)
        * 'rvec': (rx,ry,rz) radians (Rodrigues)

    Example camera_layout entry:
        1: {'pos': (0, -3, 1.5), 'target': (0, 0, 1.5)}
        2: {'pos': (3, 0, 1.5), 'ypr': (90, 0, 0)}
    """
    out: Dict[int, Extrinsics] = {}
    for cid, spec in camera_layout.items():
        pos = spec["pos"]
        if "target" in spec:
            ext = compose_extrinsics_from(
                pos,
                {"target": spec["target"]},
                orientation_mode="lookat",
                up=spec.get("up", default_up),
            )
        elif "ypr" in spec:
            yaw, pitch, roll = spec["ypr"]
            ext = compose_extrinsics_from(
                pos,
                {"yaw": yaw, "pitch": pitch, "roll": roll},
                orientation_mode="ypr",
                up=spec.get("up", default_up),
            )
        elif "rvec" in spec:
            ext = compose_extrinsics_from(
                pos,
                {"rvec": spec["rvec"]},
                orientation_mode="rodrigues",
                up=spec.get("up", default_up),
            )
        else:
            raise ValueError(
                f"Camera {cid} spec must contain one of: 'target', 'ypr', or 'rvec'"
            )
        out[cid] = ext
    return out


def make_projection_matrices(
    extrinsics_map: Dict[int, Extrinsics], K_map: Optional[Dict[int, np.ndarray]] = None
) -> Dict[int, np.ndarray]:
    """Optionally build projection matrices P = K [R|t] for each camera id.
    If K_map is None, use identity intrinsics.
    """
    P_map: Dict[int, np.ndarray] = {}
    for cid, ext in extrinsics_map.items():
        K = np.eye(3) if K_map is None else K_map[cid]
        Rt = np.hstack([ext.R_wc, ext.t_wc.reshape(3, 1)])
        P_map[cid] = K @ Rt
    return P_map


# -------------------------
# Example TEMPLATE (edit this with your actual positions/orientations)
# Coordinate frame assumption:
# - World: Z up, X right, Y forward (you can adapt to your convention)
# - Cameras follow OpenCV pinhole: +Z forward, +X right, +Y down
# -------------------------

CAMERA_LAYOUT_EXAMPLE: Dict[int, Dict] = {
    # Cam 1: at (-3, 0, 1.5) looking at the origin (0,0,1.5)
    1: {"pos": (-3.0, 0.0, 1.5), "target": (0.0, 0.0, 1.5)},
    # Cam 2: at ( 3, 0, 1.5) looking at the origin
    2: {"pos": (3.0, 0.0, 1.5), "target": (0.0, 0.0, 1.5)},
    # Cam 3: at (0, -3, 1.5) looking at the origin
    3: {"pos": (0.0, -3.0, 1.5), "target": (0.0, 0.0, 1.5)},
    # Cam 4: at (0,  3, 1.5) with explicit yaw/pitch/roll (deg): yaw=180 faces -Y, pitch=0, roll=0
    4: {"pos": (0.0, 3.0, 1.5), "ypr": (180.0, 0.0, 0.0)},
}

if __name__ == "__main__":
    # Build extrinsics map from the example layout
    extr_map = build_extrinsics_map(CAMERA_LAYOUT_EXAMPLE)

    # Print a compact summary
    for cid, ext in sorted(extr_map.items()):
        print(f"[Cam {cid}]")
        print(
            "R_wc=\n",
            np.array2string(ext.R_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "t_wc=",
            np.array2string(ext.t_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "C(w) =",
            np.array2string(ext.C, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print()
