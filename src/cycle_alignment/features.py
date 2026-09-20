"""Body-frame features used to align the face and side views.

The right-hand angle ``theta`` in the pelvis-centred body frame is the
one-dimensional signal shared by offset estimation and cycle segmentation.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np

# -------------------- MHR70 indices --------------------


IDX: Dict[str, int] = {
    "lhip": 9,
    "rhip": 10,
    "lsho": 5,
    "rsho": 6,
    "neck": 69,
    "rwrist": 41,
    "rindex_tip": 25,
    "rmiddle_tip": 29,
    "rpinky_tip": 37,
}


# -------------------- utils --------------------
def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def smooth_1d(x: np.ndarray, win: int = 11) -> np.ndarray:
    win = int(win)
    win = max(3, win | 1)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")


# -------------------- body frame (world -> body) --------------------
def build_body_frame_from_mhr70(
    kpts: np.ndarray, idx: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    kpts: (T,J,3)
    return pelvis_world: (T,3)
           R_body_to_world: (T,3,3) columns [x(right), y(up), z(forward)] in world
    """
    lhip = kpts[:, idx["lhip"], :]
    rhip = kpts[:, idx["rhip"], :]
    pelvis = 0.5 * (lhip + rhip)

    x_axis = _normalize(rhip - lhip)

    lsho = kpts[:, idx["lsho"], :]
    rsho = kpts[:, idx["rsho"], :]
    shoulder_center = 0.5 * (lsho + rsho)
    y_axis = _normalize(shoulder_center - pelvis)

    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return pelvis, R


def world_to_body(
    points_world: np.ndarray, pelvis_world: np.ndarray, R_body_to_world: np.ndarray
) -> np.ndarray:
    v = points_world - pelvis_world
    return np.einsum("tij,tj->ti", np.transpose(R_body_to_world, (0, 2, 1)), v)


def kpts_world_to_body(kpts_world: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
    pelvis, R = build_body_frame_from_mhr70(kpts_world, idx)
    v = kpts_world - pelvis[:, None, :]
    kpts_body = np.einsum("tij,tbj->tbi", np.transpose(R, (0, 2, 1)), v)
    return kpts_body.astype(np.float32)


def right_hand_point_world(
    kpts_world: np.ndarray,
    idx: Dict[str, int],
    mode: Literal["wrist", "hand_center"] = "hand_center",
) -> np.ndarray:
    if mode == "wrist":
        return kpts_world[:, idx["rwrist"], :]
    wrist = kpts_world[:, idx["rwrist"], :]
    index_tip = kpts_world[:, idx["rindex_tip"], :]
    middle_tip = kpts_world[:, idx["rmiddle_tip"], :]
    pinky_tip = kpts_world[:, idx["rpinky_tip"], :]
    return 0.25 * (wrist + index_tip + middle_tip + pinky_tip)


def compute_theta_unwrap_from_world(
    kpts_world: np.ndarray, idx: Dict[str, int]
) -> np.ndarray:
    """
    Compute theta(t) on BODY x-z plane (right hand), then unwrap.
    """
    pelvis, R = build_body_frame_from_mhr70(kpts_world, idx)
    hand_w = right_hand_point_world(kpts_world, idx, mode="hand_center")
    hand_b = world_to_body(hand_w, pelvis, R)
    x, z = hand_b[:, 0], hand_b[:, 2]
    theta = np.arctan2(z, x).astype(np.float32)
    theta = smooth_1d(theta, 11)
    return np.unwrap(theta)

