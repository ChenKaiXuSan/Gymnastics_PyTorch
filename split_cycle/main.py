#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Union

import numpy as np

from split_cycle.load import load_sam3d_body_sequence


# ---------------------------- Utils ----------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def smooth_1d(x: np.ndarray, win: int = 9) -> np.ndarray:
    """Simple moving average (odd window)."""
    win = int(win)
    win = max(3, win | 1)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")


@dataclass
class CycleResult:
    boundaries: List[int]
    cycles: List[Tuple[int, int]]
    theta: np.ndarray
    theta_unwrap: np.ndarray
    hand_body: np.ndarray  # (T,3)


# ---------------------------- Core geometry ----------------------------

def build_body_frame_from_mhr70(
    kpts: np.ndarray,
    idx: Dict[str, int],
    *,
    use_shoulder_center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a human-centric body frame per frame.

    Args:
        kpts: (T,J,3)
        idx: must contain lhip, rhip, lsho, rsho, neck
        use_shoulder_center: if True, y-axis uses shoulder_center - pelvis.
                             else uses neck - pelvis.

    Returns:
        pelvis_world: (T,3)
        R_body_to_world: (T,3,3) columns are [x(right), y(up), z(forward)] in world
    """
    lhip = kpts[:, idx["lhip"], :]
    rhip = kpts[:, idx["rhip"], :]
    pelvis = 0.5 * (lhip + rhip)

    x_axis = _normalize(rhip - lhip)  # right

    if use_shoulder_center:
        lsho = kpts[:, idx["lsho"], :]
        rsho = kpts[:, idx["rsho"], :]
        shoulder_center = 0.5 * (lsho + rsho)
        y_axis = _normalize(shoulder_center - pelvis)  # up
    else:
        neck = kpts[:, idx["neck"], :]
        y_axis = _normalize(neck - pelvis)

    z_axis = _normalize(np.cross(x_axis, y_axis))  # forward (right-hand)
    y_axis = _normalize(np.cross(z_axis, x_axis))  # re-orthogonalize

    R = np.stack([x_axis, y_axis, z_axis], axis=-1)  # (T,3,3)
    return pelvis, R


def world_to_body(
    points_world: np.ndarray,
    pelvis_world: np.ndarray,
    R_body_to_world: np.ndarray,
) -> np.ndarray:
    """
    points_world: (T,3)
    pelvis_world: (T,3)
    R_body_to_world: (T,3,3)
    -> points_body: (T,3)
    """
    v = points_world - pelvis_world
    return np.einsum("tij,tj->ti", np.transpose(R_body_to_world, (0, 2, 1)), v)


def get_right_hand_point_world(
    kpts: np.ndarray,
    idx: Dict[str, int],
    mode: Literal["wrist", "hand_center"] = "wrist",
) -> np.ndarray:
    """
    mode="wrist": use right_wrist
    mode="hand_center": average of wrist + finger tips (more stable)

    Requires these indices if mode="hand_center":
      right_wrist(41), right_index_tip(25), right_middle_tip(29), right_pinky_tip(37)
    """
    if mode == "wrist":
        return kpts[:, idx["rwrist"], :]

    # hand_center
    wrist = kpts[:, idx["rwrist"], :]
    index_tip = kpts[:, idx["rindex_tip"], :]
    middle_tip = kpts[:, idx["rmiddle_tip"], :]
    pinky_tip = kpts[:, idx["rpinky_tip"], :]
    return 0.25 * (wrist + index_tip + middle_tip + pinky_tip)


# ---------------------------- Cycle detection ----------------------------

def find_crossings(
    theta_unwrap: np.ndarray,
    theta_ref: float,
    fps: float,
    *,
    min_period_sec: float = 0.8,
    direction: Literal["ccw", "cw"] = "ccw",
) -> List[int]:
    """
    Detect cycle boundaries by crossing theta_ref + 2*pi*k in a consistent direction.

    Uses sign change around aligned reference.
    """
    T = len(theta_unwrap)
    min_gap = int(round(min_period_sec * fps))

    # align reference branch for each frame
    k = np.round((theta_unwrap - theta_ref) / (2 * np.pi))
    ref = theta_ref + 2 * np.pi * k
    d = theta_unwrap - ref

    s = np.sign(d)
    s[s == 0] = 1

    vel = np.gradient(theta_unwrap)
    ok_dir = vel > 0 if direction == "ccw" else vel < 0

    crossing: List[int] = []
    for t in range(1, T):
        if ok_dir[t] and (s[t - 1] < 0 and s[t] > 0):
            crossing.append(t)

    # enforce min gap
    filtered: List[int] = []
    last = -10**9
    for t in crossing:
        if t - last >= min_gap:
            filtered.append(t)
            last = t
    return filtered


def segment_cycles_from_mhr70(
    kpts: np.ndarray,
    fps: float,
    idx: Dict[str, int],
    *,
    hand_mode: Literal["wrist", "hand_center"] = "wrist",
    theta_ref: float = -3 * np.pi / 4,   # 左後ろを仮定（body座標のx<0,z<0）
    smooth_win: int = 9,
    min_period_sec: float = 0.8,
    direction: Literal["ccw", "cw"] = "ccw",
    use_shoulder_center: bool = True,
) -> CycleResult:
    """
    kpts: (T,70,3) in world coordinates
    """
    pelvis, R = build_body_frame_from_mhr70(
        kpts, idx, use_shoulder_center=use_shoulder_center
    )

    hand_w = get_right_hand_point_world(kpts, idx, mode=hand_mode)
    hand_b = world_to_body(hand_w, pelvis, R)

    # horizontal plane in body frame: x-z
    x = hand_b[:, 0]
    z = hand_b[:, 2]
    theta = np.arctan2(z, x).astype(np.float32)  # (-pi, pi]
    theta = smooth_1d(theta, win=smooth_win)
    theta_unwrap = np.unwrap(theta)

    boundaries = find_crossings(
        theta_unwrap,
        theta_ref=theta_ref,
        fps=fps,
        min_period_sec=min_period_sec,
        direction=direction,
    )
    cycles = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    return CycleResult(
        boundaries=boundaries,
        cycles=cycles,
        theta=theta,
        theta_unwrap=theta_unwrap,
        hand_body=hand_b,
    )


def save_cycles_csv(out_path: Union[str, Path], cycles: List[Tuple[int, int]], fps: float) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["cycle_idx,start_frame,end_frame,start_time_sec,end_time_sec,duration_sec\n"]
    for i, (s, e) in enumerate(cycles):
        st, et = s / fps, e / fps
        lines.append(f"{i},{s},{e},{st:.6f},{et:.6f},{(et-st):.6f}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


# ---------------------------- Main ----------------------------

if __name__ == "__main__":
    # MHR70 indices you need
    IDX = {
        "lhip": 9,
        "rhip": 10,
        "lsho": 5,
        "rsho": 6,
        "neck": 69,
        "rwrist": 41,

        # only needed if hand_mode="hand_center"
        "rindex_tip": 25,
        "rmiddle_tip": 29,
        "rpinky_tip": 37,
    }

    root_path = "/workspace/data/sam3d_body_results"
    # 你现在 loader 返回值不一致（你写 face_info, face_kpts3d）
    # 建议统一成 seq = load... 然后 seq.kpts3d
    seq = load_sam3d_body_sequence(root_path, person_id="01", subdir="face")
    kpts3d = seq.kpts3d if hasattr(seq, "kpts3d") else seq[1]  # 兼容你当前写法

    print("kpts3d:", kpts3d.shape)  # (T, 70, 3)

    fps = 30.0

    res = segment_cycles_from_mhr70(
        kpts3d,
        fps=fps,
        idx=IDX,
        hand_mode="hand_center",     # wrist 抖的话用这个更稳
        smooth_win=11,
        min_period_sec=0.8,
        direction="ccw",             # 检测不到就换 "cw"
        use_shoulder_center=True,
    )

    print("cycle boundaries:", res.boundaries[:20])
    print("first 5 cycles:", res.cycles[:5])

    save_cycles_csv("./outputs/cycles_person01.csv", res.cycles, fps=fps)
    print("saved:", "./outputs/cycles_person01.csv")
