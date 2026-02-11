#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gymnastics motion analysis metrics

Three key metrics for gymnastics analysis:
1. Twist (ねじれ): Rotation difference between shoulders and hips
2. Trunk Tilt (姿勢): Trunk angle from vertical
3. Wrist Lead Angle (脱力): Wrist lead angle relative to trunk
"""

from typing import Dict, Tuple

import numpy as np


# -------------------- MHR70 Keypoint Indices --------------------
MHR70_INDEX: Dict[str, int] = {
    "lhip": 9,
    "rhip": 10,
    "lsho": 5,
    "rsho": 6,
    "neck": 69,
    "rwrist": 41,
    "lwrist": 40,
    "rindex_tip": 25,
    "rmiddle_tip": 29,
    "rpinky_tip": 37,
    "lindex_tip": 20,
    "lmiddle_tip": 24,
    "lpinky_tip": 32,
}


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """向量归一化"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


# ============================================================================
# 1. ねじれ（Twist）: 両肩と両股関節の回旋角度の差
# ============================================================================

def compute_rotation_angle(
    p1: np.ndarray, p2: np.ndarray, reference_axis: np.ndarray
) -> np.ndarray:
    """
    計算两点连线相对于参考轴的旋转角度

    Args:
        p1, p2: (T, 3) 两个关键点的位置
        reference_axis: (3,) 参考轴（例如：水平方向 [1,0,0]）

    Returns:
        angles: (T,) 旋转角度（弧度）
    """
    # 连线向量
    vec = p2 - p1  # (T, 3)

    # 投影到水平面（去除y分量）
    vec_h = vec.copy()
    vec_h[:, 1] = 0
    vec_h = _normalize(vec_h)

    # 计算与参考轴的角度
    ref = reference_axis / np.linalg.norm(reference_axis)
    cos_angle = np.clip(np.dot(vec_h, ref), -1.0, 1.0)

    # 使用叉积判断方向
    cross = np.cross(vec_h, ref)
    angles = np.arccos(cos_angle)
    angles = np.where(cross[:, 1] < 0, -angles, angles)

    return angles


def compute_twist(kpts_world: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
    """
    計算ねじれ：両肩と両股関節の回旋角度の差

    Args:
        kpts_world: (T, J, 3) 世界坐标系下的关键点
        idx: 关键点索引字典

    Returns:
        twist: (T,) 扭转角度（弧度），正值表示上半身相对下半身逆时针旋转
    """
    lsho = kpts_world[:, idx["lsho"], :]
    rsho = kpts_world[:, idx["rsho"], :]
    lhip = kpts_world[:, idx["lhip"], :]
    rhip = kpts_world[:, idx["rhip"], :]

    # 参考轴：从左到右 [1, 0, 0]
    ref_axis = np.array([1.0, 0.0, 0.0])

    # 计算肩部和髋部的旋转角度
    shoulder_angle = compute_rotation_angle(lsho, rsho, ref_axis)
    hip_angle = compute_rotation_angle(lhip, rhip, ref_axis)

    # 扭转 = 肩部角度 - 髋部角度
    twist = shoulder_angle - hip_angle

    return twist


# ============================================================================
# 2. 姿勢（Posture）: 体幹の傾き角度
# ============================================================================

def compute_trunk_tilt(
    kpts_world: np.ndarray, idx: Dict[str, int]
) -> np.ndarray:
    """
    計算姿勢：体幹（両肩と両股関節の中点を結んだ線）の鉛直軸に対する角度

    Args:
        kpts_world: (T, J, 3) 世界坐标系下的关键点
        idx: 关键点索引字典

    Returns:
        tilt_angle: (T,) 倾斜角度（弧度），0表示垂直，正值表示向前倾
    """
    lsho = kpts_world[:, idx["lsho"], :]
    rsho = kpts_world[:, idx["rsho"], :]
    lhip = kpts_world[:, idx["lhip"], :]
    rhip = kpts_world[:, idx["rhip"], :]

    # 肩部和髋部的中点
    shoulder_center = 0.5 * (lsho + rsho)
    hip_center = 0.5 * (lhip + rhip)

    # 体幹向量（从髋到肩）
    trunk_vec = shoulder_center - hip_center  # (T, 3)

    # 鉛直軸（向上）
    vertical_axis = np.array([0.0, 1.0, 0.0])

    # 计算与垂直轴的角度
    trunk_norm = _normalize(trunk_vec)
    cos_angle = np.clip(np.dot(trunk_norm, vertical_axis), -1.0, 1.0)

    # 角度（从垂直方向的偏离）
    tilt_angle = np.arccos(cos_angle) - np.pi / 2  # 转换为倾斜角度

    # 使用前后方向（z）判断倾斜方向
    tilt_angle = np.where(trunk_vec[:, 2] > 0, tilt_angle, -tilt_angle)

    return tilt_angle


# ============================================================================
# 3. 脱力（Relaxation）: 手首の先行角度
# ============================================================================

def detect_rotation_direction(
    kpts_world: np.ndarray, idx: Dict[str, int]
) -> str:
    """
    検出旋转方向（基于初期和中期的髋部旋转）

    Returns:
        "ccw" (逆时针) or "cw" (顺时针)
    """
    lhip = kpts_world[:, idx["lhip"], :]
    rhip = kpts_world[:, idx["rhip"], :]

    # 计算髋部的水平旋转角度
    ref_axis = np.array([1.0, 0.0, 0.0])
    hip_angles = compute_rotation_angle(lhip, rhip, ref_axis)

    # 比较初期和中期的角度变化
    T = len(hip_angles)
    start_angle = np.mean(hip_angles[:T//4])
    mid_angle = np.mean(hip_angles[T//4:T//2])

    if mid_angle > start_angle:
        return "ccw"
    else:
        return "cw"


def find_frontal_facing_frames(
    kpts_world: np.ndarray, idx: Dict[str, int], threshold: float = 0.1
) -> np.ndarray:
    """
    找出体幹が正面を向いた時点（髋部接近平行于参考轴）

    Args:
        kpts_world: (T, J, 3)
        idx: 关键点索引
        threshold: 角度阈值（弧度），判断是否正面

    Returns:
        frontal_mask: (T,) bool数组，True表示该帧面向正面
    """
    lhip = kpts_world[:, idx["lhip"], :]
    rhip = kpts_world[:, idx["rhip"], :]

    ref_axis = np.array([1.0, 0.0, 0.0])
    hip_angles = compute_rotation_angle(lhip, rhip, ref_axis)

    # 接近0度或180度表示正面
    frontal_mask = np.abs(hip_angles) < threshold

    return frontal_mask


def compute_wrist_lead_angle(
    kpts_world: np.ndarray, idx: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    計算脱力：回旋方向と反側の手（遅れてくる手）の先行角度

    Args:
        kpts_world: (T, J, 3)
        idx: 关键点索引

    Returns:
        lead_angle: (T,) 手腕相对于躯干的先行角度（弧度）
        frontal_mask: (T,) bool数组，标记正面时刻
        lagging_hand: "left" or "right" 滞后的手
    """
    # 检测旋转方向
    rotation_dir = detect_rotation_direction(kpts_world, idx)

    # 确定滞后的手（旋转方向的对侧）
    if rotation_dir == "ccw":
        # 逆时针旋转，右手滞后
        lagging_hand = "right"
        wrist = kpts_world[:, idx["rwrist"], :]
    else:
        # 顺时针旋转，左手滞后
        lagging_hand = "left"
        wrist = kpts_world[:, idx["lwrist"], :]

    # 找出正面时刻
    frontal_mask = find_frontal_facing_frames(kpts_world, idx)

    # 计算手腕的水平旋转角度
    lhip = kpts_world[:, idx["lhip"], :]
    rhip = kpts_world[:, idx["rhip"], :]
    hip_center = 0.5 * (lhip + rhip)

    # 手腕相对于髋部中心的向量
    wrist_vec = wrist - hip_center

    # 投影到水平面
    wrist_h = wrist_vec.copy()
    wrist_h[:, 1] = 0
    wrist_h = _normalize(wrist_h)

    # 髋部方向（作为躯干方向）
    hip_vec = rhip - lhip
    hip_h = hip_vec.copy()
    hip_h[:, 1] = 0
    hip_h = _normalize(hip_h)

    # 计算手腕相对于躯干的角度
    cos_angle = np.clip(np.einsum('ti,ti->t', wrist_h, hip_h), -1.0, 1.0)
    cross = np.cross(hip_h, wrist_h)
    lead_angle = np.arccos(cos_angle)
    lead_angle = np.where(cross[:, 1] < 0, -lead_angle, lead_angle)

    return lead_angle, frontal_mask, lagging_hand
