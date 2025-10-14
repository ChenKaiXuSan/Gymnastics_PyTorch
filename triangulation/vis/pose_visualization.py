#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/pose_visualization.py
Project: /workspace/code/triangulation/vis
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 10:55:01 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import os

import numpy as np

import matplotlib.pyplot as plt

def draw_camera(
    ax: plt.Axes,
    rt_info: Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,  # (3,3)
    image_size: Tuple[int, int],  # (W, H) in px
    axis_len: float = 1.0,
    frustum_depth: float = 1.0,
    colors: Tuple[str, str, str] = ("r", "g", "b"),
    label: Optional[str] = None,
    convention: str = "cam2world",  # or "cam2world"
    ray_scale_mode: str = "depth",  # "depth" or "focal"
    linewidths: Optional[Dict[str, float]] = None,
    frustum_alpha: float = 1.0,
) -> np.ndarray:
    """
    在 Matplotlib 3D 轴上绘制 OpenCV 相机坐标系与视锥体。

    坐标系对应：
      OpenCV: x→右, y→下, z→前
      Matplotlib: x→右, y→前, z→上

    返回:
        C_plt: (3,) 相机中心在 Matplotlib 世界坐标中的位置
    """
    # ---------------- 参数准备 ----------------
    if linewidths is None:
        linewidths = {"axis": 1.0, "frustum": 0.5}

    R, T, C = rt_info["R"], rt_info["t"], rt_info["C"]
    K = np.asarray(K, float).reshape(3, 3)
    W, H = [float(v) for v in image_size]

    # ---------------- 相机中心计算 ----------------
    # Xc = R Xw + T → C = -R^T T
    R_wc = R
    C_world = -R.T @ T

    C_plt = C_world.ravel()

    # ---------------- 绘制相机坐标轴 ----------------
    for axis_vec, color in zip(R_wc, colors):  # R_wc 的行向量即各相机轴方向
        end_cv = axis_vec * axis_len
        end_plt = end_cv
        ax.plot(
            [C_plt[0], C_plt[0] + end_plt[0]],
            [C_plt[1], C_plt[1] + end_plt[1]],
            [C_plt[2], C_plt[2] + end_plt[2]],
            c=color,
            lw=linewidths["axis"],
        )

    # ---------------- 计算视锥体四个角点 ----------------
    corners_px = np.array(
        [
            [0, 0, 1],
            [W - 1, 0, 1],
            [W - 1, H - 1, 1],
            [0, H - 1, 1],
        ],
        dtype=float,
    )
    rays_cam = np.linalg.inv(K) @ corners_px.T  # (3,4)

    if ray_scale_mode == "depth":
        scale = frustum_depth / np.clip(rays_cam[2, :], 1e-9, None)
        rays_cam = rays_cam * scale
    else:
        fx, fy = K[0, 0], K[1, 1]
        s = max(axis_len, frustum_depth) / max((fx + fy) / 2, 1e-6)
        rays_cam *= s

    # cam→world(OpenCV)
    corners_world_cv = (R_wc @ rays_cam).T + C_world.reshape(1, 3)
    # world(OpenCV) → Matplotlib
    corners_world_plt = corners_world_cv

    # ---------------- 绘制视锥体边缘 ----------------
    for p in corners_world_plt:
        ax.plot(
            [C_plt[0], p[0]],
            [C_plt[1], p[1]],
            [C_plt[2], p[2]],
            c="k",
            lw=linewidths["frustum"],
            alpha=frustum_alpha,
        )

    loop = [0, 1, 2, 3, 0]
    ax.plot(
        corners_world_plt[loop, 0],
        corners_world_plt[loop, 1],
        corners_world_plt[loop, 2],
        c="k",
        lw=linewidths["frustum"],
        alpha=frustum_alpha,
    )

    # ---------------- 标签 ----------------
    if label:
        ax.text(C_plt[0], C_plt[1], C_plt[2], label, color="black", fontsize=9)

    return C_plt


def visualize_3d_joints(
    joints_3d,
    first_rt_info,
    second_rt_info,
    K_info,
    save_path,
    title="Triangulated 3D Joints",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    K = np.asarray(K_info["K"], np.float32).reshape(3, 3)
    distortion_coefficients = K_info.get("distortion_coefficients", None)

    # ----------- 相对姿态计算 -----------
    R1, T1, C1 = (
        first_rt_info["R"],
        first_rt_info["t"].reshape(3, 1),
        first_rt_info["C"].reshape(3, 1),
    )
    R2, T2, C2 = (
        second_rt_info["R"],
        second_rt_info["t"].reshape(3, 1),
        second_rt_info["C"].reshape(3, 1),
    )

    # 第二个相机相对第一个相机的姿态
    R_rel = R2 @ R1.T
    T_rel = T2 - R_rel @ T1

    # draw_camera(ax, np.eye(3), np.zeros(3), label="Cam1")
    # draw_camera(ax, R_rel, T_rel, label="Cam2")

    draw_camera(
        ax,
        first_rt_info,
        K,
        K_info["image_size"],
        label="Cam1_K",
        frustum_alpha=0.3,
        convention="world2cam",
    )
    draw_camera(
        ax,
        second_rt_info,
        K,
        K_info["image_size"],
        label="Cam2_K",
        frustum_alpha=0.3,
        convention="world2cam",
    )

    plots = joints_3d
    xlab, ylab, zlab = "X", "Z", "Y (up)"

    # 点与索引
    plots[:, 1] = -plots[:, 1]  # 反转Y轴以符合Y朝上习惯
    ax.scatter(plots[:, 0], plots[:, 1], plots[:, 2], c="blue", s=30)

    for i, (x, y, z) in enumerate(plots):
        ax.text(x, y, z, str(i), size=8)

    # 骨架与长度
    skeleton = COCO_SKELETON
    # 画线（在绘图坐标系）
    for i, j in skeleton:
        if i < len(plots) and j < len(plots):
            if not (np.all(np.isfinite(plots[i])) and np.all(np.isfinite(plots[j]))):
                continue
            ax.plot(
                [plots[i, 0], plots[j, 0]],
                [plots[i, 1], plots[j, 1]],
                [plots[i, 2], plots[j, 2]],
                c="red",
                linewidth=2,
            )

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    plt.tight_layout()

    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)

    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")
