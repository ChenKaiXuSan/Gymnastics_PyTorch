#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation, pose estimation, and interactive 3D visualization.

Author: Kaixu Chen
Last Modified: August 4th, 2025
"""

import os
import numpy as np
import cv2
import torch
from torchvision.io import read_video
import glob
from pathlib import Path
import matplotlib
import hydra
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py

from triangulation.camera_position import (
    estimate_camera_pose_from_sift_imgs,
    estimate_pose,
    to_gray_cv_image,
    visualize_SIFT_matches,
)

from triangulation.camera_position_mapping import prepare_camera_position

# COCO-17 骨架（左/右臂、腿、躯干、头部）
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


# ---------- 可视化工具 ----------
def draw_and_save_keypoints_from_frame(
    frame,
    keypoints,
    save_path,
    color=(0, 255, 0),
    radius=4,
    thickness=-1,
    with_index=True,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = frame.numpy() if isinstance(frame, torch.Tensor) else frame.copy()
    for i, (x, y) in enumerate(keypoints):
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        if with_index:
            cv2.putText(
                img,
                str(i),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"[INFO] Saved image with keypoints to: {save_path}")

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

    R, T, C = rt_info['R'], rt_info['t'], rt_info['C']
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


# ---------- 三角测量 ----------
def triangulate_joints(keypoints1, keypoints2, K_info, first_rt_info, second_rt_info):
    """
    使用两视点关键点进行三角测量，返回三维坐标（相机1坐标系下）。
    参数:
        keypoints1, keypoints2: (N, 2) 像素坐标
        K: (3, 3) 相机内参矩阵
        first_rt_info, second_rt_info: dict，包含 R (3x3), T (3,)
    返回:
        pts_3d: (N, 3) 三维点坐标
    """
    # ----------- 输入检查 -----------
    keypoints1 = np.asarray(keypoints1, np.float32).reshape(-1, 2)
    keypoints2 = np.asarray(keypoints2, np.float32).reshape(-1, 2)

    if keypoints1.shape != keypoints2.shape:
        raise ValueError(
            f"Keypoints shape mismatch: {keypoints1.shape} vs {keypoints2.shape}"
        )
    if keypoints1.shape[0] < 5:
        raise ValueError("Need at least 5 correspondences for triangulation.")

    K = np.asarray(K_info["K"], np.float32).reshape(3, 3)
    distortion_coefficients = K_info.get("distortion_coefficients", None)
    # ----------- 相对姿态计算 -----------
    # R1, T1: 第一个相机的外参，R2, T2: 第二个相机的外参，均在世界坐标系下
    R1, T1 = first_rt_info["R"], first_rt_info["t"].reshape(3, 1)
    R2, T2 = second_rt_info["R"], second_rt_info["t"].reshape(3, 1)

    # ----------- 投影矩阵构造 -----------
    # 投影矩阵 P = K [R|T]
    P1 = K @ np.hstack((R1, T1))
    P2 = K @ np.hstack((R2, T2))

    # ----------- 三角测量 -----------
    pts_4d_h = cv2.triangulatePoints(P1, P2, keypoints1.T, keypoints2.T)
    pts_3d = (pts_4d_h[:3] / pts_4d_h[3]).T  # 齐次归一化 (N,3)

    # ----------- 数值清理与验证 -----------
    pts_3d = np.nan_to_num(pts_3d)
    mask_valid = np.isfinite(pts_3d).all(axis=1)
    pts_3d = pts_3d[mask_valid]

    return pts_3d


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
    R1, T1, C1 = first_rt_info["R"], first_rt_info["t"].reshape(3, 1), first_rt_info["C"].reshape(3, 1)
    R2, T2, C2 = second_rt_info["R"], second_rt_info["t"].reshape(3, 1), second_rt_info["C"].reshape(3, 1)

    # 第二个相机相对第一个相机的姿态
    R_rel = R2 @ R1.T
    T_rel = T2 - R_rel @ T1

    # draw_camera(ax, np.eye(3), np.zeros(3), label="Cam1")
    # draw_camera(ax, R_rel, T_rel, label="Cam2")

    draw_camera(ax, first_rt_info, K, K_info["image_size"], label="Cam1_K", frustum_alpha=0.3, convention="world2cam")
    draw_camera(ax, second_rt_info, K, K_info["image_size"], label="Cam2_K", frustum_alpha=0.3, convention="world2cam")

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


def visualize_3d_scene_interactive(joints_3d, R, T, save_path):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=joints_3d[:, 0],
            y=joints_3d[:, 1],
            z=joints_3d[:, 2],
            mode="markers+text",
            text=[str(i) for i in range(len(joints_3d))],
            marker=dict(size=4, color="blue"),
        )
    )

    def get_camera_lines(R, T, label):
        scale = 0.1
        lines = []
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        colors = ["red", "green", "blue"]
        origin = T.reshape(3)
        for axis, color in zip(axes, colors):
            end = R @ axis * scale + origin
            lines.append(
                go.Scatter3d(
                    x=[origin[0], end[0]],
                    y=[origin[1], end[1]],
                    z=[origin[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=4),
                )
            )
        view_dir = R @ np.array([0, 0, -1]) * scale * 1.5 + origin
        lines.append(
            go.Scatter3d(
                x=[origin[0], view_dir[0]],
                y=[origin[1], view_dir[1]],
                z=[origin[2], view_dir[2]],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name=f"{label}_view",
            )
        )
        return lines

    for trace in get_camera_lines(np.eye(3), np.zeros(3), "Cam1"):
        fig.add_trace(trace)
    for trace in get_camera_lines(R, T, "Cam2"):
        fig.add_trace(trace)

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="Interactive 3D Scene",
        margin=dict(l=0, r=0, b=0, t=30),
    )
    py.plot(fig, filename=save_path, auto_open=False)
    print(f"[INFO] Saved interactive HTML to: {save_path}")


# ---------- 加载关键点 ----------
def load_keypoints_from_pt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    print(f"[INFO] Loading: {file_path}")
    data = torch.load(file_path, map_location="cpu")
    video_path = data.get("video_path", None)
    vframes = (
        read_video(video_path, pts_unit="sec", output_format="THWC")[0]
        if video_path
        else None
    )
    keypoints = np.array(data["keypoint"]["keypoint"]).squeeze(0)
    if keypoints.ndim != 3 or keypoints.shape[2] != 2:
        raise ValueError(f"Invalid shape: {keypoints.shape}")
    if vframes is not None:
        keypoints[:, :, 0] *= vframes.shape[2]
        keypoints[:, :, 1] *= vframes.shape[1]
    return keypoints, vframes


def load_keypoints_from_npz(npz_path, key="keypoints"):
    """
    从 .npz 文件加载 keypoints 数据

    参数:
        npz_path (str): .npz 文件路径
        key (str): 读取的 key，默认 'keypoints'

    返回:
        np.ndarray: 关键点数据 (形状可能是 N x K x 2 或 N x K x 3)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    if key not in data:
        raise KeyError(f"{npz_path} 中没有 key '{key}'，可选 keys: {list(data.keys())}")

    kpts = data[key]
    print(f"加载完成: {npz_path}")
    print(f"key '{key}' 的 shape: {kpts.shape}")

    kpts = kpts.squeeze()
    kpts = kpts.transpose((0, 2, 1))
    # delete the z-coordinate
    kpts = kpts[:, :, :2]

    # load video frames
    raw_video_path = Path("/workspace/data/raw/suwabe")
    video_path = list(raw_video_path.glob(f"{npz_path.parent.stem}/{npz_path.stem}*"))[
        0
    ]
    if video_path:
        video_frames = read_video(video_path, pts_unit="sec", output_format="THWC")[0]

    # 视频的frame向右旋转90°
    # video_frames = torch.rot90(video_frames, k=-1, dims=(1, 2))
    video_frames = np.rot90(video_frames, k=-1, axes=(1, 2)).copy()

    return kpts, video_frames


# ---------- 主处理函数 ----------
def process_one_video(
    first_path, second_path, first_rt_info, second_rt_info, output_path, K, extrinsics
):
    os.makedirs(output_path, exist_ok=True)
    first_kpts, first_vframes = load_keypoints_from_npz(first_path)
    second_kpts, second_vframes = load_keypoints_from_npz(second_path)

    # 如果视频长度不一样，按照短的视频进行对齐
    if first_kpts.shape[0] != second_kpts.shape[0]:
        min_frames = min(first_kpts.shape[0], second_kpts.shape[0])
        first_kpts = first_kpts[:min_frames]
        second_kpts = second_kpts[:min_frames]

    for i in range(first_kpts.shape[0]):
        l_kpt, r_kpt = first_kpts[i], second_kpts[i]

        # drop the 0 value keypoints
        assert (
            l_kpt.shape == r_kpt.shape
        ), f"Keypoints shape mismatch: {l_kpt.shape} vs {r_kpt.shape}"

        # if 0 value find in left or right keypoints, drop them
        # 把为0的点替换成 np.nan（防止误差）
        # l_kpt[l_kpt == 0] = np.nan
        # r_kpt[r_kpt == 0] = np.nan

        # 创建有效掩码：左右关键点都不是 nan 的点
        # valid_mask = ~np.isnan(l_kpt).any(axis=1) & ~np.isnan(r_kpt).any(axis=1)

        # 过滤左右关键点
        # l_kpt = l_kpt[valid_mask]
        # r_kpt = r_kpt[valid_mask]

        l_frame = first_vframes[i] if first_vframes is not None else None
        r_frame = second_vframes[i] if second_vframes is not None else None

        if l_frame is not None and r_frame is not None:
            draw_and_save_keypoints_from_frame(
                l_frame,
                l_kpt,
                os.path.join(output_path, f"frames/left/{i:04d}.png"),
                color=(0, 255, 0),
            )
            draw_and_save_keypoints_from_frame(
                r_frame,
                r_kpt,
                os.path.join(output_path, f"frames/right/{i:04d}.png"),
                color=(0, 0, 255),
            )

        # * 使用SIFT特征匹配估计相机姿态
        # R, T, pts1, pts2, mask_pose = estimate_camera_pose_from_sift_imgs(
        #     to_gray_cv_image(l_frame),
        #     to_gray_cv_image(r_frame),
        #     K,
        # )

        # visualize_SIFT_matches(
        #     to_gray_cv_image(l_frame),
        #     to_gray_cv_image(r_frame),
        #     pts1,
        #     pts2,
        #     os.path.join(output_path, f"sift/matches_{i:04d}.png"),
        # )

        # R, T, mask = estimate_pose(l_kpt, r_kpt, K)
        # if R is None or T is None:
        #     print(f"[WARN] Frame {i}: pose estimation failed")
        #     continue

        joints_3d = triangulate_joints(l_kpt, r_kpt, K, first_rt_info, second_rt_info)
        visualize_3d_joints(
            joints_3d,
            first_rt_info,
            second_rt_info,
            K,
            os.path.join(output_path, f"3d/frame_{i:04d}.png"),
            title=f"Frame {i} - 3D Joints",
        )


# ---------- 多人批量处理入口 ----------
# TODO：这里需要同时加载一个人的四个视频逻辑才行
def process_person_videos(input_path, output_path, rt_info, K, extrinsics):
    subjects = sorted(glob.glob(f"{input_path}/*/"))
    if not subjects:
        raise FileNotFoundError(f"No folders found in: {input_path}")
    print(f"[INFO] Found {len(subjects)} subjects in {input_path}")
    for person_dir in subjects:
        person_name = os.path.basename(person_dir.rstrip("/"))
        print(f"\n[INFO] Processing: {person_name}")

        # TODO: 这里预测的代码需要修改一下，统一为一个格式比较好。
        npz_file = sorted(Path(person_dir).glob("*.npz"))
        first = npz_file[0]
        second = npz_file[1]

        out_dir = Path(output_path) / person_name
        # TODO: 这里需要根据相机位置选择对应的外参
        first_rt_info = rt_info[int(first.stem)]
        second_rt_info = rt_info[int(second.stem)]

        process_one_video(
            first, second, first_rt_info, second_rt_info, out_dir, K, extrinsics
        )


@hydra.main(version_base=None, config_path="../configs", config_name="triangulation")
def main(cfg):

    # 准备相机外部参数
    camera_position_dict = prepare_camera_position(
        K=np.array(cfg.camera_K.K),
        yaws=cfg.camera_position.yaws,
        T=cfg.camera_position.T,
        r=cfg.camera_position.r,
        z=cfg.camera_position.z,
        output_path=cfg.paths.log_path,
        img_size=cfg.camera_K.image_size,
    )

    process_person_videos(
        input_path=cfg.paths.input_path,
        output_path=cfg.paths.log_path,
        rt_info=camera_position_dict["rt_info"],
        K=cfg.camera_K,
        extrinsics=camera_position_dict["extrinsics_map"],
    )


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
