#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/main.py
Project: /workspace/code/triangulation
Created Date: Friday August 22nd 2025
Author: Kaixu Chen
-----
Comment:
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation, pose estimation, and interactive 3D visualization.

Have a good code time :)
-----
Last Modified: Friday August 22nd 2025 9:46:02 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

14-10-2025	Kaixu Chen	目前只支持两视点的三角测量，后续可以扩展到多视点。
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import os
import glob
from pathlib import Path

import numpy as np
import cv2
import hydra


from triangulation.camera_position_mapping import prepare_camera_position
from triangulation.load import load_keypoints_from_npz, load_kpt_and_bbox_from_d2_pt
from triangulation.vis.frame_visualization import draw_and_save_keypoints_from_frame
from triangulation.vis.pose_visualization import draw_camera, visualize_3d_joints
from triangulation.vis.merge_video import merge_frames_to_video

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


# ---------- 主处理函数 ----------
def process_one_video(
    first_path,
    second_path,
    first_rt_info,
    second_rt_info,
    output_path: Path,
    K,
    extrinsics,
    vis: Dict[str, Any],
):
    output_path.mkdir(parents=True, exist_ok=True)

    # FIXME: 因为npz的2-4结果有问题，所以暂时不使用
    # first_kpts, first_vframes = load_keypoints_from_npz(npz_path=first_path)
    # second_kpts, second_vframes = load_keypoints_from_npz(npz_path=second_path)

    first_kpts, first_kpts_score, _, _, first_vframes = load_kpt_and_bbox_from_d2_pt(
        file_path=first_path, return_frames=True
    )
    second_kpts, second_kpts_score, _, _, second_vframes = load_kpt_and_bbox_from_d2_pt(
        file_path=second_path, return_frames=True
    )

    # 如果视频长度不一样，按照短的视频进行对齐
    if first_kpts.shape[0] != second_kpts.shape[0]:
        min_frames = min(first_kpts.shape[0], second_kpts.shape[0])
        first_kpts = first_kpts[:min_frames]
        second_kpts = second_kpts[:min_frames]

    for i in range(first_kpts.shape[0]):

        # if i > 60:
        #     break

        l_kpt, r_kpt = first_kpts[i], second_kpts[i]

        # drop the 0 value keypoints
        assert (
            l_kpt.shape == r_kpt.shape
        ), f"Keypoints shape mismatch: {l_kpt.shape} vs {r_kpt.shape}"

        l_frame = first_vframes[i] if first_vframes is not None else None
        r_frame = second_vframes[i] if second_vframes is not None else None

        if l_frame is not None and r_frame is not None or vis.save_kpts_frames:
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

        joints_3d = triangulate_joints(l_kpt, r_kpt, K, first_rt_info, second_rt_info)

        if vis.save_pose_3d_frames:
            visualize_3d_joints(
                joints_3d,
                first_rt_info,
                second_rt_info,
                K,
                os.path.join(output_path, f"3d/frame_{i:04d}.png"),
                title=f"Frame {i} - 3D Joints",
                skeleton=COCO_SKELETON,
            )

        # * 保存三维关键点

    # * merge frame to video
    if vis.merge_3d_frames_to_video:
        merge_frames_to_video(
            frame_dir=output_path / "3d",
            output_video_path=output_path / (output_path.stem + ".mp4"),
            fps=30,
        )


# ---------- 多人批量处理入口 ----------
# TODO：这里需要同时加载一个人的四个视频逻辑才行
def process_person_videos(
    input_path, output_path, rt_info, K, extrinsics, vis: Dict[str, Any]
):
    subjects = sorted(glob.glob(f"{input_path}/*/"))
    if not subjects:
        raise FileNotFoundError(f"No folders found in: {input_path}")
    print(f"[INFO] Found {len(subjects)} subjects in {input_path}")
    for person_dir in subjects:
        person_name = os.path.basename(person_dir.rstrip("/"))
        print(f"\n[INFO] Processing: {person_name}")

        # TODO: 这里预测的代码需要修改一下，统一为一个格式比较好。
        pt_file = sorted(Path(person_dir).glob("*.pt"))

        first = pt_file[0]
        second = pt_file[1]

        out_dir = Path(output_path) / person_name
        # TODO: 这里需要根据相机位置选择对应的外参
        first_rt_info = rt_info[int(first.stem)]
        second_rt_info = rt_info[int(second.stem)]

        process_one_video(
            first,
            second,
            first_rt_info,
            second_rt_info,
            out_dir,
            K,
            extrinsics,
            vis,
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
        vis=cfg.vis,
    )


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
