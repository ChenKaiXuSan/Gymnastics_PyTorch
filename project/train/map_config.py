#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/map_config.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project
Created Date: Monday March 9th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday March 9th 2026 11:22:51 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from dataclasses import dataclass

# * 这个文件定义了与 Unity MHR70 骨骼结构相关的映射和配置，供整个项目使用。
# 所有关节索引 (0-69，共70个关节)
ALL_INDICES = list(range(70))

# 过滤后的关节索引 (选中的20个关键关节)
INDICES = [
    0,  # nose
    5,
    6,  # shoulders
    7,
    8,  # elbows
    41,
    62,  # wrists
    69,  # neck
    9,
    10,  # hips
    11,
    12,  # knees
    13,
    14,  # ankles
    15,
    16,
    17,  # left foot
    18,
    19,
    20,  # right foot
]

GLOBAL_SKELETON_CONNECTIONS = [
    (0, 69),  # nose - neck
    (69, 5),  # neck - left_shoulder
    (69, 6),  # neck - right_shoulder
    (5, 7),  # left_shoulder - left_elbow
    (7, 62),  # left_elbow - left_wrist
    (6, 8),  # right_shoulder - right_elbow
    (8, 41),  # right_elbow - right_wrist
    (5, 6),  # left_shoulder - right_shoulder
    (9, 10),  # left_hip - right_hip
    (5, 9),  # left_shoulder - left_hip
    (6, 10),  # right_shoulder - right_hip
    (9, 11),  # left_hip - left_knee
    (11, 13),  # left_knee - left_ankle
    (13, 15),  # left_ankle - left_big_toe
    (13, 16),  # left_ankle - left_small_toe
    (13, 17),  # left_ankle - left_heel
    (10, 12),  # right_hip - right_knee
    (12, 14),  # right_knee - right_ankle
    (14, 18),  # right_ankle - right_big_toe
    (14, 19),  # right_ankle - right_small_toe
    (14, 20),  # right_ankle - right_heel
]


@dataclass
class PersonInfo:
    """全局映射配置类，包含与 Unity MHR70 骨骼结构相关的映射和配置。"""

    person_id: str
    turn_id: str

    cam1_video_path: str
    cam2_video_path: str

    sam3d_cam1_results_path: str
    sam3d_cam2_results_path: str

    cam1_turn_frame_start: int
    cam1_turn_frame_end: int

    cam2_turn_frame_start: int
    cam2_turn_frame_end: int

    label_twist_3class: int
    label_posture_3class: int
    label_relax_3class: int

    label_total_5class: int
    label_total_3class: int

    fused_kpt_path: str
    fused_kpt_turn_frame_start: int
    fused_kpt_turn_frame_end: int
