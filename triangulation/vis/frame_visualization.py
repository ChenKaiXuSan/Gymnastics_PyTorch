#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/frame_visualization.py
Project: /workspace/code/triangulation/vis
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 10:57:18 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import numpy as np
import cv2
import torch


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
