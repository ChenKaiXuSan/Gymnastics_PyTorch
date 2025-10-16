#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/save.py
Project: /workspace/code/triangulation
Created Date: Friday October 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday October 10th 2025 10:58:39 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
from typing import Any, Dict
from pathlib import Path
import os
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_3d_joints(
    joints_3d: np.ndarray,
    save_dir: Path,
    frame_idx: int,
    first_rt_info: dict[str, np.ndarray],
    second_rt_info: dict[str, np.ndarray],
    K_info: dict[str, np.ndarray],
    pt_path: dict[str, str],
):

    os.makedirs(save_dir, exist_ok=True)

    K_info = {k: v for k, v in K_info.items() if v is not None}

    # TODO: 这里是否应该保存视频路径呢
    data = {
        "frame": int(frame_idx),
        "num_joints": len(joints_3d),
        "joints_3d": joints_3d.tolist(),
        "first_rt_info": {k: v.tolist() for k, v in first_rt_info.items()},
        "second_rt_info": {k: v.tolist() for k, v in second_rt_info.items()},
        "pt_path": pt_path,
        "K_info": {k: v.tolist() for k, v in K_info.items()},

    }

    out_path = save_dir / f"frame_{frame_idx:04d}.json"

    # 原子写入，避免半写文件
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        # ensure_ascii=False 保留非 ASCII；allow_nan=False 保证严格 JSON（会在出现 NaN/Inf 时抛错）
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
    tmp.replace(out_path)

    logger.info(f"3D joints saved → {out_path}")
