#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/split_cycle/load.py
Project: /workspace/code/split_cycle
Created Date: Thursday January 29th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 29th 2026 10:14:54 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def load_sam3d_body_sequence(
    root: Union[str, Path],
    person_id: str = "01",
    subdir: str = "face",
    pattern: str = "*_sam3d_body.npz",
) -> Tuple[List[Dict], np.ndarray]:
    """
    Load per-frame SAM3D-Body npz outputs into a full sequence.

    Expected structure (example):
      root/
        sam3d_body_results/
          person/01/face/000000_sam3d_body.npz
          person/01/face/000001_sam3d_body.npz
          ...

    Args:
        root: path to 'results' or a path that contains 'sam3d_body_results'
        person_id: e.g., "01"
        subdir: e.g., "face" (or "body", "mesh", etc. depends on your export)
        pattern: glob pattern for frame npz files
        prefer: choose which key list to try first ("kpts3d" is default)
        strict: if True, raise when cannot find keypoints in a frame.

    Returns:
        Tuple[List[Dict], np.ndarray]: A tuple containing a list of dictionaries with frame information and a numpy array of keypoints.
    """
    root = Path(root)

    # allow passing either ".../results" or ".../results/sam3d_body_results/..."
    if (root / "sam3d_body_results").exists():
        base = root / "sam3d_body_results" / "person" / person_id / subdir
    elif root.name == "sam3d_body_results":
        base = root / "person" / person_id / subdir
    else:
        # try to find sam3d_body_results under root
        cand = root / "sam3d_body_results" / "person" / person_id / subdir
        base = cand

    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base}")

    files = sorted(base.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No npz files matched: {base}/{pattern}")

    all_kpts: List[np.ndarray] = []

    all_info: List[Dict] = []

    one_frame_info_dict: Dict = {}

    for fp in files:

        info = np.load(fp, allow_pickle=True)["output"].item()
        pred_keypoints_3d = info[
            "pred_keypoints_3d"
        ]  # (J, 3) or (1, J, 3) or (J, 4) etc.
        frame = info["frame"]  # frame index
        frame_idx = info["frame_idx"]  # frame index in video

        one_frame_info_dict[frame_idx] = info
        one_frame_info_dict["frame"] = frame
        one_frame_info_dict["pred_keypoints_3d"] = pred_keypoints_3d

        all_info.append(one_frame_info_dict)
        all_kpts.append(pred_keypoints_3d)  # take (x,y,z)

        # if frame_idx > 60:
        #     break

    # Stack kpts
    kpts3d = np.stack(all_kpts, axis=0)  # (T,J,3)

    return all_info, kpts3d


# ============================================================================
# Functions for loading fused 3D keypoints (per-frame npz files)
# ============================================================================

def load_fused_frame(frame_path: Path) -> Dict:
    """
    加载单帧的融合3D关键点
    
    Args:
        frame_path: npz文件路径
        
    Returns:
        包含关键点数据的字典
    """
    data = np.load(frame_path)
    return {
        'kpts_world': data['kpts_world'],      # (J, 3)
        'kpts_body': data['kpts_body'],        # (J, 3)
        'face_frame_idx': int(data['face_frame_idx']),
        'side_frame_idx': int(data['side_frame_idx']),
        'frame_idx': int(data['frame_idx']),
    }


def load_fused_sequence(
    person_root: Union[str, Path],
    person_id: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    加载融合后的3D关键点序列
    
    Args:
        person_root: 人物根目录 (例如: logs/fuse/person_1)
        person_id: 人物ID
        start_frame: 起始帧（含），None表示从第一帧开始
        end_frame: 结束帧（不含），None表示到最后一帧
        
    Returns:
        kpts_world: (T, J, 3) 世界坐标系下的关键点
        kpts_body: (T, J, 3) 身体坐标系下的关键点
        metadata: 元数据字典
    """
    import json
    
    person_root = Path(person_root)
    
    # 读取元数据
    metadata_path = person_root / f"fused_kpts_metadata_{person_id}.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    frames_dir = person_root / metadata['frames_dir']
    n_frames = metadata['n_frames']
    
    # 确定帧范围
    start = start_frame if start_frame is not None else 0
    end = end_frame if end_frame is not None else n_frames
    start = max(0, min(start, n_frames))
    end = max(start, min(end, n_frames))
    
    # 逐帧加载
    kpts_world_list = []
    kpts_body_list = []
    
    for t in range(start, end):
        frame_path = frames_dir / f"frame_{t:06d}.npz"
        if not frame_path.exists():
            print(f"Warning: frame {t} not found, skipping")
            continue
            
        frame_data = load_fused_frame(frame_path)
        kpts_world_list.append(frame_data['kpts_world'])
        kpts_body_list.append(frame_data['kpts_body'])
    
    kpts_world = np.stack(kpts_world_list, axis=0)  # (T, J, 3)
    kpts_body = np.stack(kpts_body_list, axis=0)    # (T, J, 3)
    
    return kpts_world, kpts_body, metadata


def get_fused_frame_mapping(
    person_root: Union[str, Path], 
    person_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取融合序列的帧映射信息
    
    Args:
        person_root: 人物根目录
        person_id: 人物ID
        
    Returns:
        face_map: (T,) 到原始face视频的帧映射
        side_map: (T,) 到原始side视频的帧映射
    """
    import json
    
    metadata_path = Path(person_root) / f"fused_kpts_metadata_{person_id}.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    face_map = np.array(metadata['face_map'], dtype=np.int32)
    side_map = np.array(metadata['side_map'], dtype=np.int32)
    
    return face_map, side_map
