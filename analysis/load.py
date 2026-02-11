#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Data loading utilities for analysis module
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


def load_fused_frame(frame_path: Union[str, Path]) -> Dict:
    """
    加载单个融合帧的数据
    
    Args:
        frame_path: npz文件路径
        
    Returns:
        包含kpts_world, kpts_body等信息的字典
    """
    from fuse.load import load_fused_frame as _load_frame
    return _load_frame(frame_path)


def load_fused_sequence(
    person_id: str,
    person_root: Optional[Union[str, Path]] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    加载融合后的3D关键点序列
    
    Args:
        person_id: 人物ID
        person_root: 人物根目录，None则使用默认路径 logs/fuse/person_{id}
        start_frame: 起始帧（含），None表示从第一帧开始
        end_frame: 结束帧（不含），None表示到最后一帧
        
    Returns:
        kpts_world: (T, J, 3) 世界坐标系下的关键点
        kpts_body: (T, J, 3) 身体坐标系下的关键点
        metadata: 元数据字典
    """
    from fuse.load import load_fused_sequence as _load_sequence
    
    if person_root is None:
        person_root = Path("logs/fuse") / f"person_{person_id}"
    
    return _load_sequence(person_root, person_id, start_frame, end_frame)


def get_fused_frame_mapping(
    person_id: str,
    person_root: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取融合序列的帧映射信息
    
    Args:
        person_id: 人物ID
        person_root: 人物根目录，None则使用默认路径
        
    Returns:
        face_map: (T,) 到原始face视频的帧映射
        side_map: (T,) 到原始side视频的帧映射
    """
    from fuse.load import get_fused_frame_mapping as _get_mapping
    
    if person_root is None:
        person_root = Path("logs/fuse") / f"person_{person_id}"
    
    return _get_mapping(person_root, person_id)
