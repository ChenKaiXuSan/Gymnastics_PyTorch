#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/save.py
Project: /workspace/code/sam3d_body
Created Date: Friday December 5th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday December 5th 2025 11:52:16 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# 数据量很大，按一帧一个 npz 文件存储（ONLY per-frame saving: one frame -> one .npz）


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_object_array(x: Any) -> np.ndarray:
    """
    Wrap arbitrary python object (dict with numpy arrays etc.)
    into numpy object array for np.savez.
    """
    return np.array(x, dtype=object)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def save_frame(
    output: Dict[str, Any],
    save_dir: Path,
    frame_idx: int,
) -> Path:
    """
    Save ONE frame result.

    Args:
        output:
            dict for one frame (can include numpy arrays, lists, etc.)
        save_dir:
            directory to save into
        frame_idx:
            frame index (used in filename)

    Returns:
        Path to saved .npz
    """
    if not isinstance(output, dict):
        raise TypeError(f"output must be Dict[str, Any], got {type(output)}")

    _ensure_dir(save_dir)

    save_path = save_dir / f"{frame_idx:06d}_sam3d_body.npz"

    np.savez_compressed(
        save_path,
        output=_to_object_array(output),
    )

    logger.info(f"[SAVE] frame {frame_idx} -> {save_path}")
    return save_path
