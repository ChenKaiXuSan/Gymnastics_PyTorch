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


def save_results(
    outputs: List[Dict[str, Any]],
    save_dir: Path,
) -> None:
    """Save all results including mesh files and visualizations."""

    # FIXME: 需要修复一下
    np.savez_compressed(
        str(save_dir) + "_sam_3d_body_outputs.npz",
        outputs,
    )
    logger.info(f"Saved outputs: {save_dir / f'sam_3d_body_outputs.npz'}")
