#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
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
    Returns:
        Tuple[List[Dict], np.ndarray]: Lightweight per-frame metadata containing
        only ``frame_idx`` and ``pred_keypoints_3d``, plus the stacked keypoint
        sequence.
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

    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            output = data["output"].item()
            frame_idx = int(np.asarray(output["frame_idx"]).item())
            pred_keypoints_3d = np.array(
                output["pred_keypoints_3d"], copy=True
            )

        frame_info = {
            "frame_idx": frame_idx,
            "pred_keypoints_3d": pred_keypoints_3d,
        }
        all_info.append(frame_info)
        all_kpts.append(pred_keypoints_3d)

    kpts3d = np.stack(all_kpts, axis=0)

    return all_info, kpts3d
