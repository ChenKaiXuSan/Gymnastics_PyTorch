#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/load.py
Project: /workspace/code/triangulation
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 10:22:44 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Optional, Tuple, Dict, Any
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_video

import logging

logger = logging.getLogger(__name__)


def _load_pt(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    logger.info(f"Loading: {file_path}")
    return torch.load(file_path, map_location="cpu")


def _get_frames_and_shape(
    data: Dict[str, Any],
    *,
    want_frames: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[int]]:
    """
    返回 (frames, H, W)。若 want_frames=False，只返回 H/W（优先从 frames 或 img_shape 获取）。
    查找顺序：
      1) data["frames"]
      2) data["frames_path"]
      3) data["img_shape"]  # (H, W)
      4) data["video_path"] -> read_video
    """
    frames = None
    H = W = None

    # 1) 直接在 pt 里
    if "frames" in data and isinstance(data["frames"], torch.Tensor):
        frames = data["frames"]
        if frames.dim() == 4:
            H, W = int(frames.shape[1]), int(frames.shape[2])

    # 2) 外置路径
    if frames is None and isinstance(data.get("frames_path"), str):
        try:
            frames = torch.load(data["frames_path"], map_location="cpu")
            if frames.dim() == 4:
                H, W = int(frames.shape[1]), int(frames.shape[2])
        except Exception as e:
            logger.warning(f"Failed to load frames_path: {data['frames_path']} ({e})")

    # 3) 元数据里的 img_shape
    if (H is None or W is None) and isinstance(data.get("img_shape"), (tuple, list)):
        try:
            H, W = int(data["img_shape"][0]), int(data["img_shape"][1])
        except Exception:
            pass

    # 4) 退回视频路径解码（仅在需要帧或还没有 H/W 时）
    if (want_frames and frames is None) or (H is None or W is None):
        vp = data.get("video_path", None)
        if (
            want_frames
            and frames is None
            and isinstance(vp, str)
            and os.path.exists(vp)
        ):
            frames = read_video(vp, pts_unit="sec", output_format="THWC")[0]
            H, W = int(frames.shape[1]), int(frames.shape[2])
        elif (H is None or W is None) and isinstance(vp, str) and os.path.exists(vp):
            # 只需要 H/W 也可以读，但这里避免再次解码整视频；保持 None
            pass

    return frames, H, W


def _to_numpy_xy(
    kpts: torch.Tensor,
) -> np.ndarray:
    """
    接受 (T,K,2|3|>=2) 的张量，返回 numpy (T,K,2) 的 x,y。
    """
    if not isinstance(kpts, torch.Tensor):
        kpts = torch.as_tensor(kpts)
    if kpts.dim() < 3 or kpts.shape[-1] < 2:
        raise ValueError(
            f"Invalid keypoints shape: {tuple(kpts.shape)} (expect (T,K,≥2))"
        )
    xy = kpts[..., :2].cpu().numpy()
    return xy


def _maybe_denorm_xy(
    xy: np.ndarray,
    H: Optional[int],
    W: Optional[int],
    assume_normalized: Optional[bool] = None,
) -> np.ndarray:
    """
    若认为 xy 是归一化（[0,1]），且提供了 H/W，则转为像素坐标。
    assume_normalized:
      - True: 强制按归一化处理
      - False: 强制当像素处理
      - None: 自动判断（max<=1.5 视为归一化）
    """
    if H is None or W is None:
        return xy  # 无法反归一化，原样返回

    if assume_normalized is None:
        m = float(np.nanmax(xy)) if xy.size else 0.0
        is_norm = m <= 1.5  # 宽松判断
    else:
        is_norm = assume_normalized

    if is_norm:
        xy = xy.copy()
        xy[..., 0] *= W
        xy[..., 1] *= H
    return xy


def _extract_scores(
    data_block: Dict[str, Any],
    fallback_kpts: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    优先使用 data_block['keypoints_score'] -> (T,K)。
    若不存在且 fallback_kpts[...,2] 存在，则取其为分数；否则全 1。
    """
    ks = data_block.get("keypoints_score", None)
    if isinstance(ks, torch.Tensor):
        return ks.cpu().numpy()
    if isinstance(ks, np.ndarray):
        return ks

    if (
        isinstance(fallback_kpts, torch.Tensor)
        and fallback_kpts.dim() >= 3
        and fallback_kpts.shape[-1] >= 3
    ):
        return fallback_kpts[..., 2].cpu().numpy()

    # 默认全 1
    if isinstance(fallback_kpts, torch.Tensor):
        T, K = int(fallback_kpts.shape[0]), int(fallback_kpts.shape[1])
    else:
        # 不知道 T,K 时返回空；上层可据此决定
        return np.array([])
    return np.ones((T, K), dtype=np.float32)


# ------------------ 对外 API：分别加载 YOLO / Detectron2 ------------------ #


def load_keypoints_from_yolo_pt(
    file_path: str,
    *,
    return_frames: bool = False,
    assume_normalized: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    """
    从 pt_info 中读取 YOLO 关键点，返回 (keypoints_xy, keypoints_score, frames)。
      - keypoints_xy: (T,K,2) 的 numpy，若可判定为归一化且提供 H/W，则已转像素坐标
      - keypoints_score: (T,K) 的 numpy；若 pt 内没有，则尝试从 kpts[...,2]；再不行则全 1
      - frames: 若 return_frames=True 则返回 (T,H,W,C) 的 torch.uint8，否则 None
    """
    data = _load_pt(file_path)
    if "YOLO" not in data or "keypoints" not in data["YOLO"]:
        raise KeyError(f"pt file missing YOLO.keypoints: {file_path}")

    frames, H, W = _get_frames_and_shape(data, want_frames=return_frames)
    kpts_t: torch.Tensor = data["YOLO"]["keypoints"]
    xy = _to_numpy_xy(kpts_t)
    xy = _maybe_denorm_xy(xy, H, W, assume_normalized=assume_normalized)

    scores = _extract_scores(data["YOLO"], fallback_kpts=kpts_t)

    # 基本形状检查
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(f"Invalid YOLO keypoints shape after processing: {xy.shape}")
    if scores.size and (
        scores.shape[0] != xy.shape[0] or scores.shape[1] != xy.shape[1]
    ):
        raise ValueError(
            f"YOLO keypoints_score shape {scores.shape} mismatches keypoints {xy.shape}"
        )

    return xy, scores, frames


def load_kpt_and_bbox_from_d2_pt(
    file_path: str,
    *,
    return_frames: bool = False,
    assume_normalized: Optional[bool] = None,
    clip_bbox_to_image: bool = True,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    """
    从 Detectron2 的 pt_info 中读取关键点与 bbox。
    返回: (keypoints_xy, keypoints_score, bboxes_xyxy, bbox_scores, frames)

    - keypoints_xy: (T, K, 2)  像素坐标 (若可判定为归一化且有 H/W，则已反归一化)
    - keypoints_score: (T, K)  若无则从 kpts[...,2] 回退；再无则全 1
    - bboxes_xyxy: (T, N, 4)   每行为 [x1, y1, x2, y2]（同样会按需反归一化）
    - bbox_scores: (T, N)      若无则全 1
    - frames: (T, H, W, C) 的 torch.uint8，若 return_frames=False 则为 None
    """
    data = _load_pt(file_path)
    if "detectron2" not in data:
        raise KeyError(f"pt file missing 'detectron2' root: {file_path}")

    d2 = data["detectron2"]

    # --------- frames / shape ----------
    frames, H, W = _get_frames_and_shape(data, want_frames=return_frames)

    # --------- keypoints ----------
    if "keypoints" not in d2:
        raise KeyError(f"pt file missing detectron2.keypoints: {file_path}")

    kpts_t: torch.Tensor = d2["keypoints"]  # 预期形状 (T, K, 3?) or (T, K, 2)
    kpts_xy = _to_numpy_xy(kpts_t)  # -> (T, K, 2), numpy
    kpts_xy = _maybe_denorm_xy(kpts_xy, H, W, assume_normalized=assume_normalized)

    kpt_scores = _extract_scores(d2, fallback_kpts=kpts_t)  # -> (T, K) 或空
    if kpt_scores.size == 0:
        kpt_scores = np.ones(kpts_xy.shape[:2], dtype=dtype)

    # 形状/类型检查
    if kpts_xy.ndim != 3 or kpts_xy.shape[2] != 2:
        raise ValueError(
            f"Invalid D2 keypoints shape after processing: {kpts_xy.shape}"
        )
    if kpt_scores.shape != kpts_xy.shape[:2]:
        raise ValueError(
            f"D2 keypoints_score shape {kpt_scores.shape} mismatches keypoints {kpts_xy.shape}"
        )

    # --------- bboxes ----------
    if "bbox" not in d2:
        raise KeyError(f"pt file missing detectron2.bbox: {file_path}")

    bboxes_t: torch.Tensor = d2["bbox"]  # 预期 (T, N, 4)
    bboxes_xyxy = bboxes_t.detach().cpu().numpy().astype(dtype, copy=False)

    # 归一化 -> 像素
    # 使用 nanmax 更稳健：容忍个别 NaN
    max_val = np.nanmax(bboxes_xyxy) if bboxes_xyxy.size else 0.0
    if assume_normalized is True or (assume_normalized is None and max_val <= 1.5):
        bboxes_xyxy[..., 0::2] *= float(W)  # x1, x2
        bboxes_xyxy[..., 1::2] *= float(H)  # y1, y2

    # 分数
    if "scores" in d2:
        bbox_scores = d2["scores"].detach().cpu().numpy().astype(dtype, copy=False)
    elif "bbox_score" in d2:
        bbox_scores = d2["bbox_score"].detach().cpu().numpy().astype(dtype, copy=False)
    else:
        bbox_scores = np.ones(bboxes_xyxy.shape[:2], dtype=dtype)

    # 形状/类型检查
    if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[-1] != 4:
        raise ValueError(f"Invalid D2 bbox shape after processing: {bboxes_xyxy.shape}")
    if bbox_scores.shape != bboxes_xyxy.shape[:2]:
        raise ValueError(
            f"BBox score shape {bbox_scores.shape} mismatches bbox {bboxes_xyxy.shape}"
        )

    # 可选：将 bbox 裁剪到图像范围内，并保证 x1<=x2, y1<=y2
    if clip_bbox_to_image and H is not None and W is not None:
        # 排序确保 x1<=x2, y1<=y2
        x1 = np.minimum(bboxes_xyxy[..., 0], bboxes_xyxy[..., 2])
        x2 = np.maximum(bboxes_xyxy[..., 0], bboxes_xyxy[..., 2])
        y1 = np.minimum(bboxes_xyxy[..., 1], bboxes_xyxy[..., 3])
        y2 = np.maximum(bboxes_xyxy[..., 1], bboxes_xyxy[..., 3])
        bboxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)

        # 裁剪
        bboxes_xyxy[..., 0] = np.clip(bboxes_xyxy[..., 0], 0, W - 1)
        bboxes_xyxy[..., 2] = np.clip(bboxes_xyxy[..., 2], 0, W - 1)
        bboxes_xyxy[..., 1] = np.clip(bboxes_xyxy[..., 1], 0, H - 1)
        bboxes_xyxy[..., 3] = np.clip(bboxes_xyxy[..., 3], 0, H - 1)

    # 视频的frame向右旋转90°
    frames= np.rot90(frames, k=-1, axes=(1, 2)).copy()

    return (
        kpts_xy.astype(dtype, copy=False),
        kpt_scores.astype(dtype, copy=False),
        bboxes_xyxy.astype(dtype, copy=False),
        bbox_scores.astype(dtype, copy=False),
        frames,
    )


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
