#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/merge_video.py
Project: /workspace/code/triangulation/vis
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 12:53:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import cv2
from pathlib import Path

import numpy as np
from typing import Iterable, Sequence, Union, Tuple, Optional


def merge_frames_to_video(
    frame_dir: str, output_video_path: str, fps: int = 30
) -> None:
    """
    将指定目录下的图像帧合并为视频。
    frame_dir: 图像帧目录，假设命名格式为 frame_0000.png, frame_0001.png, ...
    output_video_path: 输出视频路径
    fps: 视频帧率
    """

    frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frame_files:
        raise ValueError(f"No frames found in directory: {frame_dir}")

    # 读取第一张图片以获取尺寸信息
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}, skipping.")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")


def save_frames_to_video(
    frames: Union[Sequence[np.ndarray], Iterable[np.ndarray]],
    out_path: Union[str, Path],
    fps: float = 30.0,
    codec: str = "mp4v",  # 常用：mp4v (mp4), XVID (avi), MJPG (avi), avc1/H264(需要系统支持)
    color_space: str = "RGB",  # 你的帧是否是 RGB；若已是 BGR 就设 "BGR"
    resize_to: Optional[Tuple[int, int]] = None,  # (W,H)，不设则以第一帧尺寸为准
    keep_ratio_with_pad: bool = False,  # True 时按比例缩放并用黑边填充到目标尺寸
) -> Tuple[int, Tuple[int, int]]:
    """
    将帧序列写入视频文件。
    返回：写入的帧数、最终视频分辨率 (W,H)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 取第一帧，确定尺寸
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("frames 为空，无法写视频。")

    if first is None:
        raise ValueError("第一帧为 None。")

    # 统一成 uint8
    if first.dtype != np.uint8:
        first = np.clip(first, 0, 255).astype(np.uint8)

    # 灰度转 BGR（VideoWriter 期望 3 通道）
    if first.ndim == 2 or (first.ndim == 3 and first.shape[2] == 1):
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)

    # RGB -> BGR
    if color_space.upper() == "RGB":
        first = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)

    # 目标尺寸
    if resize_to is None:
        H0, W0 = first.shape[:2]
        Wt, Ht = W0, H0
    else:
        Wt, Ht = resize_to

    def _resize_frame_bgr(img_bgr: np.ndarray) -> np.ndarray:
        # 统一 dtype & 通道
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
        if img_bgr.ndim == 2 or (img_bgr.ndim == 3 and img_bgr.shape[2] == 1):
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        if (img_bgr.shape[1], img_bgr.shape[0]) == (Wt, Ht):
            return img_bgr

        if keep_ratio_with_pad:
            # 按比例缩放 + 居中填充黑边
            h, w = img_bgr.shape[:2]
            scale = min(Wt / w, Ht / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((Ht, Wt, 3), dtype=np.uint8)
            x0 = (Wt - nw) // 2
            y0 = (Ht - nh) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
            return canvas
        else:
            return cv2.resize(img_bgr, (Wt, Ht), interpolation=cv2.INTER_AREA)

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (Wt, Ht))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器，路径：{out_path}，codec={codec}")

    # 写入第一帧
    writer.write(_resize_frame_bgr(first))

    # 其余帧
    n_written = 1
    for frm in it:
        if frm is None:
            continue
        img = frm
        # 统一通道 & 颜色
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if color_space.upper() == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = _resize_frame_bgr(img)
        writer.write(img)
        n_written += 1

    writer.release()
    return n_written, (Wt, Ht)
