#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.io import write_png
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _to_fchw_uint01(frames: torch.Tensor) -> torch.Tensor:
    """
    接受 (F,H,W,C) 或 (F,C,H,W) 或 (C,F,H,W?) 的误传，统一成 (F,C,H,W) float32 in [0,1].
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be torch.Tensor, got {type(frames)}")
    t = frames
    if t.dim() != 4:
        raise ValueError(f"frames must be 4D, got {tuple(t.shape)}")

    # 统一到 (F,C,H,W)
    if t.shape[-1] in (1, 3):  # (F,H,W,C)
        t = t.permute(0, 3, 1, 2).contiguous()
    elif t.shape[1] in (1, 3):  # (F,C,H,W) already OK
        pass
    else:
        # 容错：如果是 (C,F,H,W)（常见误传），尝试纠正
        if t.shape[0] in (1, 3):
            t = t.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError(f"Cannot infer channel dim from shape {tuple(t.shape)}")

    if t.dtype == torch.uint8:
        t = t.float().div_(255.0)
    elif t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        t = t.float()

    # clamp 到 [0,1]，避免异常值
    if t.dtype != torch.float32:
        t = t.float()
    t = t.clamp_(0, 1)
    return t


def _pad_to_multiple(
    x: torch.Tensor, multiple: int = 8
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    对 (N,C,H,W) 进行对称 pad 到 H,W 为 `multiple` 的倍数。
    返回: padded, (pad_left, pad_right, pad_top, pad_bottom)
    """
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x = torch.nn.functional.pad(
        x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
    )
    return x, (pad_left, pad_right, pad_top, pad_bottom)


def _unpad(x: torch.Tensor, pads: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    撤销 _pad_to_multiple 的 pad；输入 (N,2,H,W)
    """
    pl, pr, pt, pb = pads
    if pl == pr == pt == pb == 0:
        return x
    return x[:, :, pt : x.shape[2] - pb, pl : x.shape[3] - pr]


class OpticalFlow(nn.Module):
    """
    RAFT (torchvision) 光流估计器
    - 输入: frames (F,H,W,C) 或 (F,C,H,W), uint8/float*
    - 输出: flow (F-1, 2, H, W)  对应相邻帧 (t -> t+1)
    """

    def __init__(self, param, person: str) -> None:
        super().__init__()
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()  # 会做标准化/resize 等（内部支持）
        self.device = torch.device(param.device)

        # 模型
        self.model = (
            raft_large(weights=self.weights, progress=False).to(self.device).eval()
        )

        # 保存选项
        self.save: bool = bool(param.optical_flow.save)
        self.save_path: Path = (
            Path(param.extract_dataset.save_path) / "vis" / "optical_flow" / person
        )

        # 推理分块 & 步长（可按需从 param 读取）
        self.chunk_size: int = getattr(
            param.optical_flow, "chunk_size", 1
        )  # 每次处理多少对
        self.stride: int = getattr(param.optical_flow, "stride", 1)  # 相邻距离，默认 1

        # AMP 开关（仅 CUDA 时有效）
        self.use_amp: bool = (
            bool(getattr(param.optical_flow, "amp", True)) and torch.cuda.is_available()
        )

    @torch.no_grad()
    def get_optical_flow(self, frames_fchw01: torch.Tensor) -> torch.Tensor:
        """
        核心推理：
        输入: (F,C,H,W) float32 in [0,1]
        输出: (F-1,2,H,W) float32
        """
        device = self.device
        frames = frames_fchw01.to(device, non_blocking=True)

        F, C, H, W = frames.shape
        num_pairs = max(0, F - self.stride)
        if num_pairs <= 0:
            logger.warning(
                "Not enough frames for optical flow: F=%d, stride=%d", F, self.stride
            )
            return torch.empty((0, 2, H, W), dtype=torch.float32, device=device)

        flows = []
        # 分块处理 pair 索引 [0, num_pairs-1]
        indices = torch.arange(0, num_pairs, dtype=torch.long, device=device)

        for start in tqdm(
            range(0, num_pairs, self.chunk_size),
            desc="Predict optical flow",
            leave=False,
        ):
            end = min(start + self.chunk_size, num_pairs)
            batch_idx = indices[start:end]  # (B,)

            # (B,C,H,W)
            img1 = frames[batch_idx]  # t
            img2 = frames[batch_idx + self.stride]  # t+stride

            # torchvision 的 RAFT weights.transforms 接受 (B,C,H,W) 两个张量
            img1_t, img2_t = self.transforms(img1, img2)

            # 尺寸 pad 到 8 的倍数（RAFT 更稳）
            img1_t, pads = _pad_to_multiple(img1_t, 8)
            img2_t, _ = _pad_to_multiple(img2_t, 8)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                flow_pred = self.model(img1_t, img2_t)[-1]  # (B,2,H',W')

            flow_pred = _unpad(flow_pred, pads)  # (B,2,H,W)
            flows.append(flow_pred.float().cpu())  # 累积到 CPU，降低显存峰值

            # 释放临时
            del img1, img2, img1_t, img2_t, flow_pred

        flow = torch.cat(flows, dim=0)  # (F-1,2,H,W)
        return flow

    @staticmethod
    def save_image(flow: torch.Tensor, save_dir: Path, video_name: str) -> None:
        """
        将 (N,2,H,W) 光流保存为彩色 PNG（`flow_to_image`），文件名 0..N-1
        """
        out_dir = save_dir / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        n = flow.shape[0]
        for i in tqdm(range(n), desc="Save optical flow", leave=False):
            # flow_to_image: (2,H,W) -> (3,H,W) uint8
            img = flow_to_image(flow[i].cpu())
            # write_png 需要 (C,H,W) 的 uint8
            write_png(img, str(out_dir / f"{i}_flow.png"))

    def __call__(self, frames: torch.Tensor, video_path: Path) -> torch.Tensor:
        """
        Args:
            frames: (F,H,W,C) 或 (F,C,H,W), uint8/float*
        Returns:
            flow: (F-1, 2, H, W) float32
        """
        # 统一输入
        frames_fchw01 = _to_fchw_uint01(frames)

        # 推理
        with torch.inference_mode():
            flow = self.get_optical_flow(frames_fchw01)

        # 保存可视化
        if self.save and flow.numel() > 0:
            try:
                self.save_image(flow, self.save_path, video_name=video_path.stem)
            except Exception as e:
                logger.exception(
                    "Failed to save optical flow images for %s: %s", video_path, e
                )

        return flow
