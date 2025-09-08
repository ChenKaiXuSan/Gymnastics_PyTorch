#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import DPTForDepthEstimation, DPTImageProcessor

logger = logging.getLogger(__name__)


def _to_thwc_rgb_uint8(vframes: torch.Tensor) -> torch.Tensor:
    """
    接受 (T,H,W,C) 或 (T,C,H,W) 的 torch.Tensor，返回 (T,H,W,3) uint8 RGB。
    """
    if not isinstance(vframes, torch.Tensor):
        raise TypeError(f"vframes must be torch.Tensor, got {type(vframes)}")
    if vframes.dim() != 4:
        raise ValueError(f"vframes must be 4D, got {tuple(vframes.shape)}")

    t = vframes
    # 通道归一到最后一维
    if t.shape[-1] in (1, 3):  # (T,H,W,C)
        pass
    elif t.shape[1] in (1, 3):  # (T,C,H,W) -> (T,H,W,C)
        t = t.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"Ambiguous channels in shape {tuple(t.shape)}")

    # 若是灰度 -> 3 通道
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)

    # uint8 / float* -> uint8[0,255]
    if t.dtype == torch.uint8:
        pass
    else:
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            t = t.float()
        else:
            t = t.float()
        # 允许 0..1 或 0..255，都 clamp 后判断范围
        t = t.clamp(0, 255)
        if t.max() <= 1.5:
            t = (t * 255.0).round()
        t = t.byte()

    return t  # (T,H,W,3) uint8


def _frames_to_pils(frames_thwc_u8: torch.Tensor) -> List[Image.Image]:
    """
    (T,H,W,3) uint8 -> List[PIL.Image] (RGB)
    """
    frames_np = frames_thwc_u8.cpu().numpy()  # (T,H,W,3) uint8
    return [
        Image.fromarray(frames_np[i], mode="RGB") for i in range(frames_np.shape[0])
    ]


def _minmax_uint8(x: np.ndarray) -> np.ndarray:
    """
    x: 2D float32/float64 -> 0..255 的 uint8；若常数图，直接全零
    """
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    rng = x_max - x_min
    if rng <= 1e-6 or not np.isfinite(rng):
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - x_min) / rng
    y = (y * 255.0).clip(0, 255).astype(np.uint8)
    return y


class DepthEstimator:
    """
    HuggingFace DPT 深度估计（批处理）
    - 输入：vframes (T,H,W,C) 或 (T,C,H,W)，uint8/float*
    - 输出：depth (T,1,H,W)  float32
    """

    def __init__(self, configs, person: str) -> None:
        model_name = configs.depth_estimator.model
        self.device = torch.device(configs.device)

        # processor/model
        self.processor = DPTImageProcessor.from_pretrained(
            model_name, cache_dir=configs.depth_estimator.dpt_processor_path
        )
        self.model = (
            DPTForDepthEstimation.from_pretrained(
                model_name, cache_dir=configs.depth_estimator.dpt_model_path
            )
            .to(self.device)
            .eval()
        )

        # 选项
        self.save = bool(configs.depth_estimator.save)
        self.save_path = Path(configs.extract_dataset.save_path) / "vis" / "depth" / person
        self.batch_size = int(getattr(configs.depth_estimator, "batch_size", 8))
        self.amp = (
            bool(getattr(configs.depth_estimator, "amp", True))
            and torch.cuda.is_available()
        )

    @torch.no_grad()
    def estimate_depth_batch(self, pil_batch: List[Image.Image]) -> torch.Tensor:
        """
        对一批 PIL RGB 图像估计深度。
        返回 (B,1,H_orig,W_orig) 的 float32 张量（已插值回原分辨率）。
        """
        if len(pil_batch) == 0:
            return torch.empty((0, 1, 0, 0), dtype=torch.float32)

        # 记录原始尺寸，便于插值回去
        sizes = [img.size[::-1] for img in pil_batch]  # PIL: (W,H) -> (H,W)

        inputs = self.processor(images=pil_batch, return_tensors="pt").to(self.device)
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.amp
        ):
            outputs = self.model(**inputs)
            pred = outputs.predicted_depth  # (B,H',W')

        # 插值回原尺寸
        pred = pred.unsqueeze(1)  # (B,1,H',W')
        # 若同批尺寸一致，可一次性插值；否则逐张处理
        if len(set(sizes)) == 1:
            H, W = sizes[0]
            pred_up = F.interpolate(
                pred, size=(H, W), mode="bicubic", align_corners=False
            )
        else:
            pred_up_list = []
            for i, (H, W) in enumerate(sizes):
                pred_up_list.append(
                    F.interpolate(
                        pred[i : i + 1],
                        size=(H, W),
                        mode="bicubic",
                        align_corners=False,
                    )
                )
            pred_up = torch.cat(pred_up_list, dim=0)

        return pred_up.float()  # (B,1,H,W)

    def save_image(
        self, depth_1hw: torch.Tensor, save_dir: Path, idx: int, video_name: str
    ) -> None:
        """
        保存单帧深度为 8bit 灰度 PNG（min-max 归一到 0..255）
        depth_1hw: (1,H,W) float32
        """
        out_dir = save_dir / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        arr = depth_1hw.squeeze(0).detach().cpu().numpy()
        png = _minmax_uint8(arr)
        Image.fromarray(png, mode="L").save(out_dir / f"{idx}_depth.png")

    def __call__(self, vframes: torch.Tensor, video_path: Path) -> torch.Tensor:
        """
        Args:
            vframes: (T,H,W,C) 或 (T,C,H,W) (uint8/float*)
        Returns:
            depth: (T,1,H,W) float32
        """
        # 统一到 RGB uint8
        thwc_u8 = _to_thwc_rgb_uint8(vframes)  # (T,H,W,3) uint8
        T, H0, W0, _ = thwc_u8.shape

        # 批推理（转 PIL 列表再分块）
        pils = _frames_to_pils(thwc_u8)

        preds: List[torch.Tensor] = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, T, self.batch_size), desc="Depth Estimation", leave=False
            ):
                end = min(start + self.batch_size, T)
                batch_pils = pils[start:end]
                pred_b1hw = self.estimate_depth_batch(batch_pils)  # (B,1,H,W)
                preds.append(pred_b1hw.cpu())  # 累积到 CPU，降低显存峰值

        depth = torch.cat(preds, dim=0)  # (T,1,H,W)

        # 保存可视化
        if self.save and depth.numel() > 0:
            save_dir = self.save_path
            vname = video_path.stem
            for i in tqdm(range(T), desc="Save depth images", leave=False):
                try:
                    self.save_image(depth[i], save_dir, i, vname)
                except Exception as e:
                    logger.exception(
                        "Save depth image failed at %s #%d: %s", vname, i, e
                    )

        return depth
