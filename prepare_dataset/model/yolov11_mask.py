#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from ultralytics import YOLO
from tqdm import tqdm

from prepare_dataset.utils import process_none

logger = logging.getLogger(__name__)


def _to_thwc_rgb_uint8(vframes: torch.Tensor) -> torch.Tensor:
    """
    (T,H,W,C) or (T,C,H,W) -> (T,H,W,3) uint8 RGB (on CPU)
    """
    if not isinstance(vframes, torch.Tensor):
        raise TypeError(f"vframes must be torch.Tensor, got {type(vframes)}")
    if vframes.dim() != 4:
        raise ValueError(f"vframes must be 4D, got {tuple(vframes.shape)}")

    t = vframes.detach()
    if t.is_cuda:
        t = t.cpu()

    # 通道整理到最后一维
    if t.shape[-1] in (1, 3):            # (T,H,W,C)
        pass
    elif t.shape[1] in (1, 3):           # (T,C,H,W) -> (T,H,W,C)
        t = t.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"Ambiguous channels in shape {tuple(t.shape)}")

    # 灰度 -> RGB
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)

    # 统一到 uint8 [0,255]
    if t.dtype == torch.uint8:
        u8 = t
    else:
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            t = t.float()
        else:
            t = t.float()
        t = t.clamp(0, 255)
        if t.max() <= 1.5:  # 认为是 0..1
            t = t * 255.0
        u8 = t.round().byte()

    return u8  # (T,H,W,3) uint8


def _xyxy_center(box_xyxy: torch.Tensor) -> Tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy.tolist()
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def _areas_xyxy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    w = (x2 - x1).clamp_min(0)
    h = (y2 - y1).clamp_min(0)
    return w * h


class YOLOv11Mask:
    def __init__(self, configs, person: str) -> None:
        super().__init__()

        # load model
        self.yolo_mask = YOLO(configs.YOLO.seg_ckpt)
        self.tracking: bool = bool(configs.YOLO.tracking)

        self.conf: float = float(configs.YOLO.conf)
        self.iou: float = float(configs.YOLO.iou)
        self.verbose: bool = bool(configs.YOLO.verbose)
        self.img_size: int = int(configs.YOLO.img_size)

        self.device: str = str(configs.device)

        self.save: bool = bool(configs.YOLO.save)
        self.save_path: Path = Path(configs.extract_dataset.save_path) / "vis" / "yolo" / person

    @torch.inference_mode()
    def get_YOLO_mask_result(self, vframes: torch.Tensor) -> List:
        """
        将帧转为 RGB uint8 列表，再转 BGR 给 YOLO；返回 list[Results]
        """
        thwc_rgb_u8 = _to_thwc_rgb_uint8(vframes)
        frames_bgr = [cv2.cvtColor(thwc_rgb_u8[i].numpy(), cv2.COLOR_RGB2BGR)
                      for i in range(thwc_rgb_u8.shape[0])]

        if self.tracking:
            stream = self.yolo_mask.track(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,          # person
                stream=True,
                verbose=self.verbose,
                device=self.device,
                imgsz=self.img_size,
            )
        else:
            stream = self.yolo_mask(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
                imgsz=self.img_size,
            )
        return list(stream)

    @staticmethod
    def _resize_mask_to_orig(mask_hw: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """
        mask_hw: (h, w) 概率/二值
        orig_shape: (H, W)
        returns: (1, H, W) float32
        """
        m = mask_hw.unsqueeze(0).unsqueeze(0).float()      # (1,1,h,w)
        m = F.interpolate(m, size=orig_shape, mode="bilinear", align_corners=False)
        return m.squeeze(0)                                # (1,H,W)

    @torch.inference_mode()
    def draw_and_save_masks(
        self,
        img_tensor: torch.Tensor,     # (T,*,*,*)
        masks: torch.Tensor,          # (T,1,H,W) float
        save_path: Path,
        video_path: Path,
        alpha: float = 0.5,
        bin_thresh: float = 0.5,
    ):
        if not self.save:
            return

        _video_name = video_path.stem
        out_dir = save_path / "filter_img" / "mask" / _video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        thwc_rgb_u8 = _to_thwc_rgb_uint8(img_tensor)  # (T,H,W,3)
        T, H, W, _ = thwc_rgb_u8.shape

        for i in tqdm(range(T), total=T, desc="Draw and Save Masks", leave=False):
            img = thwc_rgb_u8[i].numpy()  # RGB uint8
            mask = masks[i].squeeze(0).detach().cpu().numpy()  # (H,W) float
            binary = (mask > bin_thresh).astype(np.uint8)

            # 构造彩色遮罩 (RGB: 0,255,0)
            color = np.zeros_like(img, dtype=np.uint8)
            color[:, :, 1] = binary * 255

            blended = cv2.addWeighted(img, 1.0, color, alpha, 0)
            cv2.imwrite(str(out_dir / f"{i}_mask_filter.jpg"),
                        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    @torch.inference_mode()
    def __call__(self, vframes: torch.Tensor, video_path: Path):
        """
        Returns:
            mask: (T,1,H,W) float32  (device 与 vframes 一致)
            none_index: List[int]
            results: List[Results]
        """
        out_device = vframes.device
        results = self.get_YOLO_mask_result(vframes)
        T = vframes.shape[0]

        none_index: List[int] = []
        mask_dict: dict[int, Optional[torch.Tensor]] = {}

        prev_box: Optional[torch.Tensor] = None  # xyxy on CPU
        prev_id: Optional[int] = None

        for idx, r in tqdm(enumerate(results), total=T, desc="YOLO Mask", leave=False):
            boxes_obj = getattr(r, "boxes", None)
            masks_obj = getattr(r, "masks", None)

            if boxes_obj is None or boxes_obj.shape[0] == 0 or masks_obj is None:
                none_index.append(idx)
                mask_dict[idx] = None
                continue

            boxes_xyxy = boxes_obj.xyxy.detach().cpu()  # (N,4)
            N = boxes_xyxy.shape[0]

            # 选择目标索引
            chosen_i: Optional[int] = None

            # 尝试优先同 track_id
            try:
                if getattr(boxes_obj, "is_track", False) and boxes_obj.id is not None and prev_id is not None:
                    ids = boxes_obj.id.detach().cpu().numpy().astype(np.int64).tolist()
                    if prev_id in ids:
                        chosen_i = ids.index(prev_id)
            except Exception:
                pass

            if chosen_i is None:
                if prev_box is not None and N > 0:
                    cx_prev, cy_prev = _xyxy_center(prev_box)
                    centers = (boxes_xyxy[:, 0:2] + boxes_xyxy[:, 2:4]) * 0.5
                    dists = torch.linalg.norm(centers - torch.tensor([cx_prev, cy_prev]), dim=1)
                    chosen_i = int(torch.argmin(dists))
                else:
                    # 首帧或丢失后重启：取面积最大的
                    areas = _areas_xyxy(boxes_xyxy)
                    chosen_i = int(torch.argmax(areas))

            # 取对应 mask 并 resize 到原图
            try:
                mask_hw = masks_obj.data[chosen_i]               # (h,w) float
                H, W = masks_obj.orig_shape                      # (H,W)
                mask_1hw = self._resize_mask_to_orig(mask_hw, (H, W))  # (1,H,W)
                mask_dict[idx] = mask_1hw.cpu()
            except Exception as e:
                logger.exception("Resize mask failed at frame %d: %s", idx, e)
                none_index.append(idx)
                mask_dict[idx] = None

            # 更新 prev
            prev_box = boxes_xyxy[chosen_i]
            try:
                prev_id = int(boxes_obj.id[chosen_i].item()) if boxes_obj.id is not None else None
            except Exception:
                prev_id = None

            # YOLO 内置保存（仅在需要时）
            if self.save:
                try:
                    save_dir = self.save_path / "img" / "mask" / video_path.stem
                    crop_dir = self.save_path / "img" / "crop_mask" / video_path.stem
                    save_dir.mkdir(parents=True, exist_ok=True)
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    r.save(filename=str(save_dir / f"{idx}_mask.png"))
                    r.save_crop(save_dir=str(crop_dir), file_name=f"{idx}_mask_crop.png")
                except Exception:
                    # 某些版本的 ultralytics Results.save 需要先设置 save_dir，这里保守忽略错误
                    pass

        # 缺失帧回填
        if len(none_index) > 0:
            logger.warning("Video %s has %d frames without mask.", video_path.stem, len(none_index))
            mask_dict = process_none(mask_dict, none_index)

        # 字典 -> 张量 (T,1,H,W)，并放回原 device
        mask_cpu = torch.stack([mask_dict[k] for k in sorted(mask_dict.keys())], dim=0)  # CPU
        mask = mask_cpu.to(out_device).float()

        # 可视化（叠色保存）
        if self.save:
            try:
                self.draw_and_save_masks(
                    img_tensor=vframes,
                    masks=mask_cpu,          # 画图用 CPU
                    save_path=self.save_path,
                    video_path=video_path,
                )
            except Exception as e:
                logger.exception("draw_and_save_masks failed for %s: %s", video_path.stem, e)

        return mask, none_index, results
