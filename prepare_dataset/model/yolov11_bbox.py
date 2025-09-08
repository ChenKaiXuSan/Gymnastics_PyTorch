#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import logging
import torch
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
    if t.shape[-1] in (1, 3):  # (T,H,W,C)
        pass
    elif t.shape[1] in (1, 3):  # (T,C,H,W) -> (T,H,W,C)
        t = t.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"Ambiguous channels in shape {tuple(t.shape)}")

    # 灰度→RGB
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)

    # uint8/float* -> uint8[0,255]
    if t.dtype == torch.uint8:
        u8 = t
    else:
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            t = t.float()
        else:
            t = t.float()
        t = t.clamp(0, 255)
        if t.max() <= 1.5:
            t = t * 255.0
        u8 = t.round().byte()

    return u8  # (T,H,W,3) uint8 on CPU


def _xyxy_center(box_xyxy: torch.Tensor) -> Tuple[float, float]:
    """
    box_xyxy: (4,) -> (cx, cy)
    """
    x1, y1, x2, y2 = box_xyxy.tolist()
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def _areas_xyxy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """(N,4) -> (N,) areas"""
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    w = (x2 - x1).clamp_min(0)
    h = (y2 - y1).clamp_min(0)
    return w * h


class YOLOv11Bbox:
    def __init__(self, configs, person: str) -> None:
        super().__init__()

        # load model
        self.yolo_bbox = YOLO(configs.YOLO.bbox_ckpt)
        self.tracking: bool = bool(configs.YOLO.tracking)

        self.conf: float = float(configs.YOLO.conf)
        self.iou: float = float(configs.YOLO.iou)
        self.verbose: bool = bool(configs.YOLO.verbose)
        self.device: str = str(configs.device)

        self.img_size: int = int(configs.YOLO.img_size)

        self.save: bool = bool(configs.YOLO.save)
        self.save_path: Path = (
            Path(configs.extract_dataset.save_path) / "vis" / "yolo" / person
        )

    @torch.inference_mode()
    def get_YOLO_bbox_result(self, vframes: torch.Tensor) -> List:
        """
        统一把帧转成 RGB uint8 list，再转 BGR 给 YOLO；返回 list[Results]
        """
        thwc_rgb_u8 = _to_thwc_rgb_uint8(vframes)  # (T,H,W,3) uint8
        frames_bgr = [
            cv2.cvtColor(
                thwc_rgb_u8[i].numpy(), cv2.COLOR_RGB2BGR
            )  # list[np.ndarray BGR]
            for i in range(thwc_rgb_u8.shape[0])
        ]

        if self.tracking:
            stream = self.yolo_bbox.track(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,  # person
                stream=True,
                verbose=self.verbose,
                device=self.device,  # NOTE: Ultralytics 会解析字符串设备；如需精确绑核，可在外层设置 CUDA_VISIBLE_DEVICES
                imgsz=self.img_size,
            )
        else:
            stream = self.yolo_bbox.predict(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
                imgsz=self.img_size,
            )

        # return list(stream)  # 立刻收集，避免生成器被多次遍历的问题
        return  stream

    @torch.inference_mode()
    def draw_and_save_boxes(
        self,
        img_tensor: torch.Tensor,
        bboxes: torch.Tensor,
        save_path: Path,
        video_path: Path,
    ):
        """
        img_tensor: (T,H,W,C) 或 (T,C,H,W)，float/uint8，RGB
        bboxes: (T,4) xyxy on CPU
        """
        if not self.save:
            return

        _video_name = video_path.stem
        out_dir = save_path / "filter_img" / "bbox" / _video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        thwc_rgb_u8 = _to_thwc_rgb_uint8(img_tensor)  # (T,H,W,3) uint8 on CPU
        T = thwc_rgb_u8.shape[0]

        for i in tqdm(range(T), total=T, desc="Draw and Save BBoxes", leave=False):
            img = thwc_rgb_u8[i].numpy().copy()  # RGB uint8
            x1, y1, x2, y2 = map(int, bboxes[i].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(
                str(out_dir / f"{i}_bbox_filter.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

    @torch.inference_mode()
    def __call__(self, vframes: torch.Tensor, video_path: Path):
        """
        Returns:
            bbox: (T,4) xyxy  (device 与 vframes 一致)
            none_index: List[int]
            results: List[Results]
        """
        # device 对齐：最终把 bbox 放回输入 device
        out_device = vframes.device

        # 推理
        results = self.get_YOLO_bbox_result(vframes)
        T = vframes.shape[0]

        none_index: List[int] = []
        bbox_dict: dict[int, Optional[torch.Tensor]] = {}

        prev_box: Optional[torch.Tensor] = None  # xyxy on CPU

        for idx, r in tqdm(enumerate(results), total=T, desc="YOLO BBox", leave=False):
            boxes_obj = getattr(r, "boxes", None)
            if boxes_obj is None or boxes_obj.shape[0] == 0:
                none_index.append(idx)
                bbox_dict[idx] = None
                continue

            boxes_xyxy = boxes_obj.xyxy.detach().cpu()  # (N,4)
            N = boxes_xyxy.shape[0]

            # 单框：直接取
            if N == 1:
                chosen = boxes_xyxy[0]
                bbox_dict[idx] = chosen
                prev_box = chosen
                continue

            # 多框：优先 track_id 匹配；否则与上一帧中心最近；再退化为面积最大
            chosen: Optional[torch.Tensor] = None
            try:
                if (
                    getattr(boxes_obj, "is_track", False)
                    and boxes_obj.id is not None
                    and prev_box is not None
                ):
                    # 这里没有上一帧 id 保持机制，退化为"中心最近"
                    pass
            except Exception:
                pass

            if prev_box is not None:
                cx_prev, cy_prev = _xyxy_center(prev_box)
                centers = (boxes_xyxy[:, 0:2] + boxes_xyxy[:, 2:4]) * 0.5  # (N,2)
                dists = torch.linalg.norm(
                    centers - torch.tensor([cx_prev, cy_prev]), dim=1
                )
                best = int(torch.argmin(dists))
                chosen = boxes_xyxy[best]
            else:
                # 首帧或丢失后重启：取面积最大
                areas = _areas_xyxy(boxes_xyxy)
                best = int(torch.argmax(areas))
                chosen = boxes_xyxy[best]

            bbox_dict[idx] = chosen
            prev_box = chosen

        # 缺失补齐
        if len(none_index) > 0:
            logger.warning(
                "Video %s has %d frames without bbox.", video_path.stem, len(none_index)
            )
            bbox_dict = process_none(bbox_dict, none_index)

        # 字典 -> 张量 (T,4)，并放回原 device
        bbox_cpu = torch.stack(
            [bbox_dict[k] for k in sorted(bbox_dict.keys())], dim=0
        )  # CPU
        bbox = bbox_cpu.to(out_device)

        # 可视化
        if self.save:
            try:
                self.draw_and_save_boxes(
                    img_tensor=vframes,
                    bboxes=bbox_cpu,  # 画图用 CPU
                    save_path=self.save_path,
                    video_path=video_path,
                )
            except Exception as e:
                logger.exception(
                    "draw_and_save_boxes failed for %s: %s", video_path.stem, e
                )

        return bbox, none_index, results
