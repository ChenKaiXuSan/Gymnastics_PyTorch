#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
from pathlib import Path
from typing import Sequence, Tuple, Optional

import torch

from prepare_dataset.model.yolov11_bbox import YOLOv11Bbox
from prepare_dataset.model.yolov11_pose import YOLOv11Pose
from prepare_dataset.model.yolov11_mask import YOLOv11Mask

logger = logging.getLogger(__name__)


class PreprocessYOLO:
    """
    封装 YOLOv11 bbox/pose/mask 的预处理流程。
    - 与旧版保持一样的 __call__ 返回：bbox_none_index, bbox, mask, pose, pose_score
    - 更健壮的错误处理 & 形状/设备对齐
    """

    def __init__(self, config, person: str) -> None:
        super().__init__()
        # 全部初始化
        self.yolo_model_bbox: Optional[YOLOv11Bbox] = YOLOv11Bbox(config, person=person)
        self.yolo_model_pose: Optional[YOLOv11Pose] = YOLOv11Pose(config, person=person)
        self.yolo_model_mask: Optional[YOLOv11Mask] = YOLOv11Mask(config, person=person)

        logger.debug(
            "PreprocessYOLO init: bbox=%s pose=%s mask=%s",
            self.yolo_model_bbox is not None,
            self.yolo_model_pose is not None,
            self.yolo_model_mask is not None,
        )

    def __call__(
        self,
        vframes: torch.Tensor,
        video_path: Path,
    ) -> Tuple[Sequence[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            vframes: (T,H,W,C) 或 (T,C,H,W)，dtype uint8/float*
            video_path: 对应视频路径（仅用于日志/缓存键等）

        Returns:
            (bbox_none_index, bbox, mask, pose, pose_score)
            - bbox: (N,4) 或 (T,4) 取决于下游实现
            - mask: (T,1,H,W)
            - pose: (T,17,3)
            - pose_score: (T,17)
        """
        vframes = self._sanitize_vframes(vframes)

        T, H, W = vframes.shape[0], vframes.shape[-2], vframes.shape[-1]
        device = vframes.device

        # 预置“安全的”空输出，保证即使子任务失败也有一致形状/设备的返回
        empty_bbox = torch.empty((0, 4), dtype=torch.float32, device=device)
        empty_pose = torch.empty((0, 17, 3), dtype=torch.float32, device=device)
        empty_pose_score = torch.empty((0, 17), dtype=torch.float32, device=device)
        empty_mask = torch.empty((0, 1, H, W), dtype=torch.float32, device=device)

        bbox_none_index: Sequence[int] = []
        bbox = empty_bbox
        pose = empty_pose
        pose_score = empty_pose_score
        mask = empty_mask

        # 统一无梯度推理
        with torch.inference_mode():
            # * BBox
            if self.yolo_model_bbox is not None:
                try:
                    bbox, bbox_none_index, _ = self.yolo_model_bbox(vframes, video_path)
                except Exception as e:
                    logger.exception("BBox preprocess failed for %s: %s", video_path, e)
                    bbox, bbox_none_index = empty_bbox, []

            # * Pose
            if self.yolo_model_pose is not None:
                try:
                    pose, pose_score, _, _ = self.yolo_model_pose(vframes, video_path)
                except Exception as e:
                    logger.exception("Pose preprocess failed for %s: %s", video_path, e)
                    pose, pose_score = empty_pose, empty_pose_score

            # * Mask
            if self.yolo_model_mask is not None:
                try:
                    mask, _, _ = self.yolo_model_mask(vframes, video_path)
                except Exception as e:
                    logger.exception("Mask preprocess failed for %s: %s", video_path, e)
                    mask = empty_mask

        self._log_shapes(bbox=bbox, mask=mask, pose=pose, pose_score=pose_score)
        return bbox_none_index, bbox, mask, pose, pose_score

    # ----------------- helpers -----------------

    @staticmethod
    def _sanitize_vframes(vframes: torch.Tensor) -> torch.Tensor:
        """
        - 必须是 4D： (T,H,W,C) 或 (T,C,H,W)
        - uint8 自动转 float32 并归一化到 [0,1]
        - 其他 dtype 保留精度（非 float → float32）
        """
        if not isinstance(vframes, torch.Tensor):
            raise TypeError(f"vframes must be torch.Tensor, got {type(vframes)}")
        if vframes.dim() != 4:
            raise ValueError(
                f"vframes must be 4D (T,H,W,C) or (T,C,H,W), got {tuple(vframes.shape)}"
            )

        if vframes.dtype == torch.uint8:
            vframes = vframes.float() / 255.0
        elif vframes.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            vframes = vframes.float()

        return vframes

    @staticmethod
    def _log_shapes(
        *,
        bbox: torch.Tensor,
        mask: torch.Tensor,
        pose: torch.Tensor,
        pose_score: torch.Tensor,
    ) -> None:
        try:
            logger.debug(
                "PreprocessYOLO outputs -> bbox:%s mask:%s pose:%s pose_score:%s",
                tuple(bbox.shape),
                tuple(mask.shape),
                tuple(pose.shape),
                tuple(pose_score.shape),
            )
        except Exception:
            pass
