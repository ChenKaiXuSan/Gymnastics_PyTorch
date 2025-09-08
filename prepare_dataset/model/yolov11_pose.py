#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import logging
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from tqdm import tqdm

from prepare_dataset.utils import process_none

logger = logging.getLogger(__name__)

# COCO 17 点骨架（keypoint 索引基于 Ultralytics 的 0-16）
COCO_SKELETON = [
    (5, 7),
    (7, 9),  # 左臂
    (6, 8),
    (8, 10),  # 右臂
    (5, 6),  # 肩膀连线
    (11, 13),
    (13, 15),  # 左腿
    (12, 14),
    (14, 16),  # 右腿
    (11, 12),  # 髋部
    (5, 11),
    (6, 12),  # 上身到下身
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),  # 头部
    (1, 2),  # 左右眼
]


class YOLOv11Pose:
    """
    使用 Ultralytics YOLO 的 pose 模型对视频帧序列进行关节点推理与可视化。
    返回: (pose: (T,17,2), pose_score: (T,17), none_index: List[int], results: List[Results])
    """

    def __init__(self, configs, person: str) -> None:
        super().__init__()
        self.yolo_pose = YOLO(configs.YOLO.pose_ckpt)
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

    # --------------------- public api ---------------------

    def __call__(self, vframes: torch.Tensor, video_path: Path):
        """
        vframes: (T,H,W,C) 或 (T,C,H,W)，dtype uint8/float*
        """
        frames_rgb = self._sanitize_frames_to_rgb_uint8(
            vframes
        )  # List[np.ndarray] RGB(H,W,3), uint8
        T = len(frames_rgb)
        _video_name = video_path.stem

        img_out_dir = self.save_path / "img" / "pose" / _video_name
        crop_out_dir = self.save_path / "img" / "pose_crop" / _video_name
        if self.save:
            img_out_dir.mkdir(parents=True, exist_ok=True)
            crop_out_dir.mkdir(parents=True, exist_ok=True)

        # 推理
        results = self._infer_pose(frames_rgb)

        # 聚合/选框与关键点
        none_index: List[int] = []
        bbox_prev: Optional[np.ndarray] = None  # (4,) xywh 上一帧选择的框
        pose_list: List[torch.Tensor] = []
        pose_score_list: List[torch.Tensor] = []

        with torch.inference_mode():
            for idx, r in tqdm(
                enumerate(results), total=T, desc="YOLO Pose", leave=False
            ):
                # YOLO 返回可能为多目标，我们需要在此**选定单人**（与上一帧一致、或最近中心）
                kpt_xy, kpt_conf, chosen_box = self._select_person_kpts(r, bbox_prev)

                if kpt_xy is None:
                    none_index.append(idx)
                    # 先放占位，稍后用 process_none 修补
                    pose_list.append(torch.empty((17, 2), dtype=torch.float32))
                    pose_score_list.append(torch.empty((17,), dtype=torch.float32))
                else:
                    pose_list.append(torch.as_tensor(kpt_xy, dtype=torch.float32))
                    pose_score_list.append(
                        torch.as_tensor(kpt_conf, dtype=torch.float32)
                    )
                    bbox_prev = chosen_box

                # 保存可视化
                # * 这里保存的是多个目标的结果（如果有的话），而非我们选定的单人
                if self.save:
                    try:
                        r.save(filename=str(img_out_dir / f"{idx}_pose.png"))
                    except Exception:
                        # 某些版本的 ultralytics Results.save 需要先设置 save_dir，这里保守处理
                        pass
                    try:
                        r.save_crop(
                            save_dir=str(crop_out_dir), file_name=f"{idx}_pose_crop.png"
                        )
                    except Exception:
                        pass

        # 用前后帧补洞
        if none_index:
            logger.warning(
                "Video %s has %d frames without pose, will interpolate using process_none.",
                video_path,
                len(none_index),
            )
            pose_dict = {
                i: (None if i in none_index else pose_list[i]) for i in range(T)
            }
            pose_score_dict = {
                i: (None if i in none_index else pose_score_list[i]) for i in range(T)
            }
            pose_dict = process_none(batch_Dict=pose_dict, none_index=none_index)
            pose_score_dict = process_none(
                batch_Dict=pose_score_dict, none_index=none_index
            )
            pose_list = [pose_dict[i] for i in range(T)]
            pose_score_list = [pose_score_dict[i] for i in range(T)]

        pose = torch.stack(pose_list, dim=0)  # (T,17,2)
        pose_score = torch.stack(pose_score_list, dim=0)  # (T,17)

        # 如果需要将关键点绘制保存（更可控的自绘）
        # * 这里保存的是单人关键点
        if self.save:
            try:
                self.draw_and_save_keypoints(
                    img_tensor=self._to_tensor_rgb01(frames_rgb),
                    keypoints=pose,
                    save_root=self.save_path,
                    video_path=video_path,
                )
            except Exception as e:
                logger.exception("draw_and_save_keypoints failed: %s", e)

        return pose, pose_score, none_index, results

    # --------------------- inference & selection ---------------------

    def _infer_pose(self, frames_rgb: List[np.ndarray]):
        """
        统一从 RGB 列表转 BGR 输入 YOLO；支持 tracking / 非 tracking。
        返回 list[Results]，保证可多次遍历（不返回生成器）。
        """
        frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]

        if self.tracking:
            stream = self.yolo_pose.track(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,  # 仅人类
                stream=True,
                verbose=self.verbose,
                device=self.device,
                imgsz=self.img_size,
            )
        else:
            stream = self.yolo_pose(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
                imgsz=self.img_size,
            )
        # 将生成器收集为 list，便于多次访问
        return list(stream)

    @staticmethod
    def _extract_xywh_boxes(r) -> Optional[np.ndarray]:
        """
        提取当前帧所有人的 xywh 框，返回 (N,4) np.float32 或 None
        """
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return None
        xywh = getattr(boxes, "xywh", None)
        if xywh is None or xywh.numel() == 0:
            return None
        return xywh.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _extract_keypoints(r) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        返回:
          - kpts_xy: (N,17,2) np.float32 或 None
          - kpts_conf: (N,17) np.float32 或 None
        """
        kpts = getattr(r, "keypoints", None)
        if kpts is None:
            return None, None
        xy = getattr(kpts, "xy", None)
        conf = getattr(kpts, "conf", None)
        if xy is None or xy.numel() == 0:
            return None, None
        kpts_xy = xy.detach().cpu().numpy().astype(np.float32)  # (N,17,2)
        kpts_conf = None
        if conf is not None and conf.numel() > 0:
            kpts_conf = conf.detach().cpu().numpy().astype(np.float32)  # (N,17)
        else:
            # 若无 conf，统一置 1
            kpts_conf = np.ones((kpts_xy.shape[0], kpts_xy.shape[1]), dtype=np.float32)
        return kpts_xy, kpts_conf

    def _select_person_kpts(
        self,
        r,
        bbox_prev_xywh: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        选择当前帧中**最合适的单人**并返回其关键点与框：
        - 优先：如果存在 track id，选择与上一帧同 id（若上一帧无 id，则保持最近中心）
        - 次优：与上一帧中心最近
        - 退化：仅有一个人则取第一个；没有人则返回 None
        Returns:
          (kpt_xy:(17,2) or None, kpt_conf:(17,) or None, bbox_xywh:(4,) or None)
        """
        boxes = self._extract_xywh_boxes(r)  # (N,4) or None
        kpts_xy_all, kpts_conf_all = self._extract_keypoints(r)  # (N,17,2), (N,17)

        if boxes is None or kpts_xy_all is None or boxes.shape[0] == 0:
            return None, None, None

        N = boxes.shape[0]
        if N == 1:
            return kpts_xy_all[0], kpts_conf_all[0], boxes[0]

        # 尝试利用 track id
        try:
            boxes_obj = getattr(r, "boxes", None)
            if boxes_obj is not None and getattr(boxes_obj, "is_track", False):
                track_ids = boxes_obj.id
                if track_ids is not None:
                    track_ids = (
                        track_ids.detach().cpu().numpy().astype(np.int32)
                    )  # (N,)
                    # 如果上一帧有 id，优先同 id
                    # 这里我们没有保存上一帧 id，因此退化为“与上一帧中心最近”，但保留接口
        except Exception:
            track_ids = None

        # 与上一帧中心最近（如果上一帧有框）
        if bbox_prev_xywh is not None:
            cx_prev, cy_prev = bbox_prev_xywh[0], bbox_prev_xywh[1]
            centers = boxes[:, :2]  # (N,2) → (cx, cy)
            dists = np.linalg.norm(
                centers - np.array([cx_prev, cy_prev], dtype=np.float32), axis=1
            )
            best = int(np.argmin(dists))
            return kpts_xy_all[best], kpts_conf_all[best], boxes[best]

        # 否则选择面积最大的（更稳一些）
        areas = boxes[:, 2] * boxes[:, 3]  # w*h
        best = int(np.argmax(areas))
        return kpts_xy_all[best], kpts_conf_all[best], boxes[best]

    # --------------------- drawing ---------------------

    def draw_and_save_keypoints(
        self,
        img_tensor: torch.Tensor,
        keypoints: torch.Tensor,
        save_root: Path,
        video_path: Path,
        radius: int = 3,
    ):
        """
        img_tensor: (T,H,W,3) RGB, float32 in [0,1]
        keypoints:  (T,17,2)
        """
        assert (
            img_tensor.dim() == 4 and img_tensor.shape[-1] == 3
        ), f"img_tensor shape invalid: {tuple(img_tensor.shape)}"
        assert (
            keypoints.dim() == 3
            and keypoints.shape[1] == 17
            and keypoints.shape[2] == 2
        ), f"keypoints shape invalid: {tuple(keypoints.shape)}"

        _video_name = video_path.stem
        out_dir = save_root / "filter_img" / "pose" / _video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        T, H, W, _ = img_tensor.shape

        for idx in tqdm(range(T), total=T, desc="Draw and Save Keypoints", leave=False):
            img = (
                (img_tensor[idx].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            )  # RGB
            kpt = keypoints[idx].cpu().numpy()  # (17,2)

            # 画点
            for x, y in kpt:
                if x > 0 and y > 0:
                    cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)

            # 画骨架
            for i, j in COCO_SKELETON:
                xi, yi = kpt[i]
                xj, yj = kpt[j]
                if xi > 0 and yi > 0 and xj > 0 and yj > 0:
                    cv2.line(
                        img, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 2
                    )

            cv2.imwrite(
                str(out_dir / f"{idx}_pose_filter.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

    # --------------------- frame utils ---------------------

    @staticmethod
    def _sanitize_frames_to_rgb_uint8(vframes: torch.Tensor) -> List[np.ndarray]:
        """
        接受 (T,H,W,C) 或 (T,C,H,W) 的 torch.Tensor，返回 List[np.ndarray]，RGB uint8
        """
        if not isinstance(vframes, torch.Tensor):
            raise TypeError(f"vframes must be a torch.Tensor, got {type(vframes)}")
        if vframes.dim() != 4:
            raise ValueError(f"vframes must be 4D, got {tuple(vframes.shape)}")

        t = vframes
        # 移到 CPU
        if t.is_cuda:
            t = t.detach().cpu()
        else:
            t = t.detach()

        # 转 float32 [0,1]
        if t.dtype == torch.uint8:
            t = t.float().div_(255.0)
        elif t.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            t = t.float()

        # 归一化到 [0,1]
        if t.dtype != torch.float32:
            t = t.float()
        t = t.clamp_(0, 1)

        # 通道顺序到 (T,H,W,3)
        if t.shape[1] in (1, 3) and t.shape[-1] not in (1, 3):  # (T,C,H,W) -> (T,H,W,C)
            t = t.permute(0, 2, 3, 1).contiguous()
        elif t.shape[-1] not in (1, 3):
            raise ValueError(f"vframes channel dimension ambiguous: {tuple(t.shape)}")

        # float32[0,1] -> uint8
        t = (t * 255.0).round().byte()  # (T,H,W,C)
        frames = [t[i].numpy() for i in range(t.shape[0])]
        return frames

    @staticmethod
    def _to_tensor_rgb01(frames_rgb_uint8: List[np.ndarray]) -> torch.Tensor:
        """
        List[np.ndarray uint8 RGB] -> (T,H,W,3) float32[0,1] tensor
        """
        if len(frames_rgb_uint8) == 0:
            return torch.empty((0, 0, 0, 3), dtype=torch.float32)
        arr = np.stack(frames_rgb_uint8, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr)
