#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/sam3d.py
Project: /workspace/code/sam3d_body
Created Date: Thursday December 4th 2025
Author: Kaixu Chen
-----
Comment:
先推理SAM-3D-Body模型，然后根据面积最大的 bbox 选出主要人物进行可视化和保存

Have a good code time :)
-----
Last Modified: Thursday December 4th 2025 4:24:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
from pathlib import Path
import numpy as np

import torch
from omegaconf.omegaconf import DictConfig
from tqdm import tqdm

from .sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from .sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from .sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from .save import save_frame
from .vis import (
    vis_results,
)

logger = logging.getLogger(__name__)


# 定义一个计算面积的函数（基于 bbox: [x1, y1, x2, y2]）
def select_best_person(outputs, verbose=True):
    """
    从多个检测结果中选出面积最大（置信度最高）的一个。

    Args:
        outputs (list): 模型输出的 dict 列表.
        verbose (bool): 是否打印每个检测框的信息。

    Returns:
        tuple: (best_target, best_idx) 如果没有结果则返回 (None, None)
    """
    if not outputs:
        if verbose:
            print("⚠️ 未检测到任何目标。")
        return None, None

    areas = []
    for i, obj in enumerate(outputs):
        # 获取 bbox 坐标 [x1, y1, x2, y2]
        bbox = obj.get("bbox", [0, 0, 0, 0])

        # 计算面积
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        areas.append(area)

        if verbose:
            print(
                f"检测到序号 [{i}]: 面积 = {area:10.2f} | BBox = [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
            )

    # 找到面积最大的索引
    best_idx = np.argmax(areas)
    best_target = outputs[best_idx]

    if verbose:
        print(f"🏆 最终选定: {best_idx} 号 (面积: {areas[best_idx]:.2f})")
        print("-" * 50)

    return best_target, best_idx


def setup_visualizer():
    """Set up skeleton visualizer with MHR70 pose info"""
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


def setup_sam_3d_body(
    cfg: DictConfig,
):
    # 如果参数为空，则从环境变量中读取
    mhr_path = cfg.model.get("mhr_path", "") or os.environ.get("SAM3D_MHR_PATH", "")
    # helper params
    detector_name = cfg.model.get("detector_name", "vitdet")
    segmentor_name = cfg.model.get("segmentor_name", "")
    fov_name = cfg.model.get("fov_name", "moge2")
    detector_path = cfg.model.get("detector_path", "") or os.environ.get(
        "SAM3D_DETECTOR_PATH", ""
    )
    segmentor_path = cfg.model.get("segmentor_path", "") or os.environ.get(
        "SAM3D_SEGMENTOR_PATH", ""
    )
    fov_path = cfg.model.get("fov_path", "") or os.environ.get("SAM3D_FOV_PATH", "")

    # -------------------- 初始化主模型 -------------------- #
    sam3d_model, model_cfg = load_sam_3d_body(
        cfg.model.checkpoint_path,
        device=cfg.infer.gpu,
        mhr_path=mhr_path,
    )

    # -------------------- 可选模块：detector / segmentor / fov -------------------- #
    human_detector = None
    human_segmentor = None
    fov_estimator = None

    if detector_name:
        from .tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=detector_name,
            device=cfg.infer.gpu,
            path=detector_path,
        )

    if segmentor_name:
        from .tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name,
            device=cfg.infer.gpu,
            path=segmentor_path,
        )

    if fov_name:
        from .tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=fov_name,
            device=cfg.infer.gpu,
            path=fov_path,
        )

    # 挂到成员变量上
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=sam3d_model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    logger.info("==== SAM 3D Body Estimator Setup ====")
    logger.info(f"  Model checkpoint: {cfg.model.checkpoint_path}")
    logger.info(f"  MHR model path: {mhr_path if mhr_path else 'Default'}")
    logger.info(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    logger.info(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    logger.info(
        f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}"
    )
    return estimator


# ------------------------------------------------------------------ #
# 高级接口（处理文件夹等）
# ------------------------------------------------------------------ #
def process_frame_list(
    frame_list: list,
    out_dir: Path,
    inference_output_path: Path,
    cfg: DictConfig,
):
    """处理单个视频文件的镜头编辑。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    inference_output_path.mkdir(parents=True, exist_ok=True)

    # 初始化模型与可视化器
    estimator = setup_sam_3d_body(cfg)
    visualizer = setup_visualizer()

    for idx in tqdm(range(len(frame_list)), desc="Processing frames"):
        # if idx > 1:
        #     break

        outputs = estimator.process_one_image(
            img=frame_list[idx],
            bboxes=None,
        )

        # 在处理 outputs 时进行筛选， 选出面积最大的那个人
        # 输出最大面积的信息
        best_person, best_id = select_best_person(outputs)

        if best_person is None:
            logger.warning(f"[Skip] No person detected in frame {idx}.")
            continue

        # 可视化并保存结果
        vis_results(
            img_cv2=frame_list[idx],
            outputs=[best_person],
            save_dir=str(out_dir / "visualization" / f"frame_{idx:04d}"),
            image_name=f"frame_{idx:04d}",
            faces=estimator.faces,
            visualizer=visualizer,
            cfg=cfg.visualize,
        )

        outputs = best_person
        outputs["frame"] = frame_list[idx]
        outputs["frame_idx"] = idx

        save_frame(
            output=outputs,
            save_dir=inference_output_path,
            frame_idx=outputs["frame_idx"],
        )

    # final
    torch.cuda.empty_cache()
    del estimator
    del visualizer

    return out_dir
