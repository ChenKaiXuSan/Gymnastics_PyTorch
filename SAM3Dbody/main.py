#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/main_multi_gpu_process.py
Project: /workspace/code/SAM3Dbody
Created Date: Monday January 26th 2026
Author: Kaixu Chen
-----
Comment:
根据多GPU并行处理SAM-3D-Body推理任务。

Have a good code time :)
-----
Last Modified: Monday January 26th 2026 5:12:10 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# 假设这些是从你的其他模块导入的
from .infer import process_frame_list
from .load import load_data

# --- 常量定义 ---
REQUIRED_VIEWS = {"face", "side"}

logger = logging.getLogger(__name__)


ROTATE_CODES = {
    "ROTATE_90_CLOCKWISE": cv2.ROTATE_90_CLOCKWISE,
    "ROTATE_180": cv2.ROTATE_180,
    "ROTATE_90_COUNTERCLOCKWISE": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def maybe_rotate_frames(frames: List[np.ndarray], cfg: DictConfig) -> List[np.ndarray]:
    """Optionally rotate input frames before SAM3D-Body inference."""
    if not cfg.infer.get("rotate_frames", False):
        return frames

    rotate_code_name = cfg.infer.get("rotate_code", "ROTATE_90_CLOCKWISE")
    if rotate_code_name not in ROTATE_CODES:
        raise ValueError(
            f"Unsupported rotate_code '{rotate_code_name}'. "
            f"Expected one of: {', '.join(ROTATE_CODES)}"
        )

    return [cv2.rotate(frame, ROTATE_CODES[rotate_code_name]) for frame in frames]


# ---------------------------------------------------------------------
# 核心处理逻辑：处理单个人的数据
# ---------------------------------------------------------------------


def process_single_person(
    person_dir: Path,
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
):
    """处理单个人员的所有环境和视角（右回転処理を追加）"""
    person_id = person_dir.name
    vid_extensions = {".mp4", ".mov", ".avi", ".mkv"}

    # --- 1. Person専用のログ設定 ---
    log_dir = out_root / "person_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    person_log_file = log_dir / f"{person_id}.log"

    logger = logging.getLogger(person_id)
    logger.setLevel(logging.INFO)

    # 既存のハンドラがあればクリア（重複防止）
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(person_log_file, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"==== Starting Process for Person: {person_id} ====")

    # --- 2. 視角（View）のフィルタリング ---
    view_map: Dict[str, Path] = {}
    # person_dir 直下のファイルを走査
    for file_path in person_dir.iterdir():
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix in vid_extensions:
            view_name = file_path.name.lower()
            for v in REQUIRED_VIEWS:
                if v in view_name:
                    view_map[v] = file_path.resolve()

    # 必要な視角が揃っているかチェック
    missing_views = [v for v in REQUIRED_VIEWS if v not in view_map]
    if missing_views:
        logger.warning(f"[Skip] {person_id}: 視角が足りません。不足: {missing_views}")
        handler.close()  # 終了前に閉じる
        return

    # --- 3. データロードと回転処理 ---
    # view_frames: Dict[str, List[np.ndarray]]
    view_frames = load_data(view_map)

    for view_label, frames in view_frames.items():
        # logger.info(
        #     f" 視角 {view_label} を処理中: {len(frames)} 枠。右回転を適用します。"
        # )

        rotated_frames = maybe_rotate_frames(frames, cfg)

        # 保存先の作成
        _out_root = out_root / person_id / view_label
        _out_root.mkdir(parents=True, exist_ok=True)
        _infer_root = infer_root / person_id / view_label
        _infer_root.mkdir(parents=True, exist_ok=True)

        # 回転済みのリストを次の処理へ渡す
        process_frame_list(
            frame_list=rotated_frames,
            out_dir=_out_root,
            inference_output_path=_infer_root,
            cfg=cfg,
        )

    logger.info(f"==== Finished Person: {person_id} ====")
    handler.close()  # ログファイルを安全に閉じる


# ---------------------------------------------------------------------
# GPU Worker：进程执行函数
# ---------------------------------------------------------------------
def gpu_worker(
    gpu_id: int,
    person_dirs: List[Path],
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg_dict: dict,
):
    """
    每个进程的入口：设置环境变量，并处理分配的任务列表
    """
    # 1. 隔离 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg_dict["infer"]["gpu"] = 0  # 因为上面已经隔离了 GPU，所以这里设为 0

    # 2. 将字典转回 Hydra 配置（多进程传递对象时，转为字典更安全）
    cfg = OmegaConf.create(cfg_dict)

    logger.info(f"🟢 GPU {gpu_id} 进程启动，待处理人数: {len(person_dirs)}")

    for p_dir in person_dirs:
        try:
            process_single_person(p_dir, source_root, out_root, infer_root, cfg)
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} 处理 {p_dir.name} 时出错: {e}")

    logger.info(f"🏁 GPU {gpu_id} 所有任务处理完毕")


# ---------------------------------------------------------------------
# Main 入口
# ---------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. 経路準備
    out_root = Path(cfg.paths.log_path).resolve()
    infer_root = Path(cfg.paths.result_output_path).resolve()
    source_root = Path(cfg.paths.video_path).resolve()

    # --- 設定の追加 ---
    gpu_ids = cfg.infer.get("gpu", [0, 1])  # 使用するGPUのリスト
    workers_per_gpu = cfg.infer.get("workers_per_gpu", 2)  # 1枚あたりのプロセス数

    # 実際に起動するプロセスの数だけGPU IDを並べる (例: [0, 0, 1, 1])
    expanded_gpu_ids = []
    for gid in gpu_ids:
        expanded_gpu_ids.extend([gid] * workers_per_gpu)

    total_workers = len(expanded_gpu_ids)
    # ------------------

    # all_person_dirs = sorted([x for x in source_root.iterdir() if x.is_dir()])
    all_person_dirs = []
    for x in source_root.iterdir():
        if x.is_dir() and (
            int(x.name) in [int(pid) for pid in cfg.infer.person_list]
            or -1 in cfg.infer.person_list
        ):
            all_person_dirs.append(x)

    if not all_person_dirs:
        logger.error(f"未找到数据目录: {source_root}")
        return

    # 2. 自動分组逻辑 (プロセスの総数で分割)
    chunks = np.array_split(all_person_dirs, total_workers)

    logger.info(f"使用 GPU: {gpu_ids} (各 {workers_per_gpu} ワーカー)")
    logger.info(f"総プロセス数: {total_workers}")
    logger.info(f"総処理人数: {len(all_person_dirs)}")

    # 3. 启动并行进程
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mp.set_start_method("spawn", force=True)

    processes = []
    for i, gpu_id in enumerate(expanded_gpu_ids):
        person_list = chunks[i].tolist()
        if not person_list:
            continue

        logger.info(f"  - Worker {i} (GPU {gpu_id}) 分配任务数: {len(person_list)}")

        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                person_list,
                source_root,
                out_root,
                infer_root,
                cfg_dict,
            ),
        )
        p.start()
        processes.append(p)

    # 4. 等待所有进程完成
    for p in processes:
        p.join()

    logger.info("🎉 [SUCCESS] 所有 GPU 任务已圆满完成！")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
