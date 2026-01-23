import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from .infer import process_frame_list
from .load import load_data

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def find_files(
    subject_dir: Path,
    patterns: List[str],
    recursive: bool = False,
) -> List[Path]:
    """指定されたディレクトリ配下でパターンに一致するファイルを検索"""
    files: List[Path] = []
    search_func = subject_dir.rglob if recursive else subject_dir.glob
    for pat in patterns:
        files.extend(search_func(pat))
    return sorted({f.resolve() for f in files})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    infer_type = cfg.infer.get("type", "video")  # video or image
    recursive = bool(cfg.dataset.get("recursive", False))

    # 共通の出力パス設定
    out_root = Path(cfg.paths.log_path).resolve()
    inference_output_path = Path(cfg.paths.result_output_path).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    inference_output_path.mkdir(parents=True, exist_ok=True)

    # 検索パターンの定義
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]
    img_patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

    # ---------------------------------------------------------------------- #
    # 入力ソースの決定とスキャン
    # ---------------------------------------------------------------------- #
    if infer_type == "video":
        data_root = Path(cfg.paths.video_path).resolve()
        patterns = vid_patterns
        logger.info(f"Mode: VIDEO | Root: {data_root}")
    else:
        data_root = Path(cfg.paths.image_path).resolve()
        patterns = img_patterns
        logger.info(f"Mode: IMAGE | Root: {data_root}")

    if not data_root.exists():
        raise FileNotFoundError(f"Path not found: {data_root}")

    # Subject（サブフォルダ）の取得
    subjects_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not subjects_dirs:
        # 直下にファイルがある可能性も考慮（サブフォルダがない場合）
        subjects_dirs = [data_root]

    # { subject_name: [file paths] }
    data_map: Dict[str, List[Path]] = {}
    for s_dir in subjects_dirs:
        found_files = find_files(s_dir, patterns, recursive)
        if found_files:
            data_map[s_dir.name] = found_files
        else:
            logger.warning(f"[No files found] {s_dir}")

    # ---------------------------------------------------------------------- #
    # タスクの構築
    # ---------------------------------------------------------------------- #
    tasks: List[Tuple[str, str, Path]] = []

    for subject_name, file_list in data_map.items():
        frame_list = load_data(file_list)
        tasks.append((subject_name, frame_list))

    logger.info(f"Total tasks found: {len(tasks)}")

    # ---------------------------------------------------------------------- #
    # 准备数据
    # ---------------------------------------------------------------------- #
    for subject_name, frame_list in tasks:
        logger.info(
            f"Processing: [{infer_type.upper()}] {subject_name} | Number of frames: {len(frame_list)}"
        )

        # 保存先の階層を作成
        current_out_dir = out_root / subject_name
        current_infer_dir = inference_output_path / subject_name

        process_frame_list(
            frame_list=frame_list,
            out_dir=current_out_dir,
            inference_output_path=current_infer_dir,
            cfg=cfg,
        )

    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
