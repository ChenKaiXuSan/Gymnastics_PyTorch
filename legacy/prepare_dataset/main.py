#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Optional

import hydra
import torch
from torchvision.io import read_video

from prepare_dataset.process.preprocess import Preprocess

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


# --------------------------- 工具函数 ---------------------------

def _is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith("._") and not p.name.startswith(".")

def _iter_videos(root: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file() and _is_video(p))
    else:
        yield from (p for p in root.iterdir() if p.is_file() and _is_video(p))

def _target_pt_path(save_root: Path, person: str, video_stem: str) -> Path:
    return save_root / "pt" / person / f"{video_stem}.pt"

def _safe_save_pt(pt_path: Path, obj: Dict[str, Any], legacy_zip: bool = True) -> None:
    """原子写入，避免半文件；legacy_zip=True 使用旧序列化以降低内存峰值。"""
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pt_path.with_suffix(pt_path.suffix + ".tmp")
    torch.save(obj, tmp, _use_new_zipfile_serialization=not legacy_zip)
    os.replace(tmp, pt_path)
    try:
        size_mb = os.path.getsize(pt_path) / 1024 / 1024
        logger.info("Saved: %s (%.1f MB)", pt_path, size_mb)
    except Exception:
        logger.info("Saved: %s", pt_path)

def _rotate_thwc(frames: torch.Tensor, deg: int) -> torch.Tensor:
    """对 THWC 张量旋转，deg ∈ {0, 90, 180, 270, -90}；返回 THWC。"""
    if deg % 360 == 0:
        return frames
    # torch.rot90: k>0 逆时针；右转90°即 k=-1
    mapping = {90: 1, 180: 2, 270: 3, -90: -1}
    k = mapping.get(deg % 360 if deg > 0 else deg)
    if k is None:
        raise ValueError(f"rotate_deg must be one of {{0, 90, 180, 270, -90}}, got {deg}")
    return torch.rot90(frames, k=k, dims=(1, 2)).contiguous()


# --------------------------- 单视频：整段一次性推理 ---------------------------

def process_video_whole(
    config,
    person: str,
    video_path: Path,
    *,
    embed_frames_in_pt: bool = True,
) -> Dict[str, Any]:
    """
    整段视频一次性读取 + 推理，返回 pt_info。
    注意：长视频内存占用较高，可在 config 关闭 embed_frames_in_pt。
    """
    try:
        vframes, _, info = read_video(video_path, pts_unit="sec", output_format="THWC")
    except Exception as e:
        raise RuntimeError(f"read_video failed on {video_path}: {e}")

    if not isinstance(vframes, torch.Tensor) or vframes.numel() == 0:
        raise RuntimeError(f"Empty/invalid video tensor: {video_path}")

    # 确保是 uint8、THWC
    if vframes.dtype != torch.uint8:
        vframes = vframes.to(torch.uint8)
    vframes = vframes.contiguous()

    T, H, W, C = vframes.shape

    # 旋转（从 config.extract_dataset.rotate_deg 读取，默认不旋转）
    rotate_deg = -90
    if rotate_deg:
        vframes = _rotate_thwc(vframes, rotate_deg)
        # 旋转后 H/W 可能互换
        if rotate_deg % 180 != 0:
            H, W = W, H

    # 元信息（兼容不同 torchvision 版本的 info 字段）
    fps = float(info.get("video_fps", 0.0)) if isinstance(info, dict) else 0.0
    duration = float(info.get("duration", 0.0)) if isinstance(info, dict) else 0.0

    logger.info("Read %s | T=%d, HxW=%dx%d, C=%d, fps=%.3f, dur=%.3fs, rot=%d°",
                video_path.name, T, H, W, C, fps, duration, rotate_deg)

    # 推理（保持 THWC 传入；若你的 Preprocess 期望 TCHW，则在内部再 permute）
    preprocess = Preprocess(config=config, person=person)
    with torch.inference_mode():
        pt_info = preprocess(vframes, video_path)

    # 附加元信息
    pt_info["video_name"] = video_path.stem
    pt_info["video_path"] = str(video_path)
    pt_info["frame_count"] = T
    pt_info["img_shape"] = (H, W)
    pt_info["fps"] = fps
    pt_info["duration"] = duration
    pt_info["rotate_deg"] = rotate_deg

    if embed_frames_in_pt:
        pt_info["frames"] = vframes.cpu()  # 体积较大，按需开启

    return pt_info


# --------------------------- 按 person 处理 & Hydra 入口 ---------------------------

def process_one_person(config, person: str) -> None:
    raw_root = Path(config.extract_dataset.data_path)
    save_root = Path(config.extract_dataset.save_path)
    recursive: bool = bool(getattr(config.extract_dataset, "recursive", True))
    overwrite: bool = bool(getattr(config.extract_dataset, "overwrite", False))
    embed_frames_in_pt: bool = bool(getattr(config.extract_dataset, "embed_frames_in_pt", True))
    legacy_zip: bool = bool(getattr(config.extract_dataset, "legacy_zip", True))

    person_dir = raw_root / person
    if not person_dir.exists():
        logger.warning("Person dir not found: %s", person_dir)
        return

    num_threads: int = int(getattr(getattr(config, "system", object()), "num_threads", 0) or 0)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        logger.info("torch.set_num_threads(%d)", num_threads)

    logger.info("Start person=%s (recursive=%s overwrite=%s embed_frames=%s)",
                person, recursive, overwrite, embed_frames_in_pt)

    for video_path in _iter_videos(person_dir, recursive=recursive):
        out_pt = _target_pt_path(save_root, person, video_path.stem)
        if out_pt.exists() and not overwrite:
            logger.info("Skip existed: %s", out_pt)
            continue

        try:
            pt_info = process_video_whole(
                config,
                person,
                video_path,
                embed_frames_in_pt=embed_frames_in_pt,
            )
            _safe_save_pt(out_pt, pt_info, legacy_zip)

        except Exception as e:
            logger.exception("Failed on %s: %s", video_path, e)

        finally:
            # 释放内存，避免长跑累积
            try:
                del pt_info
            except Exception:
                pass
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@hydra.main(config_path="../configs/", config_name="prepare_dataset", version_base=None)
def main(config):
    persons_root = Path(getattr(config, "input_path", "/workspace/data/raw"))
    for person_dir in persons_root.iterdir():
        if person_dir.is_dir():
            process_one_person(config, person_dir.name)


if __name__ == "__main__":
    main()
