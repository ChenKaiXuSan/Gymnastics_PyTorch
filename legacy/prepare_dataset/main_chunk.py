#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Chunked preprocessing for large videos with low RAM footprint.

- Streaming decode via torchvision.io.VideoReader
- 1-frame overlap across chunks to keep temporal continuity
- Rotate entire chunk after stacking (THWC) for correctness & speed
- Accumulate results in lists and torch.cat once at the end
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import hydra
import torch
from torchvision.io import VideoReader, read_video

from prepare_dataset.process.preprocess import Preprocess

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


# --------------------------- utils ---------------------------


def _iter_videos(root: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from (
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )
    else:
        yield from (
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )


def _safe_save_pt(pt_path: Path, obj: Dict[str, Any], legacy_zip: bool = True) -> None:
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
    """Rotate THWC tensor by deg in {0, 90, 180, 270, -90}."""
    if not deg or deg % 360 == 0:
        return frames
    mapping = {90: 1, 180: 2, 270: 3, -90: -1}
    k = mapping.get(deg % 360 if deg > 0 else deg)
    if k is None:
        raise ValueError(f"rotate_deg must be one of {{0,90,180,270,-90}}, got {deg}")
    return torch.rot90(frames, k=k, dims=(1, 2)).contiguous()


# --------------------------- core ---------------------------


def process_video_chunked(
    config,
    person: str,
    video_path: Path,
    chunk_size: int = 128,
) -> Dict[str, Any]:
    """
    Low-RAM streaming process with 1-frame overlap between chunks.

    - Per-chunk: [overlap(1)] + [chunk_size] frames  -> stack -> (t,H,W,C) uint8
    - Rotate after stacking to THWC if needed
    - For per-frame outputs (depth/YOLO/d2): drop the first frame of subsequent chunks
    - For optical flow (T-1): keep the entire block (includes the cross-chunk pair)
    """
    preprocess = Preprocess(config=config, person=person)
    logger.info("Processing video: %s", video_path)

    # read meta (robust)
    raw_vframes, _, info = read_video(video_path, pts_unit="sec", output_format="THWC")
    fps = float(info.get("video_fps", 0.0))
    duration = float(info.get("duration", 0.0))
    vr = VideoReader(str(video_path), "video")

    vr.set_current_stream("video")
    rotate_deg = int(getattr(config.extract_dataset, "rotate_deg", -90))

    # accumulators (use lists -> cat once)
    total_T = 0
    first_chunk = True
    last_frame: Optional[torch.Tensor] = None  # HWC uint8

    depth_list: List[torch.Tensor] = []  # each: (t,1,h,w)
    flow_list: List[torch.Tensor] = []  # each: (t-1,2,h,w)
    none_idx: List[int] = []

    yolo_bbox: List[torch.Tensor] = []  # each: (t,4)
    yolo_mask: List[torch.Tensor] = []  # each: (t,1,h,w)
    yolo_kpt: List[torch.Tensor] = []  # each: (t,17,3)
    yolo_kpt_score: List[torch.Tensor] = []  # each: (t,17)

    d2_bbox: List[torch.Tensor] = []
    d2_kpt: List[torch.Tensor] = []
    d2_kpt_score: List[torch.Tensor] = []

    H = W = None  # determined after first chunk
    # FIXME: 这个写有问题，推到的时候会多出来几帧，具体是根据chunk size分的batch来的
    with torch.inference_mode():
        while True:
            buf: List[torch.Tensor] = []

            if not first_chunk and last_frame is not None:
                buf.append(last_frame)  # 1-frame overlap (HWC)

            # pull up to chunk_size frames
            for _ in range(chunk_size):
                try:
                    fr = next(vr)  # dict: data(C,H,W), pts, ...
                except StopIteration:
                    break
                # convert to HWC
                hwc = fr["data"].permute(1, 2, 0).contiguous()  # uint8
                buf.append(hwc)

            if len(buf) == 0 or (not first_chunk and len(buf) == 1):
                break  # nothing new

            # prepare next overlap
            last_frame = buf[-1].clone()

            # stack to (t,H,W,C)
            vframes = torch.stack(buf, dim=0)  # CPU, uint8

            # rotate whole chunk if required
            if rotate_deg:
                vframes = _rotate_thwc(vframes, rotate_deg)

            t, h, w, _ = vframes.shape
            if H is None:
                H, W = h, w

            # run preprocess once per chunk (expects THWC)
            out = preprocess(vframes, video_path)

            # shift local none_index to global
            local_none = list(out.get("none_index", []))
            if first_chunk:
                none_idx.extend(local_none)
            else:
                # we will drop frame-0 outputs for per-frame branches
                none_idx.extend([total_T + i - 1 for i in local_none if i >= 1])

            # per-frame branches: drop the first frame on subsequent chunks
            sl = slice(None) if first_chunk else slice(1, None)

            dep = out.get("depth", torch.empty(0))
            if isinstance(dep, torch.Tensor) and dep.numel() > 0:
                depth_list.append(dep[sl].cpu())

            flo = out.get("optical_flow", torch.empty(0))
            if isinstance(flo, torch.Tensor) and flo.numel() > 0:
                flow_list.append(flo.cpu())  # keep full, includes cross-chunk pair

            y = out.get("YOLO", {})
            if y:
                if isinstance(y.get("bbox"), torch.Tensor) and y["bbox"].numel() > 0:
                    yolo_bbox.append(y["bbox"][sl].cpu())
                if isinstance(y.get("mask"), torch.Tensor) and y["mask"].numel() > 0:
                    yolo_mask.append(y["mask"][sl].cpu())
                if (
                    isinstance(y.get("keypoints"), torch.Tensor)
                    and y["keypoints"].numel() > 0
                ):
                    yolo_kpt.append(y["keypoints"][sl].cpu())
                if (
                    isinstance(y.get("keypoints_score"), torch.Tensor)
                    and y["keypoints_score"].numel() > 0
                ):
                    yolo_kpt_score.append(y["keypoints_score"][sl].cpu())

            d2 = out.get("detectron2", {})
            if d2:
                if isinstance(d2.get("bbox"), torch.Tensor) and d2["bbox"].numel() > 0:
                    d2_bbox.append(d2["bbox"][sl].cpu())
                if (
                    isinstance(d2.get("keypoints"), torch.Tensor)
                    and d2["keypoints"].numel() > 0
                ):
                    d2_kpt.append(d2["keypoints"][sl].cpu())
                if (
                    isinstance(d2.get("keypoints_score"), torch.Tensor)
                    and d2["keypoints_score"].numel() > 0
                ):
                    d2_kpt_score.append(d2["keypoints_score"][sl].cpu())

            # update global frame count (per-frame added t or t-1)
            total_T += t if first_chunk else (t - 1)
            first_chunk = False

    # finalize concat (handle empties)
    def _maybe_cat(seq, dim=0):
        return torch.cat(seq, dim=dim) if len(seq) > 0 else torch.empty(0)

    depth = _maybe_cat(depth_list, dim=0)  # (T,1,H,W) or (0)
    flow = _maybe_cat(flow_list, dim=0)  # (T-1,2,H,W) or (0)
    y_bbox = _maybe_cat(yolo_bbox, dim=0)  # (T,4) or (0)
    y_mask = _maybe_cat(yolo_mask, dim=0)  # (T,1,H,W) or (0)
    y_kpt = _maybe_cat(yolo_kpt, dim=0)  # (T,17,3) or (0)
    y_kpt_s = _maybe_cat(yolo_kpt_score, dim=0)  # (T,17) or (0)
    d_bbox = _maybe_cat(d2_bbox, dim=0)
    d_kpt = _maybe_cat(d2_kpt, dim=0)
    d_kpt_s = _maybe_cat(d2_kpt_score, dim=0)

    # * check shapes
    logger.info(
        "check shapes: depth=%s flow=%s yolo_bbox=%s yolo_mask=%s yolo_kpt=%s yolo_kpt_s=%s d2_bbox=%s d2_kpt=%s d2_kpt_s=%s",
        tuple(depth.shape),
        tuple(flow.shape),
        tuple(y_bbox.shape),
        tuple(y_mask.shape),
        tuple(y_kpt.shape),
        tuple(y_kpt_s.shape),
        tuple(d_bbox.shape),
        tuple(d_kpt.shape),
        tuple(d_kpt_s.shape),
    )
    logger.info("Total frames processed: %d (none_index: %d)", total_T, len(none_idx))

    if depth.shape[0] != raw_vframes.shape[0] and depth.numel() > 0:
        raise ValueError(
            f"Depth frame count mismatch {depth.shape[0]} vs {raw_vframes.shape[0]}"
        )
    if flow.shape[0] != raw_vframes.shape[0] - 1 and flow.numel() > 0:
        raise ValueError(
            f"Flow frame count mismatch {flow.shape[0]} vs {raw_vframes.shape[0]-1}"
        )
    if y_bbox.shape[0] != raw_vframes.shape[0] and y_bbox.numel() > 0:
        raise ValueError(
            f"YOLO bbox frame count mismatch {y_bbox.shape[0]} vs {raw_vframes.shape[0]}"
        )
    if y_mask.shape[0] != raw_vframes.shape[0] and y_mask.numel() > 0:
        raise ValueError(
            f"YOLO mask frame count mismatch {y_mask.shape[0]} vs {raw_vframes.shape[0]}"
        )
    if y_kpt.shape[0] != raw_vframes.shape[0] and y_kpt.numel() > 0:
        raise ValueError(
            f"YOLO kpt frame count mismatch {y_kpt.shape[0]} vs {raw_vframes.shape[0]}"
        )
    if y_kpt_s.shape[0] != raw_vframes.shape[0] and y_kpt_s.numel() > 0:
        raise ValueError(
            f"YOLO kpt_score frame count mismatch {y_kpt_s.shape[0]} vs {raw_vframes.shape[0]}"
        )
    if d_bbox.shape[0] != raw_vframes.shape[0] and d_bbox.numel() > 0:
        raise ValueError(
            f"D2 bbox frame count mismatch {d_bbox.shape[0]} vs {raw_vframes.shape[0]}"
        )
    if d_kpt.shape[0] != raw_vframes.shape[0] and d_kpt.numel() > 0:
        raise ValueError(
            f"D2 kpt frame count mismatch {d_kpt.shape[0]} vs { raw_vframes.shape[0]}"
        )
    if d_kpt_s.shape[0] != raw_vframes.shape[0] and d_kpt_s.numel() > 0:
        raise ValueError(
            f"D2 kpt_score frame count mismatch {d_kpt_s.shape[0]} vs {raw_vframes.shape[0]}"
        )

    pt_info: Dict[str, Any] = {
        "optical_flow": flow.contiguous(),
        "depth": depth.contiguous(),
        "none_index": none_idx,
        "YOLO": {
            "bbox": y_bbox.contiguous(),
            "mask": y_mask.contiguous(),
            "keypoints": y_kpt.contiguous(),
            "keypoints_score": y_kpt_s.contiguous(),
        },
        "detectron2": {
            "bbox": d_bbox.contiguous(),
            "keypoints": d_kpt.contiguous(),
            "keypoints_score": d_kpt_s.contiguous(),
        },
        "video_name": video_path.stem,
        "video_path": str(video_path),
        "frame_count": int(total_T),
        "img_shape": (
            int(H) if H is not None else None,
            int(W) if W is not None else None,
        ),
        "fps": float(fps),
        "duration": float(duration),
        "rotate_deg": rotate_deg,
    }
    return pt_info


# --------------------------- per person & entry ---------------------------


def process_one_person(config, person: str) -> None:
    raw_root = Path(config.extract_dataset.data_path)
    save_root = Path(config.extract_dataset.save_path)

    person_dir = raw_root / person
    if not person_dir.exists():
        logger.warning("Person dir not found: %s", person_dir)
        return

    overwrite: bool = bool(getattr(config.extract_dataset, "overwrite", False))
    recursive: bool = bool(getattr(config.extract_dataset, "recursive", True))
    legacy_zip: bool = bool(getattr(config.extract_dataset, "legacy_zip", True))
    chunk_size: int = int(getattr(config, "chunk_size", 128))
    num_threads: int = int(getattr(config, "num_threads", 0))
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        logger.info("torch.set_num_threads(%d)", num_threads)

    logger.info(
        "Start person=%s (recursive=%s overwrite=%s chunk=%d)",
        person,
        recursive,
        overwrite,
        chunk_size,
    )

    for video_path in _iter_videos(person_dir, recursive=recursive):
        out_pt = save_root / "pt" / person / f"{video_path.stem}.pt"
        if out_pt.exists() and not overwrite:
            logger.info("Skip existed: %s", out_pt)
            continue
        try:
            pt_info = process_video_chunked(
                config, person, video_path, chunk_size=chunk_size
            )
            _safe_save_pt(out_pt, pt_info, legacy_zip=legacy_zip)
        except Exception as e:
            logger.exception("Failed on %s: %s", video_path, e)
        finally:
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
