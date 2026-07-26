#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal, Dict


def _open_video(video_path: Union[str, Path]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    return cap


def _non_overlapping_windows(
    cycles: List[Tuple[int, int]],
    n_frames: int,
    pad: int,
) -> List[Tuple[int, int]]:
    """
    cycles are [start,end) (end exclusive).
    Apply pad then clip so that windows are non-overlapping and monotonic.
    """
    if not cycles:
        return []

    padded = []
    for s, e in cycles:
        s2 = max(0, int(s) - pad)
        e2 = min(n_frames, int(e) + pad)
        if e2 > s2:
            padded.append((s2, e2))

    if not padded:
        return []

    out = [padded[0]]
    for i in range(1, len(padded)):
        ps, pe = out[-1]
        cs, ce = padded[i]
        if cs < pe:
            cut = (pe + cs) // 2
            out[-1] = (ps, cut)
            cs = cut
        out.append((cs, ce))

    # ensure valid
    out2 = []
    last_end = 0
    for s, e in out:
        s = max(s, last_end)
        e = max(e, s + 1)
        out2.append((s, e))
        last_end = e
    return out2


def save_cycles_videos(
    video_path: Union[str, Path],
    cycles: List[Tuple[int, int]],
    out_dir: Union[str, Path],
    *,
    fps_out: Optional[float] = None,               # None -> use source fps
    codec: str = "mp4v",
    prefix: str = "cycle",
    pad: int = 0,
    avoid_overlap: bool = True,                    # ⭐ pad后是否自动裁剪避免重叠
    resize_wh: Optional[Tuple[int, int]] = None,   # (w,h) after optional rotation
    rotate: Optional[Literal["cw90", "ccw90", "180"]] = "cw90",
    verbose: bool = True,
) -> List[Path]:
    """
    Save each cycle to a video by frame indices [start, end) (end exclusive).

    Notes:
      - Much faster than per-frame cap.set: seek once per cycle then sequential read.
      - If rotate is 90deg, writer frame size will be swapped automatically.
      - If pad>0 and avoid_overlap=True, padded windows will be clipped to avoid overlaps.

    Returns:
      list of saved video paths
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_video(video_path)

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = src_fps if (fps_out is None) else float(fps_out)
    if fps <= 0:
        fps = 30.0  # fallback

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build windows (apply pad + optional non-overlap)
    windows = cycles
    if pad > 0:
        windows = _non_overlapping_windows(cycles, n_frames=n_frames, pad=pad) if avoid_overlap else [
            (max(0, int(s) - pad), min(n_frames, int(e) + pad)) for s, e in cycles
        ]

    # Determine output size AFTER rotation but BEFORE resize
    if rotate in ("cw90", "ccw90"):
        base_w, base_h = src_h, src_w  # swapped
    else:
        base_w, base_h = src_w, src_h

    if resize_wh is not None:
        out_w, out_h = resize_wh
    else:
        out_w, out_h = base_w, base_h

    fourcc = cv2.VideoWriter_fourcc(*codec)
    saved_paths: List[Path] = []

    def _apply_rotate(frame):
        if rotate is None:
            return frame
        if rotate == "cw90":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotate == "ccw90":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotate == "180":
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    for ci, (s, e) in enumerate(windows):
        s = int(s); e = int(e)
        s = max(0, min(s, n_frames))
        e = max(0, min(e, n_frames))
        if e <= s + 1:
            continue

        out_path = out_dir / f"{prefix}_{ci:03d}_{s:06d}_{e:06d}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

        if verbose:
            print(f"[{ci:03d}] save {out_path.name} frames: [{s},{e}) fps={fps:.2f}")

        # Seek once to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)

        # Sequential read: faster
        for t in range(s, e):
            ok, frame = cap.read()
            if not ok or frame is None:
                # broken frame / EOF
                break

            frame = _apply_rotate(frame)

            if resize_wh is not None:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                # safety: if rotation swapped size, ensure matches writer
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(frame)

        writer.release()
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def save_fused_kpts(
    fused_world: np.ndarray,
    fused_body: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
    person_id: str,
    out_root: Path,
    fps: float,
) -> Dict[str, Path]:
    """
    保存融合后的3D关键点数据，每一帧保存为单独的npz文件
    
    Args:
        fused_world: (T, J, 3) 世界坐标系下的融合关键点
        fused_body: (T, J, 3) 身体坐标系下的融合关键点  
        face_map: (T,) 到原始face视频的帧映射
        side_map: (T,) 到原始side视频的帧映射
        person_id: 人物ID
        out_root: 输出根目录
        fps: 帧率
    
    Returns:
        保存的文件路径字典
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 创建frames子目录
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    T = len(fused_world)
    
    # 逐帧保存为npz文件
    print(f"  [save] Saving {T} frames as individual npz files...")
    for t in range(T):
        frame_path = frames_dir / f"frame_{t:06d}.npz"
        np.savez_compressed(
            frame_path,
            kpts_world=fused_world[t],  # (J, 3)
            kpts_body=fused_body[t],    # (J, 3)
            face_frame_idx=face_map[t],  # int
            side_frame_idx=side_map[t],  # int
            frame_idx=t,                 # int
        )
    
    saved_files['frames_dir'] = frames_dir
    
    # 保存元数据
    metadata = {
        "person_id": person_id,
        "fps": float(fps),
        "n_frames": int(T),
        "n_joints": int(fused_world.shape[1]),
        "frames_dir": str(frames_dir.relative_to(out_root)),
        "frame_format": "frame_{frame_idx:06d}.npz",
        "face_map": face_map.tolist(),
        "side_map": side_map.tolist(),
        "description": {
            "frames_dir": "Directory containing per-frame npz files",
            "npz_contents": {
                "kpts_world": "(J, 3) Fused 3D keypoints in world coordinates",
                "kpts_body": "(J, 3) Fused 3D keypoints in body-local coordinates",
                "face_frame_idx": "Original face video frame index (-1 if not available)",
                "side_frame_idx": "Original side video frame index (-1 if not available)",
                "frame_idx": "Fused timeline frame index",
            },
            "face_map": "Mapping from fused timeline to original face video frames",
            "side_map": "Mapping from fused timeline to original side video frames",
        }
    }
    
    metadata_path = out_root / f"fused_kpts_metadata_{person_id}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    saved_files['metadata'] = metadata_path
    
    return saved_files
