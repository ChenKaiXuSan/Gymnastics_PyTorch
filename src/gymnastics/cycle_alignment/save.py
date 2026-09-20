#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal


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
