"""Cycle segmentation on the fused body-frame trajectory and the mapping of
common-timeline cycles back to per-video frame indices."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # matplotlib is optional for visualization
    plt = None

from cycle_alignment.cycles import CycleSpan, DetectionSettings, detect_cycles
from cycle_alignment.features import smooth_1d


# -------------------- fusion in BODY coords --------------------
def fuse_body_kpts(
    face_body: np.ndarray,  # (T,J,3)
    side_body: np.ndarray,  # (T,J,3)
    face_w: np.ndarray,  # (T,J) weight (0/1 ok)
    side_w: np.ndarray,  # (T,J)
) -> np.ndarray:
    face_w = face_w.astype(np.float32)
    side_w = side_w.astype(np.float32)
    wsum = face_w + side_w + 1e-8
    fused = (face_body * face_w[..., None] + side_body * side_w[..., None]) / wsum[
        ..., None
    ]
    return fused.astype(np.float32)


# -------------------- cycle segmentation --------------------
def segment_cycles_from_fused_body(
    fused_body: np.ndarray,
    fps: float,
    *,
    wrist_idx: int = 41,
    theta_ref: float = -3 * np.pi / 4,
    min_period_sec: float = 0.8,
    both_directions: bool = False,
    auto_theta_ref: bool = False,
    verbose: bool = False,
) -> Tuple[List[CycleSpan], DetectionSettings]:
    """Detect cycles (with turn-around middles) on the fused right-wrist angle.

    The detection itself lives in :mod:`cycle_alignment.cycles`; this
    wrapper only builds the signal from the fused body-frame keypoints.

    Returns:
        cycles: ``CycleSpan`` objects (start, mid, end) on the fused timeline.
        settings: The detection settings actually used (theta_ref, direction).
    """
    hand_b = fused_body[:, wrist_idx, :]
    x, z = hand_b[:, 0], hand_b[:, 2]
    theta = np.arctan2(z, x).astype(np.float32)
    theta = smooth_1d(theta, 11)
    theta_u = np.unwrap(theta)
    settings = DetectionSettings(
        theta_ref=float(theta_ref),
        theta_ref_mode="auto_p10" if auto_theta_ref else "manual",
        min_period_sec=float(min_period_sec),
    )
    spans, used = detect_cycles(theta_u, fps, settings=settings, both_directions=both_directions)
    if verbose:
        print(f"    θ range: [{theta_u.min():.2f}, {theta_u.max():.2f}], std: {np.std(theta_u):.4f}")
        print(f"    θ_ref ({used.theta_ref_mode}): {used.theta_ref:.3f}, direction: {used.direction}, cycles: {len(spans)}")
        if spans:
            print(f"    gaps(sec): {np.diff([s.start for s in spans] + [spans[-1].end]) / fps}")
    return spans, used


def save_theta_plot(
    fused_body: np.ndarray,
    fps: float,
    out_path: Path,
    *,
    wrist_idx: int = 41,
    theta_ref: float = -3 * np.pi / 4,
    crossing_points: List[int] = None,
    auto_detected: bool = False,
) -> bool:
    if plt is None:
        print("⚠ [warn] matplotlib not available, skip theta plot")
        return False

    hand_b = fused_body[:, wrist_idx, :]
    x, z = hand_b[:, 0], hand_b[:, 2]
    theta = np.arctan2(z, x).astype(np.float32)
    theta = smooth_1d(theta, 11)
    theta_u = np.unwrap(theta)

    t = np.arange(len(theta_u), dtype=np.float32) / max(float(fps), 1e-6)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    ax.plot(t, theta_u, linewidth=1.2, label="θ (unwrapped)")
    
    # 参考线
    ref_label = f"θ_ref = {theta_ref:.3f}" + (" (auto)" if auto_detected else "")
    ax.axhline(theta_ref, color="r", linestyle="--", linewidth=1.0, alpha=0.7, label=ref_label)
    
    # 标记检测到的crossing点
    if crossing_points:
        crossing_t = np.array(crossing_points) / max(float(fps), 1e-6)
        crossing_theta = theta_u[crossing_points]
        ax.scatter(crossing_t, crossing_theta, color="green", s=50, zorder=5, 
                   marker="o", label=f"Crossings ({len(crossing_points)})")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Theta (rad)")
    ax.set_title("Right hand theta (unwrapped)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


# -------------------- mapping cycles on common timeline -> original video cycles --------------------
def cycles_t_to_video_cycles(
    cycles_t: List[Tuple[int, int]],
    frame_map: np.ndarray,  # (T,) -1 or original frame idx
) -> List[Tuple[int, int]]:
    """
    cycles_t are [ts,te) on the CROPPED common timeline.
    Convert to original video frame cycles [start,end).
    """
    out: List[Tuple[int, int]] = []
    for ts, te in cycles_t:
        seg = frame_map[int(ts) : int(te)]
        valid = seg[seg >= 0]
        if len(valid) < 2:
            continue
        s = int(valid[0])
        e = int(valid[-1]) + 1
        if e > s + 1:
            out.append((s, e))
    return out


def spans_t_to_video_spans(
    spans_t: List[CycleSpan],
    frame_map: np.ndarray,  # (T,) -1 or original frame idx
    n_frames: int,
) -> List[Optional[CycleSpan]]:
    """Map fused-timeline spans to original video frames, keeping the middle.

    Returns one entry per input span; ``None`` marks spans that do not map to
    at least three valid frames (they are dropped by the caller together
    with their partner view so both lists stay aligned).
    """
    out: List[Optional[CycleSpan]] = []
    for span in spans_t:
        seg = frame_map[span.start : span.end]
        valid = seg[seg >= 0]
        mid_seg = frame_map[span.mid : span.end]
        mid_valid = mid_seg[mid_seg >= 0]
        if len(valid) < 3 or len(mid_valid) == 0:
            out.append(None)
            continue
        s0 = max(0, min(int(valid[0]), n_frames))
        e0 = max(0, min(int(valid[-1]) + 1, n_frames))
        m0 = max(s0 + 1, min(int(mid_valid[0]), e0 - 1))
        out.append(CycleSpan(s0, m0, e0) if e0 > s0 + 2 else None)
    return out


def clamp_cycles(cycles: List[Tuple[int, int]], n_frames: int) -> List[Tuple[int, int]]:
    out = []
    for s, e in cycles:
        s = max(0, min(int(s), n_frames))
        e = max(0, min(int(e), n_frames))
        if e > s + 1:
            out.append((s, e))
    return out


def get_video_nframes(video_path: Union[str, Path]) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

