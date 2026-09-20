"""Side-to-face temporal offset estimation.

Keypoint DTW on the right-hand angle, audio-envelope cross-correlation and the
rule that chooses between them, plus the helpers that place both views on a
common timeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import librosa
except Exception:
    librosa = None


def estimate_offset_by_dtw(a: np.ndarray, b: np.ndarray):
    """
    使用 DTW 估算信号 b 相对于 a 的偏移量
    """
    # 1. 预处理：标准化（DTW 对数值量级敏感）
    a_norm = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
    b_norm = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-8)

    if librosa is None:
        offset, _ = estimate_offset_from_audio_envelopes(
            a_norm, b_norm, hop_seconds=1.0, fps=1.0
        )
        return offset

    # 2. 计算 DTW 路径。这里不用 librosa，避免运行环境里的 numba cache 问题
    # 影响 cycle 切分主流程。
    wp = dtw_warp_path_1d(a_norm, b_norm)

    # 3. 从路径中估算偏移
    # 路径中的每一对 [i, j] 代表 a[i] 和 b[j] 是匹配的
    # 偏移 s = j - i
    offsets = wp[:, 1] - wp[:, 0]

    # 取中位数或平均数作为整体偏移量
    best_s = int(np.median(offsets))

    return best_s


def dtw_warp_path_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return DTW warp path pairs [index_a, index_b] for two 1D sequences.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("DTW input sequences must be non-empty")

    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            acc[i, j] = cost + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = (
            acc[i - 1, j - 1],
            acc[i - 1, j],
            acc[i, j - 1],
        )
        step = int(np.argmin(candidates))
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    while i > 0:
        path.append((i - 1, 0))
        i -= 1
    while j > 0:
        path.append((0, j - 1))
        j -= 1

    path.reverse()
    return np.asarray(path, dtype=np.int32)


def _normalize_audio_envelope(env: np.ndarray) -> np.ndarray:
    env = np.asarray(env, dtype=np.float32).reshape(-1)
    if len(env) == 0:
        return env
    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
    env = env - np.mean(env)
    std = np.std(env)
    if std < 1e-8:
        return np.zeros_like(env, dtype=np.float32)
    return (env / std).astype(np.float32)


def estimate_offset_from_audio_envelopes(
    face_env: np.ndarray,
    side_env: np.ndarray,
    *,
    hop_seconds: float,
    fps: float,
) -> Tuple[int, float]:
    """
    Estimate side-to-face frame offset from two audio envelopes.

    Positive offset keeps the same convention as keypoint DTW:
      side_idx[t] = t + offset
    """
    face_norm = _normalize_audio_envelope(face_env)
    side_norm = _normalize_audio_envelope(side_env)
    if len(face_norm) == 0 or len(side_norm) == 0:
        raise ValueError("audio envelope is empty")

    denom = float(np.linalg.norm(face_norm) * np.linalg.norm(side_norm))
    if denom < 1e-8:
        raise ValueError("audio envelope has too little energy variation")

    corr = np.correlate(side_norm, face_norm, mode="full") / denom
    lags = np.arange(-(len(face_norm) - 1), len(side_norm), dtype=np.int32)
    best = int(np.argmax(corr))
    lag_env = int(lags[best])
    confidence = float(np.clip(corr[best], 0.0, 1.0))
    offset_frames = int(round(lag_env * float(hop_seconds) * float(fps)))
    return offset_frames, confidence


def extract_audio_envelope(
    video_path: Union[str, Path],
    *,
    sr: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Tuple[np.ndarray, float]:
    """
    Load video audio and return a short-time RMS energy envelope.

    librosa/audioread will use the system media backend for MOV audio.
    """
    if librosa is None:
        raise ImportError("librosa is not installed; audio alignment is unavailable")
    y, _ = librosa.load(str(video_path), sr=sr, mono=True)
    if y is None or len(y) == 0:
        raise ValueError(f"no audio samples loaded from {video_path}")
    env = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    return env.astype(np.float32), float(hop_length) / float(sr)


def estimate_offset_by_audio_xcorr(
    face_video: Union[str, Path],
    side_video: Union[str, Path],
    *,
    fps: float,
    sr: int = 16000,
) -> Tuple[int, float]:
    face_env, hop_seconds = extract_audio_envelope(face_video, sr=sr)
    side_env, side_hop_seconds = extract_audio_envelope(side_video, sr=sr)
    if abs(hop_seconds - side_hop_seconds) > 1e-9:
        raise ValueError("face/side audio envelopes use different hop sizes")
    return estimate_offset_from_audio_envelopes(
        face_env, side_env, hop_seconds=hop_seconds, fps=fps
    )


def choose_alignment_offset(
    *,
    offset_kpt: int,
    offset_audio: Optional[int],
    audio_confidence: float = 0.0,
    tolerance_frames: int = 10,
    min_audio_confidence: float = 0.15,
) -> Tuple[int, str]:
    if offset_audio is None or audio_confidence < min_audio_confidence:
        return int(offset_kpt), "kpt"
    if abs(int(offset_audio) - int(offset_kpt)) <= int(tolerance_frames):
        return int(round((int(offset_kpt) + int(offset_audio)) / 2.0)), "kpt_audio_avg"
    return int(offset_kpt), "kpt"


# -------------------- 先对齐（union）再裁剪（overlap） --------------------
def align_to_common_timeline(
    face: np.ndarray,  # (Tf, ...)
    side: np.ndarray,  # (Ts, ...)
    offset_side_to_face: int,
    *,
    pad_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a common UNION timeline.
      face_idx[t] = original face frame index, -1 if missing
      side_idx[t] = original side frame index, -1 if missing
    """
    s = int(offset_side_to_face)
    Tf = len(face)
    Ts = len(side)

    t_min = min(0, -s)
    t_max = max(Tf, Ts - s)
    L = max(0, t_max - t_min)

    t = np.arange(t_min, t_max, dtype=np.int32)  # length L
    face_idx = t.copy()
    side_idx = t + s

    face_valid = (face_idx >= 0) & (face_idx < Tf)
    side_valid = (side_idx >= 0) & (side_idx < Ts)

    face_map = np.where(face_valid, face_idx, -1).astype(np.int32)
    side_map = np.where(side_valid, side_idx, -1).astype(np.int32)

    # allocate
    out_shape_f = (L,) + face.shape[1:]
    out_shape_s = (L,) + side.shape[1:]

    # if pad_value is NaN, store as float32
    if np.isnan(pad_value):
        face_src = face.astype(np.float32, copy=False)
        side_src = side.astype(np.float32, copy=False)
        out_dtype = np.float32
    else:
        face_src = face
        side_src = side
        out_dtype = face.dtype

    face_aligned = np.full(out_shape_f, pad_value, dtype=out_dtype)
    side_aligned = np.full(out_shape_s, pad_value, dtype=out_dtype)

    face_aligned[face_valid] = face_src[face_idx[face_valid]]
    side_aligned[side_valid] = side_src[side_idx[side_valid]]

    return face_aligned, side_aligned, face_map, side_map


def crop_to_overlap(
    face_aligned: np.ndarray,
    side_aligned: np.ndarray,
    face_map: np.ndarray,
    side_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Crop to the maximal contiguous range where both views exist.
    Returns cropped arrays + cropped maps + (t0,t1) crop indices on the union timeline.
    """
    valid = (face_map >= 0) & (side_map >= 0)
    if not np.any(valid):
        return face_aligned[:0], side_aligned[:0], face_map[:0], side_map[:0], 0, 0

    t0 = int(np.argmax(valid))
    t1 = int(len(valid) - np.argmax(valid[::-1]))  # end exclusive

    return (
        face_aligned[t0:t1],
        side_aligned[t0:t1],
        face_map[t0:t1],
        side_map[t0:t1],
        t0,
        t1,
    )

