#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/split_cycle/main.py
Project: /workspace/code/split_cycle
Created Date: Friday January 30th 2026
Author: Kaixu Chen
-----
Comment:
读取多个人的视频

処理フローの概要：マルチ視点データの同期と周期分割

本プログラムは、2つの視点（正面・側面）の動画から得られた3Dスケルトンデータを統合し、動作の周期（サイクル）ごとに動画を切り出すものです。

1. データ準備と特徴量抽出

3Dキーポイントの読み込み: 各視点のデータをロードします。

角度特徴の計算: 世界座標を身体局所座標系（骨盤中心）に変換し、右手の回転角度 $\theta$ を算出します。unwrap 処理で角度の不連続性を解消し、同期用の特徴量とします。

2. 時間軸の同期 (DTW)

オフセット推定: DTW（動的時間伸長法） を用いて、正面と側面の角度データのズレ（何フレーム分か）を特定します。

共通タイムラインの構築: 両方の視点が存在する**重複区間（Overlap）**のみを抽出します。

3. データ融合 (Data Fusion)

座標変換と統合: 両視点のデータを身体座標系で統合します。

重み付き平均: 両方のカメラで見えている点は平均化することで、遮蔽（オクルージョン）による誤差やノイズを低減し、精度の高い骨格データを作成します。

4. 周期分割 (Cycle Segmentation)

ゼロ交差判定: 統合された右手の軌跡から、特定の基準角を通過するタイミング（例：手が最下点に来る瞬間）を検出します。

サイクル定義: 連続する通過点を1つの周期として切り出します。

5. マッピングと動画保存

フレーム再マッピング: 共通タイムライン上で見つけた周期を、元の各動画のフレーム番号に逆算して戻します。

動画切り出し: 各周期を個別の動画ファイルとして書き出します。

Have a good code time :)
-----
Last Modified: Friday January 30th 2026 1:24:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import cv2
import librosa
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # matplotlib is optional for visualization
    plt = None

from split_cycle.load import load_sam3d_body_sequence
from split_cycle.save import save_cycles_videos

# -------------------- MHR70 indices --------------------
IDX: Dict[str, int] = {
    "lhip": 9,
    "rhip": 10,
    "lsho": 5,
    "rsho": 6,
    "neck": 69,
    "rwrist": 41,
    "rindex_tip": 25,
    "rmiddle_tip": 29,
    "rpinky_tip": 37,
}


# -------------------- utils --------------------
def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def smooth_1d(x: np.ndarray, win: int = 11) -> np.ndarray:
    win = int(win)
    win = max(3, win | 1)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")


# -------------------- body frame (world -> body) --------------------
def build_body_frame_from_mhr70(
    kpts: np.ndarray, idx: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    kpts: (T,J,3)
    return pelvis_world: (T,3)
           R_body_to_world: (T,3,3) columns [x(right), y(up), z(forward)] in world
    """
    lhip = kpts[:, idx["lhip"], :]
    rhip = kpts[:, idx["rhip"], :]
    pelvis = 0.5 * (lhip + rhip)

    x_axis = _normalize(rhip - lhip)

    lsho = kpts[:, idx["lsho"], :]
    rsho = kpts[:, idx["rsho"], :]
    shoulder_center = 0.5 * (lsho + rsho)
    y_axis = _normalize(shoulder_center - pelvis)

    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return pelvis, R


def world_to_body(
    points_world: np.ndarray, pelvis_world: np.ndarray, R_body_to_world: np.ndarray
) -> np.ndarray:
    v = points_world - pelvis_world
    return np.einsum("tij,tj->ti", np.transpose(R_body_to_world, (0, 2, 1)), v)


def kpts_world_to_body(kpts_world: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
    pelvis, R = build_body_frame_from_mhr70(kpts_world, idx)
    v = kpts_world - pelvis[:, None, :]
    kpts_body = np.einsum("tij,tbj->tbi", np.transpose(R, (0, 2, 1)), v)
    return kpts_body.astype(np.float32)


def right_hand_point_world(
    kpts_world: np.ndarray,
    idx: Dict[str, int],
    mode: Literal["wrist", "hand_center"] = "hand_center",
) -> np.ndarray:
    if mode == "wrist":
        return kpts_world[:, idx["rwrist"], :]
    wrist = kpts_world[:, idx["rwrist"], :]
    index_tip = kpts_world[:, idx["rindex_tip"], :]
    middle_tip = kpts_world[:, idx["rmiddle_tip"], :]
    pinky_tip = kpts_world[:, idx["rpinky_tip"], :]
    return 0.25 * (wrist + index_tip + middle_tip + pinky_tip)


def compute_theta_unwrap_from_world(
    kpts_world: np.ndarray, idx: Dict[str, int]
) -> np.ndarray:
    """
    Compute theta(t) on BODY x-z plane (right hand), then unwrap.
    """
    pelvis, R = build_body_frame_from_mhr70(kpts_world, idx)
    hand_w = right_hand_point_world(kpts_world, idx, mode="hand_center")
    hand_b = world_to_body(hand_w, pelvis, R)
    x, z = hand_b[:, 0], hand_b[:, 2]
    theta = np.arctan2(z, x).astype(np.float32)
    theta = smooth_1d(theta, 11)
    return np.unwrap(theta)


def estimate_offset_by_dtw(a: np.ndarray, b: np.ndarray):
    """
    使用 DTW 估算信号 b 相对于 a 的偏移量
    """
    # 1. 预处理：标准化（DTW 对数值量级敏感）
    a_norm = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
    b_norm = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-8)

    # 2. 计算 DTW 路径
    # D: 累计距离矩阵, wp: 匹配路径 (warp path)
    # wp 是一个形状为 (N, 2) 的数组，每一行是 [index_a, index_b]
    D, wp = librosa.sequence.dtw(a_norm, b_norm, backtrack=True)

    # 3. 从路径中估算偏移
    # 路径中的每一对 [i, j] 代表 a[i] 和 b[j] 是匹配的
    # 偏移 s = j - i
    offsets = wp[:, 1] - wp[:, 0]

    # 取中位数或平均数作为整体偏移量
    best_s = int(np.median(offsets))

    return best_s


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
def find_crossings(
    theta_unwrap: np.ndarray,
    fps: float,
    *,
    theta_ref: float = -3 * np.pi / 4,
    min_period_sec: float = 0.8,
    direction: Literal["ccw", "cw"] = "ccw",
    verbose: bool = False,
) -> List[int]:
    min_gap = int(round(min_period_sec * fps))

    k = np.round((theta_unwrap - theta_ref) / (2 * np.pi))
    ref = theta_ref + 2 * np.pi * k
    d = theta_unwrap - ref

    sgn = np.sign(d)
    sgn[sgn == 0] = 1

    vel = np.gradient(theta_unwrap)
    ok = vel > 0 if direction == "ccw" else vel < 0

    crossing: List[int] = []
    for t in range(1, len(theta_unwrap)):
        if ok[t] and (sgn[t - 1] < 0 and sgn[t] > 0):
            crossing.append(t)

    out: List[int] = []
    last = -(10**9)
    for t in crossing:
        if t - last >= min_gap:
            out.append(t)
            last = t
    
    if verbose:
        print(f"    [{direction}] raw crossings: {len(crossing)}, filtered: {len(out)}", end="")
        if len(out) > 0:
            gaps = np.diff(out)
            print(f", gaps(sec): {gaps/fps}")
        else:
            print()
    
    return out


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
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Returns:
        cycles: List of (start, end) tuples
        theta_ref_used: The actual theta_ref used (auto-detected or input)
    """
    hand_b = fused_body[:, wrist_idx, :]
    x, z = hand_b[:, 0], hand_b[:, 2]
    theta = np.arctan2(z, x).astype(np.float32)
    theta = smooth_1d(theta, 11)
    theta_u = np.unwrap(theta)

    # 自动检测 theta_ref
    if auto_theta_ref:
        # 使用第10百分位作为参考点（接近波谷）
        theta_ref = float(np.percentile(theta_u, 10))
        if verbose:
            print(f"    [auto] theta_ref detected: {theta_ref:.3f}")

    if verbose:
        print(f"    θ range: [{theta_u.min():.2f}, {theta_u.max():.2f}], std: {np.std(theta_u):.4f}")
        if not auto_theta_ref:
            print(f"    θ_ref (manual): {theta_ref:.3f}")

    # 首先尝试逆时针方向
    bds_ccw = find_crossings(
        theta_u, fps=fps, theta_ref=theta_ref, min_period_sec=min_period_sec, 
        direction="ccw", verbose=verbose
    )
    
    if both_directions and len(bds_ccw) < 2:
        # 如果逆时针找不到足够的cycle，尝试顺时针
        bds_cw = find_crossings(
            theta_u, fps=fps, theta_ref=theta_ref, min_period_sec=min_period_sec, 
            direction="cw", verbose=verbose
        )
        if len(bds_cw) >= 2:
            bds = bds_cw
        else:
            bds = bds_ccw
    else:
        bds = bds_ccw
    
    cycles = [(bds[i], bds[i + 1]) for i in range(len(bds) - 1)]
    return cycles, theta_ref


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


# -------------------- process single person --------------------
def process_person(person_id: str, raw_root: Path, kpt_root: Path, log_root: Path):
    """
    处理单个人物的对齐、融合和周期分割
    """
    try:
        person_log_root = log_root / f"person_{person_id}"

        face_video = raw_root / f"person/{person_id}/ID{person_id}_face.MOV"
        side_video = raw_root / f"person/{person_id}/ID{person_id}_side.MOV"

        # 检查数据是否存在
        if not face_video.exists() or not side_video.exists():
            print(f"⚠ [skip] person_{person_id}: video files not found")
            return False

        # 1) load kpts (world coords)
        try:
            face_seq = load_sam3d_body_sequence(kpt_root, person_id=person_id, subdir="face")
            side_seq = load_sam3d_body_sequence(kpt_root, person_id=person_id, subdir="side")
        except Exception as e:
            print(f"⚠ [skip] person_{person_id}: failed to load kpts - {e}")
            return False

        face_k = face_seq.kpts3d if hasattr(face_seq, "kpts3d") else face_seq[1]
        side_k = side_seq.kpts3d if hasattr(side_seq, "kpts3d") else side_seq[1]

        print(f"✓ [load] person_{person_id}: face {face_k.shape}, side {side_k.shape}")

        # 2) estimate offset using theta in body frame
        theta_face = compute_theta_unwrap_from_world(face_k, IDX)
        theta_side = compute_theta_unwrap_from_world(side_k, IDX)

        offset = estimate_offset_by_dtw(theta_face, theta_side)
        print(f"✓ [align] person_{person_id}: offset_side_to_face = {offset}")

        # 2-1) 先对齐到同一时间轴（union）
        face_u, side_u, face_map_u, side_map_u = align_to_common_timeline(
            face_k, side_k, offset, pad_value=np.nan
        )
        print(
            f"✓ [union] person_{person_id}: length {len(face_u)}, "
            f"face_exist {np.sum(face_map_u >= 0)}, side_exist {np.sum(side_map_u >= 0)}"
        )

        # 2-2) 再裁剪：只保留两路都存在的最大连续区间（overlap）
        face_k2, side_k2, face_map, side_map, t0, t1 = crop_to_overlap(
            face_u, side_u, face_map_u, side_map_u
        )
        if len(face_k2) == 0:
            print(f"⚠ [skip] person_{person_id}: no overlap segment found")
            return False

        # 检查overlap质量
        overlap_ratio = len(face_k2) / max(len(face_k), len(side_k))
        print(f"✓ [crop] person_{person_id}: overlap [{t0},{t1}) length={len(face_k2)}, ratio={overlap_ratio:.2%}")
        if overlap_ratio < 0.3:
            print(f"⚠ [warn] person_{person_id}: low overlap ratio, may have quality issues")

        # 3) fuse in BODY coords
        face_body = kpts_world_to_body(face_k2, IDX)
        side_body = kpts_world_to_body(side_k2, IDX)

        # weights: 1 when this frame exists in that view (map>=0)
        wf = (face_map >= 0).astype(np.float32)[:, None]
        ws = (side_map >= 0).astype(np.float32)[:, None]
        # expand to (T,J)
        T, J, _ = face_body.shape
        wf = np.repeat(wf, J, axis=1)
        ws = np.repeat(ws, J, axis=1)

        fused_body = fuse_body_kpts(face_body, side_body, wf, ws)
        print(f"✓ [fuse] person_{person_id}: fused shape {fused_body.shape}")

        # 4) segment cycles on fused
        fps_kpt = 60.0
        print(f"  [cycle] person_{person_id}: segmenting cycles...")
        
        cycles_t, theta_ref_used = segment_cycles_from_fused_body(
            fused_body, fps=fps_kpt, wrist_idx=IDX["rwrist"],
            min_period_sec=0.8, both_directions=True, auto_theta_ref=True, verbose=True
        )
        print(f"✓ [cycle] person_{person_id}: found {len(cycles_t)} cycles")
        if len(cycles_t) == 0:
            print(f"⚠ [warn] person_{person_id}: no cycles found, check data quality")
        
        # 生成可视化（带检测点标记）
        theta_plot_path = person_log_root / "theta_unwrap.png"
        # 提取crossing点用于可视化
        crossing_pts = [cycles_t[i][0] for i in range(len(cycles_t))]
        if len(cycles_t) > 0:
            crossing_pts.append(cycles_t[-1][1])  # 添加最后一个结束点
        save_theta_plot(
            fused_body,
            fps=fps_kpt,
            out_path=theta_plot_path,
            wrist_idx=IDX["rwrist"],
            theta_ref=theta_ref_used,
            crossing_points=crossing_pts if crossing_pts else None,
            auto_detected=True,
        )

        # 5) save videos (face/side) using mapping
        face_cycles = cycles_t_to_video_cycles(cycles_t, frame_map=face_map)
        side_cycles = cycles_t_to_video_cycles(cycles_t, frame_map=side_map)

        # clamp by actual video length
        nF = get_video_nframes(face_video)
        nS = get_video_nframes(side_video)
        face_cycles = clamp_cycles(face_cycles, nF)
        side_cycles = clamp_cycles(side_cycles, nS)

        out_face = person_log_root / "face"
        out_side = person_log_root / "side"
        out_face.mkdir(parents=True, exist_ok=True)
        out_side.mkdir(parents=True, exist_ok=True)

        pad = 0
        save_cycles_videos(
            face_video, face_cycles, out_face, pad=pad, avoid_overlap=True, prefix="cycle"
        )
        save_cycles_videos(
            side_video, side_cycles, out_side, pad=pad, avoid_overlap=True, prefix="cycle"
        )
        print(f"✓ [save] person_{person_id}: videos saved")

        # 6) 记录对齐与周期数据
        alignment_data = {
            "metadata": {
                "person_id": person_id,
                "offset_side_to_face": int(offset),
                "fps": fps_kpt,
                "overlap_union_range": [t0, t1],
            },
            "cycles": [],
        }

        for i, (f_cyc, s_cyc) in enumerate(zip(face_cycles, side_cycles)):
            cycle_info = {
                "cycle_index": i,
                "face_video_frames": {"start": f_cyc[0], "end": f_cyc[1]},
                "side_video_frames": {"start": s_cyc[0], "end": s_cyc[1]},
            }
            alignment_data["cycles"].append(cycle_info)

        record_path = person_log_root / f"alignment_record_{person_id}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(alignment_data, f, indent=4)

        print(f"✓ [record] person_{person_id}: saved to {record_path}\n")
        return True

    except Exception as e:
        print(f"✗ [error] person_{person_id}: {e}\n")
        return False


# -------------------- main --------------------
def main(num_threads: int = 4):
    """
    主处理函数
    
    Args:
        num_threads: 并发线程数，默认为 4
    """
    raw_root = Path("/workspace/data/raw")
    kpt_root = Path("/workspace/data/sam3d_body_results")
    log_root = Path("logs/split_cycle")

    # 获取所有人物文件夹
    person_root = kpt_root / "person"
    if not person_root.exists():
        print(f"✗ Error: person directory not found at {person_root}")
        return

    person_ids = sorted([d.name for d in person_root.iterdir() if d.is_dir()], key=int)
    # person_ids = ['5', '9', '14', '18', '29', '37', '40', '44']
    # person_ids = ['9', '29', '37', '40', '44']
    print(f"Found {len(person_ids)} persons: {person_ids}\n")
    print(f"Using {num_threads} threads for processing\n")

    # 线程安全计数器
    lock = threading.Lock()
    results = {"success": 0, "fail": 0}

    def worker(person_id: str):
        """处理单个人物的工作函数"""
        success = process_person(person_id, raw_root, kpt_root, log_root)
        with lock:
            if success:
                results["success"] += 1
            else:
                results["fail"] += 1

    # 使用 ThreadPoolExecutor 管理线程池
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = [executor.submit(worker, person_id) for person_id in person_ids]
        
        # 等待所有任务完成
        for future in futures:
            future.result()

    # 输出统计
    print(f"\n{'='*60}")
    print(f"Summary: {results['success']}/{len(person_ids)} persons processed successfully")
    print(f"         {results['fail']} persons failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    # 从命令行参数获取线程数，默认为 15
    num_threads = 11
    if len(sys.argv) > 1:
        try:
            num_threads = int(sys.argv[1])
            if num_threads < 1:
                print("Error: num_threads must be >= 1")
                sys.exit(1)
        except ValueError:
            print(f"Error: invalid num_threads '{sys.argv[1]}', must be an integer")
            sys.exit(1)
    
    main(num_threads=num_threads)
