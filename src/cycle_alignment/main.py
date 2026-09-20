#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
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

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from common.paths import RAW_VIDEO_ROOT, SAM3D_RESULTS_ROOT
from cycle_alignment.features import IDX, compute_theta_unwrap_from_world, kpts_world_to_body
from cycle_alignment.load import load_sam3d_body_sequence
from cycle_alignment.offset import (
    align_to_common_timeline,
    choose_alignment_offset,
    crop_to_overlap,
    estimate_offset_by_audio_xcorr,
    estimate_offset_by_dtw,
)
from cycle_alignment.save import save_cycles_videos
from cycle_alignment.segmentation import (
    fuse_body_kpts,
    get_video_nframes,
    save_theta_plot,
    segment_cycles_from_fused_body,
    spans_t_to_video_spans,
)


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

        fps_kpt = 60.0

        # 2) estimate offset using theta in body frame, optionally assisted by audio
        theta_face = compute_theta_unwrap_from_world(face_k, IDX)
        theta_side = compute_theta_unwrap_from_world(side_k, IDX)

        offset_kpt = estimate_offset_by_dtw(theta_face, theta_side)
        offset_audio = None
        audio_confidence = 0.0
        try:
            offset_audio, audio_confidence = estimate_offset_by_audio_xcorr(
                face_video, side_video, fps=fps_kpt
            )
            print(
                f"✓ [align-audio] person_{person_id}: "
                f"offset_audio={offset_audio}, confidence={audio_confidence:.3f}"
            )
        except Exception as e:
            print(f"⚠ [align-audio] person_{person_id}: audio offset unavailable - {e}")

        offset, offset_source = choose_alignment_offset(
            offset_kpt=offset_kpt,
            offset_audio=offset_audio,
            audio_confidence=audio_confidence,
            tolerance_frames=10,
        )
        print(
            f"✓ [align] person_{person_id}: offset_side_to_face = {offset} "
            f"(source={offset_source}, kpt={offset_kpt}, audio={offset_audio})"
        )

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
        print(f"  [cycle] person_{person_id}: segmenting cycles...")
        
        spans_t, detection = segment_cycles_from_fused_body(
            fused_body, fps=fps_kpt, wrist_idx=IDX["rwrist"],
            min_period_sec=0.8, both_directions=True, auto_theta_ref=True, verbose=True
        )
        theta_ref_used = detection.theta_ref
        cycles_t = [(span.start, span.end) for span in spans_t]
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

        # 5) save videos (face/side) using mapping (start, mid, end per view)
        nF = get_video_nframes(face_video)
        nS = get_video_nframes(side_video)
        face_spans_all = spans_t_to_video_spans(spans_t, frame_map=face_map, n_frames=nF)
        side_spans_all = spans_t_to_video_spans(spans_t, frame_map=side_map, n_frames=nS)
        face_spans = [f for f, s_ in zip(face_spans_all, side_spans_all) if f is not None and s_ is not None]
        side_spans = [s_ for f, s_ in zip(face_spans_all, side_spans_all) if f is not None and s_ is not None]
        face_cycles = [(span.start, span.end) for span in face_spans]
        side_cycles = [(span.start, span.end) for span in side_spans]

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
                "offset_source": offset_source,
                "offset_keypoint_dtw": int(offset_kpt),
                "offset_audio_xcorr": None if offset_audio is None else int(offset_audio),
                "audio_confidence": float(audio_confidence),
                "fps": fps_kpt,
                "overlap_union_range": [t0, t1],
                # Provenance of the cycle / middle detection (see alignment/cycles.py).
                "cycle_detection": detection.to_dict(),
            },
            "cycles": [],
        }

        for i, (f_span, s_span) in enumerate(zip(face_spans, side_spans)):
            cycle_info = {
                "cycle_index": i,
                "face_video_frames": {"start": f_span.start, "mid": f_span.mid, "end": f_span.end},
                "side_video_frames": {"start": s_span.start, "mid": s_span.mid, "end": s_span.end},
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


def resolve_person_ids(person_root: Path, wanted: Optional[Sequence[str]]) -> List[str]:
    person_ids = sorted([d.name for d in person_root.iterdir() if d.is_dir()], key=int)
    if wanted is None:
        return person_ids
    wanted_set = {str(person_id) for person_id in wanted}
    return [person_id for person_id in person_ids if person_id in wanted_set]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align face/side SAM3D-Body sequences and split motion cycles."
    )
    parser.add_argument(
        "legacy_threads",
        nargs="?",
        type=int,
        help="Legacy positional thread count, e.g. python -m cycle_alignment align 11",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--person", nargs="*", default=None, help="Person ids to process")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_VIDEO_ROOT,
        help="Root that contains person/<id>/ID<id>_<view>.MOV",
    )
    parser.add_argument(
        "--kpt-root",
        type=Path,
        default=SAM3D_RESULTS_ROOT,
        help="Root that contains person/<id>/<view>/*_sam3d_body.npz",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("local/runs/split_cycle"),
        help="Output root for person_<id>/alignment_record_<id>.json",
    )
    args = parser.parse_args(argv)
    args.threads = (
        args.threads
        if args.threads is not None
        else args.legacy_threads
        if args.legacy_threads is not None
        else 11
    )
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    return args


# -------------------- main --------------------
def main(
    num_threads: int = 4,
    raw_root: Path = RAW_VIDEO_ROOT,
    kpt_root: Path = SAM3D_RESULTS_ROOT,
    log_root: Path = Path("local/runs/split_cycle"),
    person_ids: Optional[Sequence[str]] = None,
):
    """
    主处理函数
    
    Args:
        num_threads: 并发线程数，默认为 4
    """
    raw_root = Path(raw_root)
    kpt_root = Path(kpt_root)
    log_root = Path(log_root)

    # 获取所有人物文件夹
    person_root = kpt_root / "person"
    if not person_root.exists():
        print(f"✗ Error: person directory not found at {person_root}")
        return

    person_ids = resolve_person_ids(person_root, person_ids)
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


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(argv) if argv is not None else None
    if argv_list and argv_list[0] == "cycles":
        # Offline cycle/middle annotation for every dataset (alignment/annotate_cycles.py).
        from cycle_alignment.annotate_cycles import main as annotate_main

        return annotate_main(argv_list[1:])
    args = parse_args(argv_list)
    main(
        num_threads=args.threads,
        raw_root=args.raw_root,
        kpt_root=args.kpt_root,
        log_root=args.log_root,
        person_ids=args.person,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())

