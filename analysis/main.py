#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/analysis/main.py
Project: /workspace/code/analysis
Created Date: Wednesday February 11th 2026
Author: Kaixu Chen
-----
Comment:
体操動作の3D解析
- ねじれ（扭转）：両肩と両股関節の回旋角度の差
- 姿勢（姿势）：体幹の傾き角度
- 脱力（放松）：手首の先行角度

Have a good code time :)
-----
Last Modified: Wednesday February 11th 2026 8:41:06 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fuse.load import load_fused_sequence as _load_fused_seq  # noqa: E402
from analysis.metrics import (  # noqa: E402
    MHR70_INDEX,
    compute_twist,
    compute_trunk_tilt,
    compute_wrist_lead_angle,
)
from analysis.visualize import (  # noqa: E402
    plot_all_visualizations,
    plot_derived_metrics_summary,
)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """简单滑动平均。"""
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """1D中值滤波（无scipy依赖）。"""
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.median(padded[i:i + window])
    return out


def _interpolate_nan(values: np.ndarray) -> np.ndarray:
    """对NaN进行线性插值（两端外推到最近有效值）。"""
    out = values.copy()
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out

    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(out)

    out[nan_mask] = np.interp(
        np.where(nan_mask)[0],
        valid_idx,
        out[valid_idx],
    )
    return out


def _filter_unsmooth_signal(
    values: np.ndarray,
    median_window: int = 9,
    z_thresh: float = 3.5,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """过滤不平滑点：MAD异常检测 + 插值 + 轻度平滑。"""
    baseline = _median_filter(values, median_window)
    residual = values - baseline

    mad = np.median(np.abs(residual - np.median(residual)))
    robust_z = np.abs(residual - np.median(residual)) / (mad + 1e-8)
    outlier_mask = robust_z > z_thresh

    cleaned = values.copy()
    cleaned[outlier_mask] = np.nan
    cleaned = _interpolate_nan(cleaned)
    smoothed = _moving_average(cleaned, smooth_window)
    return smoothed, outlier_mask


def _find_local_extrema(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回局部极大值与局部极小值。"""
    if len(values) < 3:
        return np.array([]), np.array([])

    prev_v = values[:-2]
    curr_v = values[1:-1]
    next_v = values[2:]

    max_mask = (curr_v > prev_v) & (curr_v >= next_v)
    min_mask = (curr_v < prev_v) & (curr_v <= next_v)

    local_max = curr_v[max_mask]
    local_min = curr_v[min_mask]
    return local_max, local_min


def _find_local_extrema_with_index(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回局部极大/极小的索引和值。"""
    if len(values) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    prev_v = values[:-2]
    curr_v = values[1:-1]
    next_v = values[2:]

    max_mask = (curr_v > prev_v) & (curr_v >= next_v)
    min_mask = (curr_v < prev_v) & (curr_v <= next_v)

    max_idx = np.where(max_mask)[0] + 1
    min_idx = np.where(min_mask)[0] + 1
    return max_idx, min_idx


def _compute_twist_peak_gap_deg(twist_deg: np.ndarray) -> Dict[str, Any]:
    """同时输出全局极值均值差与相邻极值周期振幅均值。"""
    local_max, local_min = _find_local_extrema(twist_deg)

    max_mean = float(local_max.mean()) if len(local_max) > 0 else None
    min_mean = float(local_min.mean()) if len(local_min) > 0 else None
    peak_gap = (max_mean - min_mean) if (
        max_mean is not None and min_mean is not None
    ) else None

    max_idx, min_idx = _find_local_extrema_with_index(twist_deg)
    extrema = []
    for idx in max_idx:
        extrema.append((int(idx), "max", float(twist_deg[idx])))
    for idx in min_idx:
        extrema.append((int(idx), "min", float(twist_deg[idx])))
    extrema.sort(key=lambda x: x[0])

    adjacent_cycle_deltas = []
    for i in range(len(extrema) - 1):
        _, kind_a, val_a = extrema[i]
        _, kind_b, val_b = extrema[i + 1]
        if kind_a != kind_b:
            adjacent_cycle_deltas.append(abs(val_b - val_a))

    cycle_amp_mean = (
        float(np.mean(adjacent_cycle_deltas))
        if len(adjacent_cycle_deltas) > 0
        else None
    )

    return {
        "peak_max_mean_deg": max_mean,
        "peak_min_mean_deg": min_mean,
        "peak_gap_deg": peak_gap,
        "peak_gap_legacy_deg": peak_gap,
        "cycle_amp_mean_deg": cycle_amp_mean,
        "n_cycle_pairs": int(len(adjacent_cycle_deltas)),
        "n_local_max": int(len(local_max)),
        "n_local_min": int(len(local_min)),
    }


def _compute_shoulder_frontal_mask(
    kpts_world: np.ndarray,
    shoulder_threshold_rad: float = 0.1,
) -> np.ndarray:
    """肩ラインが正面を向くフレームを抽出。"""
    lsho = kpts_world[:, MHR70_INDEX["lsho"], :]
    rsho = kpts_world[:, MHR70_INDEX["rsho"], :]

    shoulder_vec = rsho - lsho
    shoulder_h = shoulder_vec.copy()
    shoulder_h[:, 1] = 0

    norms = np.linalg.norm(shoulder_h, axis=1)
    valid = norms > 1e-8

    unit = np.zeros_like(shoulder_h)
    unit[valid] = shoulder_h[valid] / norms[valid, None]

    ref_axis = np.array([1.0, 0.0, 0.0])
    cos_angle = np.clip(unit @ ref_axis, -1.0, 1.0)
    shoulder_angle = np.arccos(cos_angle)

    return valid & (
        (np.abs(shoulder_angle) < shoulder_threshold_rad)
        | (np.abs(np.pi - shoulder_angle) < shoulder_threshold_rad)
    )


def _compute_frontal_wrist_position(
    kpts_world: np.ndarray,
    lagging_hand: str,
    shoulder_frontal_mask: np.ndarray,
) -> Dict[str, Any]:
    """肩が正面を向いた時点での手首位置平均を算出。"""
    wrist_idx = (
        MHR70_INDEX["rwrist"]
        if lagging_hand == "right"
        else MHR70_INDEX["lwrist"]
    )
    wrist_xyz = kpts_world[:, wrist_idx, :]

    frontal_count = int(shoulder_frontal_mask.sum())
    if frontal_count == 0:
        return {
            "frontal_frame_count": 0,
            "mean_wrist_position_xyz": None,
        }

    mean_pos = wrist_xyz[shoulder_frontal_mask].mean(axis=0)
    return {
        "frontal_frame_count": frontal_count,
        "mean_wrist_position_xyz": [
            float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2])
        ],
    }


def _group_high_low_by_median(
    person_metrics: Dict[str, Dict[str, Any]],
    metric_key: str,
) -> Dict[str, Any]:
    """指定metricで高群/低群を中央値で分割。"""
    valid_items = []
    for person_id, metrics in person_metrics.items():
        value = metrics.get(metric_key)
        if value is not None and np.isfinite(value):
            valid_items.append((person_id, float(value)))

    if not valid_items:
        return {
            "metric": metric_key,
            "median": None,
            "high_group": [],
            "low_group": [],
            "high_mean": None,
            "low_mean": None,
            "group_gap": None,
        }

    values = np.array([v for _, v in valid_items])
    median_val = float(np.median(values))

    high_group = [
        {"person_id": pid, "value": val}
        for pid, val in sorted(valid_items, key=lambda x: x[1], reverse=True)
        if val >= median_val
    ]
    low_group = [
        {"person_id": pid, "value": val}
        for pid, val in sorted(valid_items, key=lambda x: x[1])
        if val < median_val
    ]

    high_vals = np.array([item["value"] for item in high_group])
    low_vals = np.array([item["value"] for item in low_group])

    high_mean = float(high_vals.mean()) if len(high_vals) > 0 else None
    low_mean = float(low_vals.mean()) if len(low_vals) > 0 else None
    group_gap = (high_mean - low_mean) if (
        high_mean is not None and low_mean is not None
    ) else None

    return {
        "metric": metric_key,
        "median": median_val,
        "high_group": high_group,
        "low_group": low_group,
        "high_mean": high_mean,
        "low_mean": low_mean,
        "group_gap": group_gap,
    }


def _save_comparison_summary(
    output_root: Path,
    person_metrics: Dict[str, Dict[str, Any]],
) -> Path:
    """人物間比較（高くなりそうな人 vs 低くなりそうな人）を保存。"""
    comparisons = {
        "twist_cycle_amp_mean_deg": _group_high_low_by_median(
            person_metrics, "twist_cycle_amp_mean_deg"
        ),
        "twist_peak_gap_deg": _group_high_low_by_median(
            person_metrics, "twist_peak_gap_deg"
        ),
        "tilt_abs_mean_deg": _group_high_low_by_median(
            person_metrics, "tilt_abs_mean_deg"
        ),
        "wrist_frontal_mean_y": _group_high_low_by_median(
            person_metrics, "wrist_frontal_mean_y"
        ),
    }

    summary = {
        "n_persons": len(person_metrics),
        "person_metrics": person_metrics,
        "comparisons": comparisons,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_path


def _save_person_derived_values(
    output_dir: Path,
    person_id: str,
    values: Dict[str, Any],
) -> Path:
    """保存单人三项新增指标的数值文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"analysis_{person_id}_derived_values.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, ensure_ascii=False)
    return out_path


def _save_all_persons_derived_csv(
    output_root: Path,
    rows: list[Dict[str, Any]],
) -> Path:
    """保存全体人物的新增指标汇总表。"""
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "derived_metrics_all_persons.csv"
    fieldnames = [
        "person_id",
        "twist_cycle_amp_mean_deg",
        "twist_peak_gap_legacy_deg",
        "tilt_abs_mean_deg",
        "wrist_frontal_mean_x",
        "wrist_frontal_mean_y",
        "wrist_frontal_mean_z",
        "shoulder_frontal_frame_count",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


# ============================================================================
# 可视化和分析
# ============================================================================

def analyze_sequence(
    person_id: str,
    person_root: Path,
    output_dir: Optional[Path] = None,
):
    """
    分析单个人物的運動序列
    
    Args:
        person_id: 人物ID
        person_root: 人物数据根目录（完整路径）
        output_dir: 输出目录，None则使用默认路径
    """
    person_root = Path(person_root).resolve()
    
    if output_dir is None:
        output_dir = (
            Path("/workspace/code/logs/analysis") / f"person_{person_id}"
        )
    
    output_dir = Path(output_dir).resolve()
    
    print(f"\n{'='*60}")
    print(f"Analyzing person {person_id}")
    print(f"{'='*60}")
    
    # 加载数据
    kpts_world, _, metadata = _load_fused_seq(person_root, person_id)
    
    T = len(kpts_world)
    fps = metadata['fps']
    time = np.arange(T) / fps
    
    print(f"Loaded {T} frames at {fps} fps ({T/fps:.2f} seconds)")
    
    # 计算各项指标
    print("\n[1/3] Computing twist (ねじれ)...")
    twist = compute_twist(kpts_world, MHR70_INDEX)
    twist_deg_raw = np.degrees(np.unwrap(twist))
    twist_deg, twist_outlier_mask = _filter_unsmooth_signal(twist_deg_raw)
    
    print("[2/3] Computing trunk tilt (姿勢)...")
    tilt = compute_trunk_tilt(kpts_world, MHR70_INDEX)
    tilt_deg_raw = np.degrees(tilt)
    tilt_deg, tilt_outlier_mask = _filter_unsmooth_signal(tilt_deg_raw)
    
    print("[3/3] Computing wrist lead angle (脱力)...")
    lead_angle, frontal_mask, lagging_hand = compute_wrist_lead_angle(
        kpts_world, MHR70_INDEX
    )
    lead_angle_deg_raw = np.degrees(np.unwrap(lead_angle))
    lead_angle_deg, lead_outlier_mask = _filter_unsmooth_signal(
        lead_angle_deg_raw
    )

    print("\n[Derived] Computing requested summary metrics...")
    twist_peak = _compute_twist_peak_gap_deg(twist_deg)
    tilt_abs_deg = np.abs(tilt_deg)
    shoulder_frontal_mask = _compute_shoulder_frontal_mask(kpts_world)
    frontal_wrist_pos = _compute_frontal_wrist_position(
        kpts_world, lagging_hand, shoulder_frontal_mask
    )
    wrist_idx = (
        MHR70_INDEX["rwrist"]
        if lagging_hand == "right"
        else MHR70_INDEX["lwrist"]
    )
    wrist_y = kpts_world[:, wrist_idx, 1]
    print("\nFiltering summary:")
    print(f"  Twist outliers filtered: {int(twist_outlier_mask.sum())}/{T}")
    print(f"  Tilt outliers filtered: {int(tilt_outlier_mask.sum())}/{T}")
    print(f"  Lead outliers filtered: {int(lead_outlier_mask.sum())}/{T}")
    
    # 统计信息
    print("\n--- Statistics ---")
    print("Twist (ねじれ):")
    print(f"  Mean: {twist_deg.mean():.2f}°, Std: {twist_deg.std():.2f}°")
    print(f"  Range: [{twist_deg.min():.2f}°, {twist_deg.max():.2f}°]")
    
    print("\nTrunk Tilt (姿勢):")
    print(f"  Mean: {tilt_deg.mean():.2f}°, Std: {tilt_deg.std():.2f}°")
    print(f"  Range: [{tilt_deg.min():.2f}°, {tilt_deg.max():.2f}°]")
    print(f"  |Deviation| Mean (from vertical): {tilt_abs_deg.mean():.2f}°")
    
    print("\nWrist Lead Angle (脱力):")
    print(f"  Lagging hand: {lagging_hand}")
    print(f"  Frontal frames: {frontal_mask.sum()}/{T}")
    if frontal_mask.sum() > 0:
        frontal_lead = lead_angle_deg[frontal_mask]
        print(
            f"  At frontal position: "
            f"{frontal_lead.mean():.2f}° ± {frontal_lead.std():.2f}°"
        )
    print("\nRequested derived metrics:")
    print(
        "  Twist cycle amp mean (adjacent extrema): "
        f"{twist_peak['cycle_amp_mean_deg']:.2f}°"
        if twist_peak["cycle_amp_mean_deg"] is not None
        else "  Twist cycle amp mean: N/A"
    )
    print(
        "  Twist legacy peak gap (max_mean - min_mean): "
        f"{twist_peak['peak_gap_deg']:.2f}°"
        if twist_peak["peak_gap_deg"] is not None
        else "  Twist legacy peak gap: N/A"
    )
    print(f"  Tilt abs mean (vertical deviation): {tilt_abs_deg.mean():.2f}°")
    print(
        "  Shoulder-frontal wrist mean XYZ: "
        f"{frontal_wrist_pos['mean_wrist_position_xyz']} "
        f"(frames={frontal_wrist_pos['frontal_frame_count']})"
    )
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'person_id': person_id,
        'fps': float(fps),
        'n_frames': int(T),
        'twist_deg': twist_deg.tolist(),
        'tilt_deg': tilt_deg.tolist(),
        'lead_angle_deg': lead_angle_deg.tolist(),
        'filtered_outliers': {
            'twist_n': int(twist_outlier_mask.sum()),
            'tilt_n': int(tilt_outlier_mask.sum()),
            'lead_n': int(lead_outlier_mask.sum()),
        },
        'frontal_mask': frontal_mask.tolist(),
        'lagging_hand': lagging_hand,
        'derived_metrics': {
            'twist_peak': twist_peak,
            'tilt_abs_mean_deg': float(tilt_abs_deg.mean()),
            'shoulder_frontal_wrist': frontal_wrist_pos,
        },
        'statistics': {
            'twist': {
                'mean': float(twist_deg.mean()),
                'std': float(twist_deg.std()),
                'min': float(twist_deg.min()),
                'max': float(twist_deg.max()),
            },
            'tilt': {
                'mean': float(tilt_deg.mean()),
                'std': float(tilt_deg.std()),
                'min': float(tilt_deg.min()),
                'max': float(tilt_deg.max()),
            },
            'lead_angle': {
                'mean': float(lead_angle_deg.mean()),
                'std': float(lead_angle_deg.std()),
                'frontal_mean': (
                    float(lead_angle_deg[frontal_mask].mean())
                    if frontal_mask.sum() > 0 else None
                ),
            }
        }
    }
    
    result_path = output_dir / f"analysis_{person_id}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {result_path}")

    wrist_xyz = frontal_wrist_pos["mean_wrist_position_xyz"]
    derived_values = {
        "person_id": person_id,
        "twist_cycle_amp_mean_deg": twist_peak["cycle_amp_mean_deg"],
        "twist_peak_gap_legacy_deg": twist_peak["peak_gap_legacy_deg"],
        "tilt_abs_mean_deg": float(tilt_abs_deg.mean()),
        "wrist_frontal_mean_x": (
            float(wrist_xyz[0]) if wrist_xyz is not None else None
        ),
        "wrist_frontal_mean_y": (
            float(wrist_xyz[1]) if wrist_xyz is not None else None
        ),
        "wrist_frontal_mean_z": (
            float(wrist_xyz[2]) if wrist_xyz is not None else None
        ),
        "shoulder_frontal_frame_count": int(
            frontal_wrist_pos["frontal_frame_count"]
        ),
    }
    derived_values_path = _save_person_derived_values(
        output_dir, person_id, derived_values
    )
    print(f"✓ Derived values saved to {derived_values_path}")
    
    # 可视化
    print("\nGenerating visualizations...")
    plot_all_visualizations(
        time, twist_deg, tilt_deg, lead_angle_deg, frontal_mask,
        output_dir, person_id
    )
    plot_derived_metrics_summary(
        time=time,
        twist_deg=twist_deg,
        tilt_deg=tilt_deg,
        wrist_y=wrist_y,
        shoulder_frontal_mask=shoulder_frontal_mask,
        twist_peak=twist_peak,
        tilt_abs_mean_deg=float(tilt_abs_deg.mean()),
        lagging_hand=lagging_hand,
        output_path=output_dir / f"analysis_{person_id}_derived_metrics.png",
    )
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    """主函数"""
    fuse_root = Path("/workspace/code/logs/fuse").resolve()
    output_root = Path("/workspace/code/logs/analysis").resolve()
    
    # 获取所有人物
    person_dirs = sorted([
        d for d in fuse_root.iterdir()
        if d.is_dir() and d.name.startswith("person_")
    ])
    
    if not person_dirs:
        print(f"No person directories found in {fuse_root}")
        return
    
    print(f"Found {len(person_dirs)} persons")

    person_metrics_for_compare: Dict[str, Dict[str, Any]] = {}
    derived_rows: list[Dict[str, Any]] = []
    
    # 分析每个人物
    for person_dir in person_dirs:
        person_id = person_dir.name.replace("person_", "")
        output_dir = output_root / f"person_{person_id}"
        try:
            results = analyze_sequence(person_id, person_dir, output_dir)

            shoulder_frontal = results['derived_metrics'][
                'shoulder_frontal_wrist'
            ]
            wrist_mean_xyz = shoulder_frontal['mean_wrist_position_xyz']
            person_metrics_for_compare[person_id] = {
                'twist_cycle_amp_mean_deg': results['derived_metrics'][
                    'twist_peak'
                ]['cycle_amp_mean_deg'],
                'twist_peak_gap_deg': results['derived_metrics']['twist_peak'][
                    'peak_gap_legacy_deg'
                ],
                'tilt_abs_mean_deg': results['derived_metrics'][
                    'tilt_abs_mean_deg'
                ],
                'wrist_frontal_mean_y': (
                    float(wrist_mean_xyz[1])
                    if wrist_mean_xyz is not None
                    else None
                ),
            }

            derived_rows.append({
                'person_id': person_id,
                'twist_cycle_amp_mean_deg': results['derived_metrics'][
                    'twist_peak'
                ]['cycle_amp_mean_deg'],
                'twist_peak_gap_legacy_deg': results['derived_metrics'][
                    'twist_peak'
                ]['peak_gap_legacy_deg'],
                'tilt_abs_mean_deg': results['derived_metrics'][
                    'tilt_abs_mean_deg'
                ],
                'wrist_frontal_mean_x': (
                    float(wrist_mean_xyz[0])
                    if wrist_mean_xyz is not None
                    else None
                ),
                'wrist_frontal_mean_y': (
                    float(wrist_mean_xyz[1])
                    if wrist_mean_xyz is not None
                    else None
                ),
                'wrist_frontal_mean_z': (
                    float(wrist_mean_xyz[2])
                    if wrist_mean_xyz is not None
                    else None
                ),
                'shoulder_frontal_frame_count': results['derived_metrics'][
                    'shoulder_frontal_wrist'
                ]['frontal_frame_count'],
            })
        except Exception as e:  # pylint: disable=broad-except
            print(f"✗ Error analyzing person {person_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if person_metrics_for_compare:
        summary_path = _save_comparison_summary(
            output_root, person_metrics_for_compare
        )
        print(f"\n✓ Comparison summary saved to {summary_path}")
    if derived_rows:
        derived_csv_path = _save_all_persons_derived_csv(
            output_root, derived_rows
        )
        print(f"✓ Derived metrics table saved to {derived_csv_path}")
    
    print(f"\n{'='*60}")
    print("Analysis completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

