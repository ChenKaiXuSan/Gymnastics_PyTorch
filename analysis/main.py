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
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fuse.load import load_fused_sequence as _load_fused_seq  # noqa: E402
from analysis.metrics import (  # noqa: E402
    MHR70_INDEX,
    compute_twist,
    compute_trunk_tilt,
    compute_wrist_lead_angle,
)
from analysis.visualize import plot_all_visualizations  # noqa: E402


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
    twist_deg = np.degrees(twist)
    
    print("[2/3] Computing trunk tilt (姿勢)...")
    tilt = compute_trunk_tilt(kpts_world, MHR70_INDEX)
    tilt_deg = np.degrees(tilt)
    
    print("[3/3] Computing wrist lead angle (脱力)...")
    lead_angle, frontal_mask, lagging_hand = compute_wrist_lead_angle(
        kpts_world, MHR70_INDEX
    )
    lead_angle_deg = np.degrees(lead_angle)
    
    # 统计信息
    print("\n--- Statistics ---")
    print("Twist (ねじれ):")
    print(f"  Mean: {twist_deg.mean():.2f}°, Std: {twist_deg.std():.2f}°")
    print(f"  Range: [{twist_deg.min():.2f}°, {twist_deg.max():.2f}°]")
    
    print("\nTrunk Tilt (姿勢):")
    print(f"  Mean: {tilt_deg.mean():.2f}°, Std: {tilt_deg.std():.2f}°")
    print(f"  Range: [{tilt_deg.min():.2f}°, {tilt_deg.max():.2f}°]")
    
    print("\nWrist Lead Angle (脱力):")
    print(f"  Lagging hand: {lagging_hand}")
    print(f"  Frontal frames: {frontal_mask.sum()}/{T}")
    if frontal_mask.sum() > 0:
        frontal_lead = lead_angle_deg[frontal_mask]
        print(
            f"  At frontal position: "
            f"{frontal_lead.mean():.2f}° ± {frontal_lead.std():.2f}°"
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
        'frontal_mask': frontal_mask.tolist(),
        'lagging_hand': lagging_hand,
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
    
    # 可视化
    print("\nGenerating visualizations...")
    plot_all_visualizations(
        time, twist_deg, tilt_deg, lead_angle_deg, frontal_mask,
        output_dir, person_id
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
    
    # 分析每个人物
    for person_dir in person_dirs:
        person_id = person_dir.name.replace("person_", "")
        output_dir = output_root / f"person_{person_id}"
        try:
            analyze_sequence(person_id, person_dir, output_dir)
        except Exception as e:  # pylint: disable=broad-except
            print(f"✗ Error analyzing person {person_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Analysis completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

