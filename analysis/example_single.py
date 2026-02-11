#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Example: Analyze single person
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.main import analyze_sequence


if __name__ == "__main__":
    # 指定人物ID
    person_id = "1"
    
    # 输入输出路径（使用绝对路径）
    workspace_root = Path("/workspace/code").resolve()
    person_root = workspace_root / "logs" / "fuse" / f"person_{person_id}"
    output_dir = workspace_root / "logs" / "analysis" / f"person_{person_id}"
    
    # 检查输入是否存在
    if not person_root.exists():
        print(f"Error: {person_root} does not exist")
        print("Please run the fusion process first:")
        print("  python fuse/main.py")
        sys.exit(1)
    
    # 运行分析
    print("Running analysis...")
    try:
        results = analyze_sequence(person_id, person_root, output_dir)
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("="*60)
        
        # 显示关键统计信息
        print("\nKey Statistics:")
        stats = results['statistics']
        
        print(f"\nねじれ (Twist):")
        print(f"  平均: {stats['twist']['mean']:.2f}°")
        print(f"  範囲: [{stats['twist']['min']:.2f}°, {stats['twist']['max']:.2f}°]")
        
        print(f"\n姿勢 (Trunk Tilt):")
        print(f"  平均: {stats['tilt']['mean']:.2f}°")
        print(f"  範囲: [{stats['tilt']['min']:.2f}°, {stats['tilt']['max']:.2f}°]")
        
        print(f"\n脱力 (Wrist Lead Angle):")
        print(f"  遅れてくる手: {results['lagging_hand']}")
        if stats['lead_angle']['frontal_mean'] is not None:
            print(f"  正面時の平均: {stats['lead_angle']['frontal_mean']:.2f}°")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
