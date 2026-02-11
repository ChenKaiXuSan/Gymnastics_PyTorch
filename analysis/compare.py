#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Compare analysis results across multiple persons
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_analysis_results(analysis_root: Path):
    """加载所有人物的分析结果"""
    results = {}
    
    for person_dir in sorted(analysis_root.iterdir()):
        if (not person_dir.is_dir() or
                not person_dir.name.startswith("person_")):
            continue
        
        person_id = person_dir.name.replace("person_", "")
        result_file = person_dir / f"analysis_{person_id}.json"
        
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                results[person_id] = json.load(f)
    
    return results


def plot_comparison(results: dict, output_path: Path):
    """绘制多人物比较图"""
    person_ids = sorted(results.keys(), key=int)
    
    if not person_ids:
        print("No results to plot")
        return
    
    # 提取统计数据
    twist_means = [results[pid]['statistics']['twist']['mean']
                   for pid in person_ids]
    twist_stds = [results[pid]['statistics']['twist']['std']
                  for pid in person_ids]
    
    tilt_means = [results[pid]['statistics']['tilt']['mean']
                  for pid in person_ids]
    tilt_stds = [results[pid]['statistics']['tilt']['std']
                 for pid in person_ids]
    
    lead_means = []
    for pid in person_ids:
        frontal_mean = results[pid]['statistics']['lead_angle']['frontal_mean']
        lead_means.append(frontal_mean if frontal_mean is not None else 0)
    
    x = np.arange(len(person_ids))
    width = 0.6
    
    _, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Twist comparison
    axes[0].bar(x, twist_means, width, yerr=twist_stds, capsize=5,
                color='steelblue', alpha=0.7, label='Mean ± Std')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Twist (°)', fontsize=12)
    axes[0].set_title(
        'ねじれ (Twist) Comparison',
        fontsize=14, fontweight='bold'
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Person {pid}" for pid in person_ids])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend()
    
    # Tilt comparison
    axes[1].bar(x, tilt_means, width, yerr=tilt_stds, capsize=5,
                color='forestgreen', alpha=0.7, label='Mean ± Std')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Tilt Angle (°)', fontsize=12)
    axes[1].set_title(
        '姿勢 (Trunk Tilt) Comparison',
        fontsize=14, fontweight='bold'
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Person {pid}" for pid in person_ids])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()
    
    # Lead angle comparison
    colors = ['orangered' if results[pid]['lagging_hand'] == 'right'
              else 'darkorange' for pid in person_ids]
    bars = axes[2].bar(x, lead_means, width, color=colors, alpha=0.7)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Lead Angle (°)', fontsize=12)
    axes[2].set_title(
        '脱力 (Wrist Lead Angle at Frontal) Comparison',
        fontsize=14, fontweight='bold'
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Person {pid}" for pid in person_ids])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # 添加lagging hand标签
    for bar, pid in zip(bars, person_ids):
        hand = results[pid]['lagging_hand']
        axes[2].text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{hand[0].upper()}", ha='center', va='bottom', fontsize=9
        )
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orangered', alpha=0.7, label='Right hand lagging'),
        Patch(facecolor='darkorange', alpha=0.7, label='Left hand lagging')
    ]
    axes[2].legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved to {output_path}")


def print_summary_table(results: dict):
    """打印汇总表格"""
    person_ids = sorted(results.keys(), key=int)
    
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'Person':<10} {'Twist(°)':<15} {'Tilt(°)':<15} "
          f"{'Lead@Front(°)':<15} {'Hand':<10}")
    print("-"*80)
    
    for pid in person_ids:
        stats = results[pid]['statistics']
        twist = f"{stats['twist']['mean']:.1f}±{stats['twist']['std']:.1f}"
        tilt = f"{stats['tilt']['mean']:.1f}±{stats['tilt']['std']:.1f}"
        
        frontal_mean = stats['lead_angle']['frontal_mean']
        lead = f"{frontal_mean:.1f}" if frontal_mean is not None else "N/A"
        
        hand = results[pid]['lagging_hand']
        
        print(f"{pid:<10} {twist:<15} {tilt:<15} {lead:<15} {hand:<10}")
    
    print("="*80)


def main():
    """主函数"""
    analysis_root = Path("/workspace/code/logs/analysis").resolve()
    
    if not analysis_root.exists():
        print(f"Error: {analysis_root} does not exist")
        print("Please run the analysis first:")
        print("  python analysis/main.py")
        sys.exit(1)
    
    # 加载结果
    print("Loading analysis results...")
    results = load_analysis_results(analysis_root)
    
    if not results:
        print("No analysis results found")
        sys.exit(1)
    
    print(f"Found results for {len(results)} persons")
    
    # 打印汇总表格
    print_summary_table(results)
    
    # 绘制比较图
    print("\nGenerating comparison plot...")
    output_path = analysis_root / "comparison.png"
    plot_comparison(results, output_path)
    
    print("\n" + "="*80)
    print("Comparison completed!")
    print("="*80)


if __name__ == "__main__":
    main()
