#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Visualization tools for gymnastics analysis results
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Use default matplotlib font (English only)


def plot_time_series(
    time: np.ndarray,
    twist_deg: np.ndarray,
    tilt_deg: np.ndarray,
    lead_angle_deg: np.ndarray,
    frontal_mask: np.ndarray,
    output_path: Path,
):
    """绘制时间序列图"""
    _, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Twist
    axes[0].plot(time, twist_deg, 'b-', linewidth=1.5, label='Twist')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Twist (°)', fontsize=12)
    axes[0].set_title(
        'Twist: Shoulder - Hip Rotation Difference',
        fontsize=14
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Tilt
    axes[1].plot(time, tilt_deg, 'g-', linewidth=1.5, label='Trunk Tilt')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Tilt Angle (°)', fontsize=12)
    axes[1].set_title(
        'Posture: Trunk Tilt from Vertical',
        fontsize=14
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Lead Angle
    axes[2].plot(
        time, lead_angle_deg, 'r-',
        linewidth=1.5, alpha=0.7, label='Wrist Lead Angle'
    )
    # 标记正面时刻
    if frontal_mask.sum() > 0:
        axes[2].scatter(
            time[frontal_mask], lead_angle_deg[frontal_mask],
            c='orange', s=50, zorder=5, label='Frontal Position', alpha=0.6
        )
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Lead Angle (°)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_title(
        'Relaxation: Wrist Lead Angle at Frontal Position',
        fontsize=14
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Time series plot saved to {output_path}")


def plot_distributions(
    twist_deg: np.ndarray,
    tilt_deg: np.ndarray,
    lead_angle_deg: np.ndarray,
    frontal_mask: np.ndarray,
    output_path: Path,
):
    """绘制数值分布图（直方图 + 箱线图）"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    metrics = [
        ('Twist', twist_deg, 'steelblue'),
        ('Trunk Tilt', tilt_deg, 'forestgreen'),
        ('Lead Angle', lead_angle_deg, 'orangered'),
    ]
    
    for i, (name, data, color) in enumerate(metrics):
        # 直方图
        ax_hist = fig.add_subplot(gs[i, 0])
        _, _, _ = ax_hist.hist(
            data, bins=30, color=color, alpha=0.7, edgecolor='black'
        )
        
        # 添加统计线
        mean_val = data.mean()
        std_val = data.std()
        ax_hist.axvline(
            mean_val, color='darkred', linestyle='--',
            linewidth=2, label=f'Mean: {mean_val:.1f}°'
        )
        ax_hist.axvline(
            mean_val - std_val, color='orange', linestyle=':',
            linewidth=1.5, label='±1σ'
        )
        ax_hist.axvline(
            mean_val + std_val, color='orange', linestyle=':',
            linewidth=1.5
        )
        
        ax_hist.set_xlabel('Angle (°)', fontsize=11)
        ax_hist.set_ylabel('Frequency', fontsize=11)
        ax_hist.set_title(
            f'{name} - Distribution',
            fontsize=12, fontweight='bold'
        )
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3, axis='y')
        
        # 箱线图
        ax_box = fig.add_subplot(gs[i, 1])
        
        if i == 2 and frontal_mask.sum() > 0:
            # Lead angle: 显示全部 + 正面位置
            bp = ax_box.boxplot(
                [data, data[frontal_mask]],
                labels=['All Frames', 'Frontal Only'],
                patch_artist=True,
                widths=0.6,
            )
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            bp = ax_box.boxplot(
                [data],
                labels=['All Frames'],
                patch_artist=True,
                widths=0.5,
            )
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)
        
        ax_box.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax_box.set_ylabel('Angle (°)', fontsize=11)
        ax_box.set_title(f'{name} - Box Plot', fontsize=12, fontweight='bold')
        ax_box.grid(True, alpha=0.3, axis='y')
        
        # 添加统计文本
        stats_text = (
            f"Mean: {data.mean():.1f}°\n"
            f"Std: {data.std():.1f}°\n"
            f"Range: [{data.min():.1f}°, {data.max():.1f}°]"
        )
        if i == 2 and frontal_mask.sum() > 0:
            frontal_data = data[frontal_mask]
            stats_text += f"\n\nFrontal ({frontal_mask.sum()} frames):\n"
            stats_text += f"Mean: {frontal_data.mean():.1f}°"
        
        ax_box.text(
            0.98, 0.02, stats_text,
            transform=ax_box.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Distribution plot saved to {output_path}")


def plot_cumulative_distribution(
    twist_deg: np.ndarray,
    tilt_deg: np.ndarray,
    lead_angle_deg: np.ndarray,
    output_path: Path,
):
    """绘制累积分布函数（CDF）"""
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = [
        ('Twist', twist_deg, 'steelblue', axes[0]),
        ('Trunk Tilt', tilt_deg, 'forestgreen', axes[1]),
        ('Lead Angle', lead_angle_deg, 'orangered', axes[2]),
    ]
    
    for name, data, color, ax in metrics:
        # 计算CDF
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax.plot(sorted_data, cdf, color=color, linewidth=2)
        ax.axvline(
            data.mean(), color='darkred', linestyle='--',
            linewidth=2, label=f'Mean: {data.mean():.1f}°'
        )
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # 标记百分位数
        percentiles = [25, 50, 75]
        for p in percentiles:
            val = np.percentile(data, p)
            ax.axvline(val, color='orange', linestyle=':', alpha=0.5)
            ax.text(
                val, 0.02, f'P{p}',
                fontsize=8, rotation=90, verticalalignment='bottom'
            )
        
        ax.set_xlabel('Angle (°)', fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ CDF plot saved to {output_path}")


def plot_correlation_matrix(
    twist_deg: np.ndarray,
    tilt_deg: np.ndarray,
    lead_angle_deg: np.ndarray,
    output_path: Path,
):
    """绘制指标之间的相关性矩阵"""
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Twist vs Tilt
    axes[0].scatter(twist_deg, tilt_deg, alpha=0.5, s=20, c='steelblue')
    corr_tt = np.corrcoef(twist_deg, tilt_deg)[0, 1]
    axes[0].set_xlabel('Twist (°)', fontsize=11)
    axes[0].set_ylabel('Trunk Tilt (°)', fontsize=11)
    axes[0].set_title(
        f'Twist vs Tilt\n(r = {corr_tt:.3f})',
        fontsize=12, fontweight='bold'
    )
    axes[0].grid(True, alpha=0.3)
    
    # Twist vs Lead Angle
    axes[1].scatter(twist_deg, lead_angle_deg, alpha=0.5, s=20, c='forestgreen')
    corr_tl = np.corrcoef(twist_deg, lead_angle_deg)[0, 1]
    axes[1].set_xlabel('Twist (°)', fontsize=11)
    axes[1].set_ylabel('Lead Angle (°)', fontsize=11)
    axes[1].set_title(
        f'Twist vs Lead Angle\n(r = {corr_tl:.3f})',
        fontsize=12, fontweight='bold'
    )
    axes[1].grid(True, alpha=0.3)
    
    # Tilt vs Lead Angle
    axes[2].scatter(tilt_deg, lead_angle_deg, alpha=0.5, s=20, c='orangered')
    corr_la = np.corrcoef(tilt_deg, lead_angle_deg)[0, 1]
    axes[2].set_xlabel('Trunk Tilt (°)', fontsize=11)
    axes[2].set_ylabel('Lead Angle (°)', fontsize=11)
    axes[2].set_title(
        f'Tilt vs Lead Angle\n(r = {corr_la:.3f})',
        fontsize=12, fontweight='bold'
    )
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation plot saved to {output_path}")


def plot_all_visualizations(
    time: np.ndarray,
    twist_deg: np.ndarray,
    tilt_deg: np.ndarray,
    lead_angle_deg: np.ndarray,
    frontal_mask: np.ndarray,
    output_dir: Path,
    person_id: Optional[str] = None,
):
    """生成所有可视化图表"""
    prefix = f"analysis_{person_id}" if person_id else "analysis"
    
    # 1. 时间序列图
    plot_time_series(
        time, twist_deg, tilt_deg, lead_angle_deg, frontal_mask,
        output_dir / f"{prefix}_timeseries.png"
    )
    
    # 2. 分布图
    plot_distributions(
        twist_deg, tilt_deg, lead_angle_deg, frontal_mask,
        output_dir / f"{prefix}_distributions.png"
    )
    
    # 3. 累积分布函数
    plot_cumulative_distribution(
        twist_deg, tilt_deg, lead_angle_deg,
        output_dir / f"{prefix}_cdf.png"
    )
    
    # 4. 相关性分析
    plot_correlation_matrix(
        twist_deg, tilt_deg, lead_angle_deg,
        output_dir / f"{prefix}_correlation.png"
    )
    
    print(f"\n✓ All visualizations saved to {output_dir}")
