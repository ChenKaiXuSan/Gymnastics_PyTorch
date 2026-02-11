#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Analysis module for gymnastics motion analysis
"""

from .load import (
    load_fused_frame,
    load_fused_sequence,
    get_fused_frame_mapping,
)

from .metrics import (
    MHR70_INDEX,
    compute_twist,
    compute_trunk_tilt,
    compute_wrist_lead_angle,
    detect_rotation_direction,
    find_frontal_facing_frames,
)

from .main import (
    analyze_sequence,
)

from .visualize import (
    plot_time_series,
    plot_distributions,
    plot_cumulative_distribution,
    plot_correlation_matrix,
    plot_all_visualizations,
)

__all__ = [
    'load_fused_frame',
    'load_fused_sequence',
    'get_fused_frame_mapping',
    'MHR70_INDEX',
    'compute_twist',
    'compute_trunk_tilt',
    'compute_wrist_lead_angle',
    'detect_rotation_direction',
    'find_frontal_facing_frames',
    'analyze_sequence',
    'plot_time_series',
    'plot_distributions',
    'plot_cumulative_distribution',
    'plot_correlation_matrix',
    'plot_all_visualizations',
]
