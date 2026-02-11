#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Example: Load and use fused 3D keypoints
"""

from pathlib import Path
from fuse.load import load_fused_frame, load_fused_sequence, get_fused_frame_mapping


def example_load_single_frame():
    """示例：加载单个帧"""
    print("="*60)
    print("Example 1: Load single frame")
    print("="*60)
    
    person_id = "1"
    person_root = Path(f"logs/fuse/person_{person_id}")
    frame_idx = 50
    
    frame_path = person_root / "frames" / f"frame_{frame_idx:06d}.npz"
    
    if frame_path.exists():
        frame_data = load_fused_frame(frame_path)
        print(f"\nFrame {frame_data['frame_idx']}:")
        print(f"  World keypoints shape: {frame_data['kpts_world'].shape}")
        print(f"  Body keypoints shape: {frame_data['kpts_body'].shape}")
        print(f"  Face video frame idx: {frame_data['face_frame_idx']}")
        print(f"  Side video frame idx: {frame_data['side_frame_idx']}")
        print(f"\n  Sample world coords (first joint):")
        print(f"    {frame_data['kpts_world'][0]}")
    else:
        print(f"Frame not found: {frame_path}")


def example_load_full_sequence():
    """示例：加载完整序列"""
    print("\n" + "="*60)
    print("Example 2: Load full sequence")
    print("="*60)
    
    person_id = "1"
    person_root = Path(f"logs/fuse/person_{person_id}")
    
    try:
        kpts_world, kpts_body, metadata = load_fused_sequence(person_root, person_id)
        
        print(f"\nLoaded person {person_id}:")
        print(f"  Total frames: {len(kpts_world)}")
        print(f"  World keypoints shape: {kpts_world.shape}")
        print(f"  Body keypoints shape: {kpts_body.shape}")
        print(f"  FPS: {metadata['fps']}")
        print(f"  Number of joints: {metadata['n_joints']}")
        
        # 显示一些统计信息
        print(f"\n  World coords statistics:")
        print(f"    Mean: {kpts_world.mean(axis=(0,1))}")
        print(f"    Std: {kpts_world.std(axis=(0,1))}")
        print(f"    Min: {kpts_world.min(axis=(0,1))}")
        print(f"    Max: {kpts_world.max(axis=(0,1))}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")


def example_load_partial_sequence():
    """示例：加载部分帧"""
    print("\n" + "="*60)
    print("Example 3: Load partial sequence (frames 10-20)")
    print("="*60)
    
    person_id = "1"
    person_root = Path(f"logs/fuse/person_{person_id}")
    
    try:
        kpts_world, kpts_body, metadata = load_fused_sequence(
            person_root, 
            person_id, 
            start_frame=10, 
            end_frame=20
        )
        
        print(f"\nLoaded {len(kpts_world)} frames (10-19)")
        print(f"  World keypoints shape: {kpts_world.shape}")
        print(f"  Body keypoints shape: {kpts_body.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")


def example_get_frame_mapping():
    """示例：获取帧映射信息"""
    print("\n" + "="*60)
    print("Example 4: Get frame mapping")
    print("="*60)
    
    person_id = "1"
    person_root = Path(f"logs/fuse/person_{person_id}")
    
    try:
        face_map, side_map = get_fused_frame_mapping(person_root, person_id)
        
        print(f"\nFrame mapping loaded:")
        print(f"  Total frames: {len(face_map)}")
        print(f"  Face map (first 10): {face_map[:10]}")
        print(f"  Side map (first 10): {side_map[:10]}")
        
        # 找出同时在两个视角中都存在的帧
        both_valid = (face_map >= 0) & (side_map >= 0)
        print(f"\n  Frames with both views: {both_valid.sum()}/{len(face_map)}")
        
        # 找出只在face视角的帧
        face_only = (face_map >= 0) & (side_map < 0)
        print(f"  Frames with face only: {face_only.sum()}")
        
        # 找出只在side视角的帧
        side_only = (face_map < 0) & (side_map >= 0)
        print(f"  Frames with side only: {side_only.sum()}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")


def example_iterate_frames():
    """示例：逐帧迭代处理"""
    print("\n" + "="*60)
    print("Example 5: Iterate through frames")
    print("="*60)
    
    person_id = "1"
    person_root = Path(f"logs/fuse/person_{person_id}")
    frames_dir = person_root / "frames"
    
    if not frames_dir.exists():
        print(f"Frames directory not found: {frames_dir}")
        return
    
    # 获取所有帧文件
    frame_files = sorted(frames_dir.glob("frame_*.npz"))
    print(f"\nFound {len(frame_files)} frame files")
    
    # 处理前5帧作为示例
    print("\nProcessing first 5 frames:")
    for i, frame_path in enumerate(frame_files[:5]):
        frame_data = load_fused_frame(frame_path)
        
        # 计算一些统计信息
        kpts_world = frame_data['kpts_world']
        center = kpts_world.mean(axis=0)  # 计算所有关键点的中心
        
        print(f"  Frame {frame_data['frame_idx']:3d}: "
              f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")


if __name__ == "__main__":
    # 运行所有示例
    example_load_single_frame()
    example_load_full_sequence()
    example_load_partial_sequence()
    example_get_frame_mapping()
    example_iterate_frames()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
