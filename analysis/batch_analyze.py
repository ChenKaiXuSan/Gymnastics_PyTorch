#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Batch analysis for multiple persons
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.main import analyze_sequence  # noqa: E402


def batch_analyze(fuse_root: Path, output_root: Path):
    """批量分析所有人物"""
    if not fuse_root.exists():
        print(f"Error: {fuse_root} does not exist")
        sys.exit(1)
    
    # 查找所有person目录
    person_dirs = sorted(
        [d for d in fuse_root.iterdir()
         if d.is_dir() and d.name.startswith("person_")]
    )
    
    if not person_dirs:
        print(f"No person directories found in {fuse_root}")
        sys.exit(1)
    
    print(f"Found {len(person_dirs)} persons to analyze")
    print("="*80)
    
    success_count = 0
    fail_count = 0
    
    for person_dir in person_dirs:
        person_id = person_dir.name.replace("person_", "")
        output_dir = output_root / f"person_{person_id}"
        print(f"\nAnalyzing Person {person_id}...")
        print("-"*80)
        
        try:
            analyze_sequence(person_id, person_dir, output_dir)
            print(f"✓ Analysis completed for person {person_id}")
            success_count += 1
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"✗ Error analyzing person {person_id}: {e}")
            traceback.print_exc()
            fail_count += 1
    
    # 汇总
    print("\n" + "="*80)
    print("Batch Analysis Summary")
    print("="*80)
    print(f"Total persons: {len(person_dirs)}")
    print(f"Success:       {success_count}")
    print(f"Failed:        {fail_count}")
    print("="*80)


def main():
    """主函数"""
    fuse_root = Path("/workspace/code/logs/fuse").resolve()
    output_root = Path("/workspace/code/logs/analysis").resolve()
    
    print("="*80)
    print("Batch Analysis for Multiple Persons")
    print("="*80)
    print(f"Input:  {fuse_root}")
    print(f"Output: {output_root}")
    
    batch_analyze(fuse_root, output_root)
    
    print("\nAll analyses completed!")
    print("\nTo compare results across persons, run:")
    print("  python analysis/compare.py")


if __name__ == "__main__":
    main()
