#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_pt.py - 读取 pt_info 并验证形状/一致性

用法:
  python validate_pt.py /path/to/file.pt
  python validate_pt.py /path/to/dir -r
  python validate_pt.py /path/to/file.pt --expect-T 300 --expect-H 720 --expect-W 1280 --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# 尝试复用你项目里已经实现的校验函数（若不可用就用本脚本内置的精简版）
try:
    from prepare_dataset.process.preprocess import check_pt_info_shapes as _project_checker  # type: ignore
except Exception:
    _project_checker = None  # noqa: N816


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def _shape(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(x.shape)


def _infer_T_from_pt(pt: Dict[str, Any]) -> Optional[int]:
    """从常见字段推断 T；若只有光流，则返回 flow_T+1。都没有返回 None。"""
    cands: List[int] = []
    # depth/mask/bbox/kpt/kpt_score 都是严格等于 T 的候选
    for path in [
        ("depth",),
        ("YOLO", "mask"),
        ("YOLO", "bbox"),
        ("YOLO", "keypoints"),
        ("YOLO", "keypoints_score"),
        ("detectron2", "bbox"),
        ("detectron2", "keypoints"),
        ("detectron2", "keypoints_score"),
    ]:
        cur = pt
        try:
            for k in path:
                cur = cur[k]
            if _is_tensor(cur) and cur.numel() > 0:
                cands.append(int(cur.shape[0]))
        except Exception:
            pass
    if cands:
        return max(cands)

    # 没有严格候选，则尝试 optical_flow
    flow = pt.get("optical_flow", None)
    if _is_tensor(flow) and flow.dim() == 4 and flow.numel() > 0:
        return int(flow.shape[0]) + 1
    return None


def _builtin_check(
    pt_info: Dict[str, Any],
    *,
    T: Optional[int] = None,
    H: Optional[int] = None,
    W: Optional[int] = None,
    kp_num: int = 17,
    allow_empty: bool = True,
) -> Dict[str, Any]:
    """本脚本内置的精简版校验器（若项目内的更完整版本不可导入，就用它）"""
    problems: List[str] = []

    def warn(msg: str):
        problems.append(msg)

    # 推断 T/H/W
    T_infer = T if T is not None else _infer_T_from_pt(pt_info)

    # 从 depth 或 mask 推断 H/W
    hw_sources = []
    for k in ["depth"]:
        t = pt_info.get(k, None)
        if _is_tensor(t) and t.dim() == 4 and t.numel() > 0:
            hw_sources.append(("depth", int(t.shape[-2]), int(t.shape[-1])))
    ymask = pt_info.get("YOLO", {}).get("mask", None)
    if _is_tensor(ymask) and ymask.dim() == 4 and ymask.numel() > 0:
        hw_sources.append(("YOLO.mask", int(ymask.shape[-2]), int(ymask.shape[-1])))

    if not hw_sources:
        flow = pt_info.get("optical_flow", None)
        if _is_tensor(flow) and flow.dim() == 4 and flow.numel() > 0:
            hw_sources.append(
                ("optical_flow", int(flow.shape[-2]), int(flow.shape[-1]))
            )

    if hw_sources:
        src, H0, W0 = hw_sources[0]
    else:
        H0 = W0 = None

    H_infer = H if H is not None else H0
    W_infer = W if W is not None else W0

    # 开始逐字段检查
    depth = pt_info.get("depth", None)
    if _is_tensor(depth) and (allow_empty or depth.numel() > 0):
        if depth.dim() != 4:
            warn(f"depth 维度应为 4，实际 {_shape(depth)}")
        else:
            if depth.shape[1] != 1:
                warn(f"depth 通道应为 1，实际 {depth.shape[1]}")
            if T_infer is not None and depth.shape[0] != T_infer:
                warn(f"depth T={depth.shape[0]} != {T_infer}")
            if H_infer is not None and depth.shape[2] != H_infer:
                warn(f"depth H={depth.shape[2]} != {H_infer}")
            if W_infer is not None and depth.shape[3] != W_infer:
                warn(f"depth W={depth.shape[3]} != {W_infer}")

    flow = pt_info.get("optical_flow", None)
    if _is_tensor(flow) and (allow_empty or flow.numel() > 0):
        if flow.dim() != 4:
            warn(f"optical_flow 维度应为 4，实际 {_shape(flow)}")
        else:
            if flow.shape[1] != 2:
                warn(f"optical_flow 通道应为 2，实际 {flow.shape[1]}")
            if T_infer is not None:
                exp = max(T_infer - 1, 0)
                if flow.shape[0] != exp:
                    warn(f"optical_flow T={flow.shape[0]} 应为 T-1={exp}")
            if H_infer is not None and flow.shape[2] != H_infer:
                warn(f"optical_flow H={flow.shape[2]} != {H_infer}")
            if W_infer is not None and flow.shape[3] != W_infer:
                warn(f"optical_flow W={flow.shape[3]} != {W_infer}")

    def check_det(prefix: str, d: Dict[str, Any]):
        bbox = d.get("bbox", None)
        if _is_tensor(bbox) and (allow_empty or bbox.numel() > 0):
            if not (bbox.dim() == 2 and bbox.shape[1] == 4):
                warn(f"{prefix}.bbox 应为 (T,4)，实际 {_shape(bbox)}")
            elif T_infer is not None and bbox.shape[0] != T_infer:
                warn(f"{prefix}.bbox T={bbox.shape[0]} != {T_infer}")

        mask = d.get("mask", None)
        if _is_tensor(mask) and (allow_empty or mask.numel() > 0):
            if not (mask.dim() == 4 and mask.shape[1] == 1):
                warn(f"{prefix}.mask 应为 (T,1,H,W)，实际 {_shape(mask)}")
            else:
                if T_infer is not None and mask.shape[0] != T_infer:
                    warn(f"{prefix}.mask T={mask.shape[0]} != {T_infer}")
                if H_infer is not None and mask.shape[2] != H_infer:
                    warn(f"{prefix}.mask H={mask.shape[2]} != {H_infer}")
                if W_infer is not None and mask.shape[3] != W_infer:
                    warn(f"{prefix}.mask W={mask.shape[3]} != {W_infer}")

        kpt = d.get("keypoints", None)
        if _is_tensor(kpt) and (allow_empty or kpt.numel() > 0):
            if not (kpt.dim() == 3 and kpt.shape[1] == 17 and kpt.shape[2] in (2, 3)):
                warn(f"{prefix}.keypoints 应为 (T,17,2|3)，实际 {_shape(kpt)}")
            elif T_infer is not None and kpt.shape[0] != T_infer:
                warn(f"{prefix}.keypoints T={kpt.shape[0]} != {T_infer}")

        ks = d.get("keypoints_score", None)
        if _is_tensor(ks) and (allow_empty or ks.numel() > 0):
            if not (ks.dim() == 2 and ks.shape[1] == 17):
                warn(f"{prefix}.keypoints_score 应为 (T,17)，实际 {_shape(ks)}")
            elif T_infer is not None and ks.shape[0] != T_infer:
                warn(f"{prefix}.keypoints_score T={ks.shape[0]} != {T_infer}")

    if isinstance(pt_info.get("YOLO", None), dict):
        check_det("YOLO", pt_info["YOLO"])
    if isinstance(pt_info.get("detectron2", None), dict):
        check_det("detectron2", pt_info["detectron2"])

    # frames / frames_path 兜底
    frames = pt_info.get("frames", None)
    if frames is None and isinstance(pt_info.get("frames_path", None), str):
        try:
            frames = torch.load(pt_info["frames_path"], map_location="cpu")
        except Exception as e:
            warn(f"无法读取 frames_path: {pt_info['frames_path']} ({e})")

    if _is_tensor(frames) and frames.numel() > 0:
        if not (frames.dim() == 4 and frames.shape[-1] in (1, 3)):
            warn(f"frames 应为 (T,H,W,C)，实际 {_shape(frames)}")
        else:
            if T_infer is not None and frames.shape[0] != T_infer:
                warn(f"frames T={frames.shape[0]} != {T_infer}")

    ok = len(problems) == 0
    return {"ok": ok, "T": T_infer, "H": H_infer, "W": W_infer, "problems": problems}


def check_pt_info_shapes(
    pt_info: Dict[str, Any],
    *,
    T: Optional[int] = None,
    H: Optional[int] = None,
    W: Optional[int] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """统一入口：优先调用项目内的正式版；否则用内置精简版。"""
    if _project_checker is not None:
        report = _project_checker(
            pt_info, T=T, H=H, W=W, strict=False, check_frames=True  # 这里不抛错，由本函数处理
        )
    else:
        report = _builtin_check(pt_info, T=T, H=H, W=W)

    if strict and not report["ok"]:
        raise AssertionError(
            "Shape check failed:\n- " + "\n- ".join(report["problems"])
        )
    return report


def validate_one(
    path: Path,
    *,
    expect_T: Optional[int],
    expect_H: Optional[int],
    expect_W: Optional[int],
    strict: bool,
) -> bool:
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[FAIL] {path}: 无法加载（{e}）")
        return False

    if not isinstance(obj, dict) or "optical_flow" not in obj or "YOLO" not in obj:
        print(f"[FAIL] {path}: 不是预期的 pt_info 字典")
        return False

    report = check_pt_info_shapes(obj, T=expect_T, H=expect_H, W=expect_W, strict=False)
    status = "OK" if report["ok"] else "FAIL"
    print(f"[{status}] {path}  T={report['T']}  HxW={report['H']}x{report['W']}")
    if not report["ok"]:
        for p in report["problems"]:
            print("  -", p)
        if strict:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Validate pt_info shapes")
    ap.add_argument("path", type=str, help="pt 文件或目录")
    ap.add_argument("-r", "--recursive", action="store_true", help="目录递归")
    ap.add_argument("--expect-T", type=int, default=None, help="期望帧数 T（可选）")
    ap.add_argument("--expect-H", type=int, default=None, help="期望高度 H（可选）")
    ap.add_argument("--expect-W", type=int, default=None, help="期望宽度 W（可选）")
    ap.add_argument("--strict", action="store_true", help="有问题时返回非 0")
    args = ap.parse_args()

    p = Path(args.path)
    targets: List[Path] = []
    if p.is_file():
        targets = [p]
    elif p.is_dir():
        if args.recursive:
            targets = [q for q in p.rglob("*.pt")]
        else:
            targets = [q for q in p.glob("*.pt")]
    else:
        print(f"路径不存在: {p}")
        sys.exit(1)

    all_ok = True
    for f in sorted(targets):
        ok = validate_one(
            f,
            expect_T=args.expect_T,
            expect_H=args.expect_H,
            expect_W=args.expect_W,
            strict=args.strict,
        )
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
