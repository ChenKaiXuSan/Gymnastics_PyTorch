#!/usr/bin/env python3
"""Compare the assumed camera layout against estimated per-person extrinsics.

Reprojection error alone is a weak verdict on a two-view reconstruction: it
measures epipolar consistency and says nothing about whether the recovered depth
structure is right.  This tool therefore scores both extrinsic sources on three
metrics that fail in different ways:

* **reprojection error** -- epipolar consistency, in pixels;
* **shape error against SAM3D's own monocular 3D** -- Procrustes distance after
  removing rotation and scale, so a globally warped reconstruction cannot hide;
* **bone-length stability and plausibility** -- a rigid body must keep constant
  limb lengths across frames, and those lengths must be anatomically sensible.

Run ``triangulation.estimate_extrinsics`` first to produce the extrinsics file.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf

if __package__ is None:  # allow `python triangulation/tools/compare_extrinsics.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triangulation.estimate_extrinsics import (
    _camera_frame_3d,
    _valid_mask,
    assumed_relative_pose,
    undistort,
)
from triangulation.sam3d_from_split_cycle import (
    load_calibration,
    load_sam3d_output,
    sam3d_frame_path,
    save_frame_json,
)


# MHR-70 segments that should hold a constant length on a rigid body.
BONES: List[Tuple[str, int, int]] = [
    ("shoulder_width", 5, 6),
    ("hip_width", 9, 10),
    ("l_upperarm", 5, 7),
    ("r_upperarm", 6, 8),
    ("l_thigh", 9, 11),
    ("r_thigh", 10, 12),
    ("l_shank", 11, 13),
    ("r_shank", 12, 14),
    ("l_torso", 5, 9),
    ("r_torso", 6, 10),
]
THIGH_INDEX = 4  # l_thigh, used as the headline anatomical scale check


def procrustes_mm(A: np.ndarray, B: np.ndarray) -> float:
    """Per-joint shape difference (mm) after removing translation, rotation, scale."""
    m = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    if m.sum() < 8:
        return float("nan")
    Ac, Bc = A[m] - A[m].mean(axis=0), B[m] - B[m].mean(axis=0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    scale = S.sum() / max((Bc**2).sum(), 1e-12)
    return float(np.mean(np.linalg.norm(Ac - scale * (Bc @ R.T), axis=1)) * 1000.0)


def triangulate(
    ka: np.ndarray,
    kb: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    calib_a: Dict[str, np.ndarray],
    calib_b: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate one frame in the face-camera frame; returns ``(points, mask)``."""
    mask = _valid_mask(ka) & _valid_mask(kb)
    points = np.full((ka.shape[0], 3), np.nan)
    if mask.sum() < 8:
        return points, mask
    na, nb = undistort(ka[mask], calib_a), undistort(kb[mask], calib_b)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, na.T, nb.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (X[:3] / X[3]).T
    X[~np.isfinite(X)] = np.nan
    points[mask] = X
    return points, mask


def score_person(args: Tuple[Any, ...]) -> Dict[str, Any] | None:
    """Score both extrinsic sources for one person on the same sampled frames."""
    (
        record_path,
        sam3d_root,
        max_frames,
        calib_face,
        calib_side,
        R_assumed,
        t_assumed,
        R_est,
        t_est,
    ) = args
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    person_id = str(record["metadata"]["person_id"])

    pairs: List[Tuple[int, int]] = []
    for cycle in record.get("cycles", []):
        fr, sr = cycle["face_video_frames"], cycle["side_video_frames"]
        fs, fe = int(fr["start"]), int(fr["end"])
        ss, se = int(sr["start"]), int(sr["end"])
        pairs.extend((fs + i, ss + i) for i in range(min(fe - fs, se - ss)))
    if not pairs:
        return None
    pairs = pairs[:: max(1, len(pairs) // max_frames)][:max_frames]

    focal_face = float(calib_face["K"][0, 0])
    focal_side = float(calib_side["K"][0, 0])
    variants = {"assumed": (R_assumed, t_assumed), "estimated": (R_est, t_est)}
    acc: Dict[str, Dict[str, List]] = {
        name: {"reproj": [], "shape": [], "bones": []} for name in variants
    }

    for face_idx, side_idx in pairs:
        pa = sam3d_frame_path(Path(sam3d_root), person_id, "face", face_idx)
        pb = sam3d_frame_path(Path(sam3d_root), person_id, "side", side_idx)
        if not pa.exists() or not pb.exists():
            continue
        try:
            oa, ob = load_sam3d_output(pa), load_sam3d_output(pb)
            ka = np.asarray(oa["pred_keypoints_2d"], dtype=np.float64)
            kb = np.asarray(ob["pred_keypoints_2d"], dtype=np.float64)
        except (KeyError, ValueError, OSError):
            continue
        if ka.shape != kb.shape or ka.ndim != 2:
            continue
        mono = _camera_frame_3d(oa, focal_face)

        for name, (R, t) in variants.items():
            X, mask = triangulate(ka, kb, R, t, calib_face, calib_side)
            if not mask.any() or not np.isfinite(X[mask]).all():
                continue
            rvec, _ = cv2.Rodrigues(R)
            pa_, _ = cv2.projectPoints(
                X[mask], np.zeros(3), np.zeros(3), calib_face["K"], calib_face["dist"]
            )
            pb_, _ = cv2.projectPoints(
                X[mask], rvec, t.reshape(3, 1), calib_side["K"], calib_side["dist"]
            )
            acc[name]["reproj"].append(
                0.5
                * (
                    np.mean(np.linalg.norm(pa_.reshape(-1, 2) - ka[mask], axis=1))
                    + np.mean(np.linalg.norm(pb_.reshape(-1, 2) - kb[mask], axis=1))
                )
            )
            acc[name]["shape"].append(procrustes_mm(X, mono))
            acc[name]["bones"].append(
                [np.linalg.norm(X[i] - X[j]) for _, i, j in BONES]
            )

    out: Dict[str, Any] = {"person_id": person_id, "num_frames": len(acc["assumed"]["reproj"])}
    for name in variants:
        bones = np.asarray(acc[name]["bones"], dtype=np.float64)
        if bones.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                cv_pct = np.nanmean(
                    np.nanstd(bones, axis=0) / np.abs(np.nanmean(bones, axis=0))
                ) * 100.0
            thigh = float(np.nanmedian(bones[:, THIGH_INDEX]))
        else:
            cv_pct, thigh = float("nan"), float("nan")
        out[f"{name}_reproj_px"] = _nanmean(acc[name]["reproj"])
        out[f"{name}_shape_mm"] = _nanmean(acc[name]["shape"])
        out[f"{name}_bone_cv_pct"] = float(cv_pct)
        out[f"{name}_thigh_m"] = thigh
    return out


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def _summarise(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    arr = np.asarray([r[key] for r in rows], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"median": float("nan"), "mean": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare assumed vs estimated extrinsics on triangulation quality."
    )
    parser.add_argument("--config", default="configs/sam3d_triangulation.yaml")
    parser.add_argument(
        "--extrinsics", default="logs/analysis/extrinsics/estimated_extrinsics.json"
    )
    parser.add_argument("--max-frames", type=int, default=80, help="Frames scored per person.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", default="logs/analysis/extrinsics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    extrinsics = json.loads(Path(args.extrinsics).read_text(encoding="utf-8"))

    sam3d_root = Path(cfg["paths"]["sam3d_person_root"])
    split_root = Path(cfg["paths"]["split_cycle_root"])
    calib_face = load_calibration(Path(cfg["calibration"]["face"]))
    calib_side = load_calibration(Path(cfg["calibration"]["side"]))
    R_assumed, t_assumed = assumed_relative_pose(cfg)

    tasks = []
    for record_path in sorted(split_root.glob("person_*/alignment_record_*.json")):
        pid = record_path.parent.name.removeprefix("person_")
        entry = extrinsics["persons"].get(pid)
        if entry is None:
            continue
        tasks.append(
            (
                str(record_path),
                str(sam3d_root),
                args.max_frames,
                calib_face,
                calib_side,
                R_assumed,
                t_assumed,
                np.asarray(entry["R"], dtype=np.float64),
                np.asarray(entry["t"], dtype=np.float64),
            )
        )

    print(f"[INFO] Scoring {len(tasks)} persons on {args.max_frames} frames each")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for r in pool.map(score_person, tasks) if r is not None]
    rows.sort(key=lambda r: int(r["person_id"]))

    metrics = ["reproj_px", "shape_mm", "bone_cv_pct", "thigh_m"]
    summary = {
        m: {v: _summarise(rows, f"{v}_{m}") for v in ("assumed", "estimated")}
        for m in metrics
    }
    improved = {
        m: int(
            sum(
                1
                for r in rows
                if np.isfinite(r[f"estimated_{m}"])
                and np.isfinite(r[f"assumed_{m}"])
                and r[f"estimated_{m}"] < r[f"assumed_{m}"]
            )
        )
        for m in ("reproj_px", "shape_mm")
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "extrinsics_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_frame_json(
        out_dir / "extrinsics_comparison.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "extrinsics": args.extrinsics,
            "num_persons": len(rows),
            "frames_per_person": args.max_frames,
            "summary": summary,
            "improved_person_count": improved,
        },
    )

    lines = [
        "# Extrinsics Comparison: Assumed Layout vs Estimated Per-Person",
        "",
        f"- Persons scored: `{len(rows)}`",
        f"- Frames scored per person: `{args.max_frames}`",
        f"- Extrinsics file: `{args.extrinsics}`",
        "",
        "Assumed layout = the synthetic 4-camera circle in "
        "`configs/sam3d_triangulation.yaml` (face/side 90 deg apart, 4.95 m baseline), "
        "shared by every person. Estimated = per-person essential-matrix pose with "
        "metric scale from the monocular 3D.",
        "",
        "## Aggregate",
        "",
        "| metric | source | median | mean | p90 | max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {
        "reproj_px": "reprojection (px)",
        "shape_mm": "shape err vs mono-3D (mm)",
        "bone_cv_pct": "bone-length CV (%)",
        "thigh_m": "thigh length (m)",
    }
    for m in metrics:
        for v in ("assumed", "estimated"):
            s = summary[m][v]
            lines.append(
                f"| {labels[m]} | {v} | {s['median']:.2f} | {s['mean']:.2f} "
                f"| {s['p90']:.2f} | {s['max']:.2f} |"
            )
    lines += [
        "",
        f"Persons improved on reprojection: `{improved['reproj_px']}/{len(rows)}`  ",
        f"Persons improved on shape error: `{improved['shape_mm']}/{len(rows)}`",
        "",
        "## Worst 15 Persons After Estimation (by reprojection)",
        "",
        "| person | assumed px | estimated px | assumed mm | estimated mm | est. bone CV % | est. thigh m |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    worst = sorted(
        (r for r in rows if np.isfinite(r["estimated_reproj_px"])),
        key=lambda r: -r["estimated_reproj_px"],
    )[:15]
    for r in worst:
        lines.append(
            f"| {r['person_id']} | {r['assumed_reproj_px']:.2f} | {r['estimated_reproj_px']:.2f} "
            f"| {r['assumed_shape_mm']:.1f} | {r['estimated_shape_mm']:.1f} "
            f"| {r['estimated_bone_cv_pct']:.1f} | {r['estimated_thigh_m']:.3f} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- Per-person CSV: `{out_dir / 'extrinsics_comparison.csv'}`",
        f"- Machine-readable summary: `{out_dir / 'extrinsics_comparison.json'}`",
        "",
    ]
    report = out_dir / "extrinsics_comparison.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'metric':<28} {'assumed':>12} {'estimated':>12}")
    for m in metrics:
        print(
            f"{labels[m]:<28} {summary[m]['assumed']['median']:>12.2f} "
            f"{summary[m]['estimated']['median']:>12.2f}   (median)"
        )
    print(f"\nimproved on reprojection: {improved['reproj_px']}/{len(rows)}")
    print(f"improved on shape error : {improved['shape_mm']}/{len(rows)}")
    print(f"[DONE] wrote {report}")


if __name__ == "__main__":
    main()
