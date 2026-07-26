#!/usr/bin/env python3
"""Express fusion errors as a fraction of each subject's own body size.

Two-view geometry is scale-free: the extrinsics recovered by
``triangulation.estimate_extrinsics`` fix rotation and translation *direction*,
while metric scale comes from SAM3D's monocular predictions and is not
independently calibrated.  If that scale is off by a factor ``s``, every
similarity-aligned MPJPE in ``local/runs/fuse_experiments`` is off by the same ``s`` --
method rankings are untouched, but the absolute millimetres are not.

Dividing each person's error by that person's own limb-chain length cancels
``s`` exactly, so the resulting percentages hold regardless of how the metric
scale is eventually pinned down.  The chain (ankle-knee-hip-shoulder) is summed
from per-frame medians, which is robust to individual joint outliers and to the
subject moving.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

DEFAULT_TRIANGULATED_ROOT = Path("/home/data/xchen/gymnastics/sam3d_triangulated/person")
DEFAULT_METRICS = Path("local/runs/fuse_experiments/metrics_by_person.csv")
DEFAULT_OUT_DIR = Path("local/runs/analysis/body_scale_normalized")

# MHR-70 indices: 5/6 shoulders, 9/10 hips, 11/12 knees, 13/14 ankles.
CHAIN = (("ankle_knee", 11, 13), ("knee_hip", 9, 11), ("hip_shoulder", 5, 9))
# Head/neck above the shoulder line, used only for the stature estimate.
HEAD_NECK_ALLOWANCE_M = 0.255


def person_body_scale(person_root: Path) -> Dict[str, float]:
    """Median limb-chain length (m) for one person, over every triangulated frame."""
    sequences = []
    for path in sorted(person_root.glob("cycle_*/joints_3d_sequence.npz")):
        with np.load(path, allow_pickle=True) as data:
            joints = np.asarray(data["joints_3d"], dtype=np.float64)
        if joints.size:
            sequences.append(joints)
    if not sequences:
        return {}
    joints = np.concatenate(sequences, axis=0)

    segments: Dict[str, float] = {}
    for name, a, b in CHAIN:
        lengths = np.linalg.norm(joints[:, a] - joints[:, b], axis=1)
        lengths = lengths[np.isfinite(lengths)]
        if lengths.size:
            segments[name] = float(np.median(lengths))
    if len(segments) != len(CHAIN):
        return {}
    segments["chain"] = float(sum(segments[name] for name, _, _ in CHAIN))
    segments["stature_estimate"] = segments["chain"] + HEAD_NECK_ALLOWANCE_M
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize fusion MPJPE by each subject's own body scale."
    )
    parser.add_argument("--triangulated-root", type=Path, default=DEFAULT_TRIANGULATED_ROOT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--reference-stature",
        type=float,
        nargs="*",
        default=[1.50, 1.55, 1.60, 1.65],
        help="Candidate true mean statures (m) to report the implied scale factor for.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scales: Dict[str, Dict[str, float]] = {}
    for person_root in sorted(
        args.triangulated_root.glob("person_*"),
        key=lambda p: int(p.name.removeprefix("person_")),
    ):
        segments = person_body_scale(person_root)
        if segments:
            scales[person_root.name.removeprefix("person_")] = segments
    if not scales:
        raise SystemExit(f"No triangulated sequences under {args.triangulated_root}")

    rows: List[Dict[str, object]] = []
    by_method: Dict[str, List[float]] = defaultdict(list)
    missing = set()
    for record in csv.DictReader(args.metrics.open(encoding="utf-8")):
        person_id, method = record["person_id"], record["method"]
        segments = scales.get(person_id)
        if segments is None:
            missing.add(person_id)
            continue
        try:
            mpjpe = float(record["mpjpe"])
        except ValueError:
            continue
        if not np.isfinite(mpjpe):
            continue
        pct = 100.0 * mpjpe / segments["chain"]
        rows.append(
            {
                "person_id": person_id,
                "method": method,
                "mpjpe_m": mpjpe,
                "body_chain_m": round(segments["chain"], 6),
                "stature_estimate_m": round(segments["stature_estimate"], 6),
                "mpjpe_pct_body": round(pct, 6),
            }
        )
        by_method[method].append(pct)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "metrics_by_person_normalized.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    chain = np.array([s["chain"] for s in scales.values()])
    stature = np.array([s["stature_estimate"] for s in scales.values()])
    ranking = sorted(by_method.items(), key=lambda kv: float(np.mean(kv[1])))
    summary = {
        "persons": len(scales),
        "body_chain_m": {
            "median": float(np.median(chain)),
            "p5": float(np.percentile(chain, 5)),
            "p95": float(np.percentile(chain, 95)),
        },
        "stature_estimate_m": {
            "median": float(np.median(stature)),
            "p5": float(np.percentile(stature, 5)),
            "p95": float(np.percentile(stature, 95)),
            "head_neck_allowance_m": HEAD_NECK_ALLOWANCE_M,
        },
        "implied_scale_factor": {
            f"{h:.2f}": float(h / np.median(stature)) for h in args.reference_stature
        },
        "methods": {
            method: {
                "mean_pct_body": float(np.mean(values)),
                "median_pct_body": float(np.median(values)),
                "persons": len(values),
            }
            for method, values in ranking
        },
        "note": (
            "mpjpe_pct_body is invariant to the unknown metric scale of the "
            "triangulated pseudo-GT; absolute millimetres are proportional to it."
        ),
    }
    if missing:
        summary["persons_missing_body_scale"] = sorted(missing, key=int)
    (args.out_dir / "body_scale_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"persons with body scale: {len(scales)}")
    print(
        f"limb chain (ankle->shoulder): median {np.median(chain):.3f} m "
        f"[{np.percentile(chain, 5):.3f}, {np.percentile(chain, 95):.3f}]"
    )
    print(
        f"stature estimate: median {np.median(stature):.2f} m "
        f"[{np.percentile(stature, 5):.2f}, {np.percentile(stature, 95):.2f}]"
    )
    print("\nimplied scale factor if the true mean stature were:")
    for h in args.reference_stature:
        print(f"  {h:.2f} m -> s = {h / np.median(stature):.3f}")
    print(f"\n{'method':<36}{'% body':>9}{'mm':>9}")
    print("-" * 54)
    for method, values in ranking:
        mm = 1000.0 * float(np.mean([r["mpjpe_m"] for r in rows if r["method"] == method]))
        print(f"{method:<36}{np.mean(values):9.2f}{mm:9.1f}")
    print(f"\n[save] {out_csv}")
    print(f"[save] {args.out_dir / 'body_scale_summary.json'}")


if __name__ == "__main__":
    main()
