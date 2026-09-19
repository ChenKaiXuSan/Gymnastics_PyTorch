"""Premise check for rotation-phase-aware view gating.

For every FreeMan session in the evaluated cohort, compute per-frame,
per-view PA error (17 joints, Procrustes per frame, mm) and the person's
facing angle theta relative to each selected camera (0 deg = facing the
camera, 90 = side-on, 180 = back turned), then test whether theta predicts
per-view reliability. Reads only existing caches and public files.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/work/1/HP260146/chenkaixu/Gymnastics_PyTorch")
sys.path.insert(0, str(REPO / "src"))
from gymnastics.common.skeletons.mhr70 import MHR70_INDEX  # noqa: E402

COCO17 = (
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
)
IDX17 = np.array([MHR70_INDEX[n] for n in COCO17])
L_HIP, R_HIP, L_SH, R_SH = 11, 12, 5, 6
UP = np.array([0.0, 0.0, 1.0])
FREEMAN = Path("/work/1/HP260146/chenkaixu/public_datasets/multiview_human/FreeMan")
OUT = REPO / "local/runs/analysis/view_reliability_phase"
OUT.mkdir(parents=True, exist_ok=True)


def load_reference(session_id: str) -> np.ndarray | None:
    for subset in ("30FPS", "60FPS"):
        p = FREEMAN / "work/shared" / subset / "keypoints3d" / f"{session_id}.npy"
        if p.is_file():
            raw = np.load(p, allow_pickle=True)
            payload = raw.item() if raw.shape == () else raw[0]
            for key in ("keypoints3d_optim", "keypoints3d"):
                if key in payload:
                    pts = np.asarray(payload[key], dtype=np.float64)
                    if pts.ndim == 4:
                        pts = pts[0]
                    return pts  # [F,17,3] in cm, world frame
    return None


def camera_towards(session_id: str, view: str, person_xy: np.ndarray) -> np.ndarray | None:
    cams = json.loads((FREEMAN / "videos_extracted" / session_id / "cameras.json").read_text())
    cam = next((c for c in cams if c["name"] == view), None)
    if cam is None:
        return None
    rot = cv2.Rodrigues(np.asarray(cam["rotation"], dtype=np.float64))[0]
    center = -rot.T @ np.asarray(cam["translation"], dtype=np.float64)
    towards = center[None, :2] - person_xy  # horizontal person->camera, [T,2]
    norm = np.linalg.norm(towards, axis=1, keepdims=True)
    return np.where(norm > 1e-6, towards / norm, np.nan)


def batched_procrustes_mm(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-frame similarity-aligned mean joint distance, in mm (inputs cm)."""
    finite = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(ref).all(axis=(1, 2))
    pred = np.where(finite[:, None, None], np.nan_to_num(pred), 0.0)
    ref_c = np.where(finite[:, None, None], np.nan_to_num(ref), 0.0)
    pc = pred - pred.mean(axis=1, keepdims=True)
    rc = ref_c - ref_c.mean(axis=1, keepdims=True)
    np_norm = np.linalg.norm(pc, axis=(1, 2))
    r_norm = np.linalg.norm(rc, axis=(1, 2))
    ok = (np_norm > 1e-8) & (r_norm > 1e-8)
    h = np.einsum("tji,tjk->tik", pc, rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", u, vt)))
    u[:, :, -1] *= d[:, None]
    rot = np.einsum("tij,tjk->tik", u, vt)
    aligned = np.einsum("tji,tik->tjk", pc, rot)
    scale = np.where(np_norm > 1e-8, r_norm / np_norm, 1.0)
    err = np.linalg.norm(aligned * scale[:, None, None] - rc, axis=2).mean(axis=1)
    return np.where(ok & finite, err * 10.0, np.nan)  # cm -> mm


def facing_theta(ref: np.ndarray, towards: np.ndarray) -> np.ndarray:
    lateral = (ref[:, L_HIP, :2] + ref[:, L_SH, :2]) / 2 - (ref[:, R_HIP, :2] + ref[:, R_SH, :2]) / 2
    facing = np.stack([lateral[:, 1], -lateral[:, 0]], axis=1)  # cross(left, up) horizontal
    fn = np.linalg.norm(facing, axis=1, keepdims=True)
    facing = np.where(fn > 1e-6, facing / fn, np.nan)
    cosang = np.clip((facing * towards).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def main() -> None:
    pairs = {}
    with open(REPO / "local/runs/freeman_benchmark_cluster/report/camera_pairs.csv") as fh:
        for row in csv.DictReader(fh):
            pairs[row["session_id"]] = (int(row["subject_id"]), row["view_a"], row["view_b"])
    rows, skipped = [], 0
    for session_id, (subject, view_a, view_b) in sorted(pairs.items()):
        ref_full = load_reference(session_id)
        if ref_full is None:
            skipped += 1
            continue
        sdir = REPO / f"local/runs/freeman_benchmark_cluster/sam3d/subject_{subject:02d}" / session_id
        per_view = {}
        for view in (view_a, view_b):
            npz_path = sdir / view / "prediction.npz"
            if not npz_path.is_file():
                break
            data = np.load(npz_path)
            frame_ids = data["frame_ids"]
            frame_ids = frame_ids[frame_ids < len(ref_full)]
            pred = data["points3d"][: len(frame_ids)][:, IDX17].astype(np.float64)
            valid = data["valid3d"][: len(frame_ids)][:, IDX17].all(axis=1)
            ref = ref_full[frame_ids]
            ref_ok = np.isfinite(ref).all(axis=(1, 2))
            person_xy = np.nan_to_num(ref[:, [L_HIP, R_HIP], :2].mean(axis=1))
            towards = camera_towards(session_id, view, person_xy)
            if towards is None:
                break
            err = batched_procrustes_mm(pred * 100.0, ref)  # pred in m-ish? see note below
            theta = facing_theta(ref, towards)
            usable = valid & ref_ok & np.isfinite(err) & np.isfinite(theta)
            per_view[view] = (frame_ids, err, theta, usable)
        if len(per_view) != 2:
            skipped += 1
            continue
        (fa, ea, ta, ua), (fb, eb, tb, ub) = per_view[view_a], per_view[view_b]
        common = min(len(fa), len(fb))
        both = ua[:common] & ub[:common]
        for i in np.nonzero(both)[0]:
            rows.append((subject, session_id, int(fa[i]),
                         round(float(ea[i]), 3), round(float(ta[i]), 2),
                         round(float(eb[i]), 3), round(float(tb[i]), 2)))
    with open(OUT / "frame_level.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "session", "frame", "err_a_mm", "theta_a_deg", "err_b_mm", "theta_b_deg"])
        w.writerows(rows)
    print(f"sessions used: {len(pairs) - skipped}/{len(pairs)}, frames: {len(rows)}")


if __name__ == "__main__":
    main()
