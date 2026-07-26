#!/usr/bin/env python3
"""Estimate per-person face/side camera extrinsics from SAM3D 2D correspondences.

The legacy path builds extrinsics from a synthetic layout shared by every person
(``camera_position`` in ``configs/triangulation/sam3d_triangulation.yaml``: four cameras on a
3.5 m circle, face/side assumed exactly 90 deg apart with a 4.95 m baseline).
The rig was actually re-positioned between recording sessions, so a single
assumed pose cannot fit the whole dataset.  This module recovers the relative
pose per person from the data itself:

* rotation and translation *direction* from the essential matrix (RANSAC) over
  the calibrated, undistorted 2D correspondences of that person;
* metric *scale* from a Kabsch fit between the two views' monocular 3D
  predictions (``pred_keypoints_3d + pred_cam_t`` is metric in each camera
  frame);
* a clustering pass that groups persons by rig configuration.  The side camera
  sat on the subject's left in some sessions and on their right in others, which
  makes one group's relative rotation the transpose of the other's; each cluster
  gets its own consensus pose, used as a fallback when a per-person fit does not
  beat it on held-out frames.

The ``face``/``side`` folder labels are *not* interchanged anywhere in the
dataset -- SAM3D's ``global_rot`` shows the ``face`` folder holds the frontal
view for every person -- so each folder always keeps its own calibration.

Every reported error is measured on held-out frames (the fit uses even-indexed
samples, the evaluation odd-indexed ones) so the numbers are not self-confirming.
The stored extrinsics are refit on all samples once that check has passed.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from gymnastics.triangulation.sam3d_from_split_cycle import (
    load_calibration,
    load_sam3d_output,
    sam3d_frame_path,
    save_frame_json,
)


# ----------------------------- rotation helpers -----------------------------


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def geodesic_deg(A: np.ndarray, B: np.ndarray) -> float:
    return rotation_angle_deg(A.T @ B)


def chordal_mean(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """L2 (chordal) mean rotation: project the arithmetic mean back onto SO(3)."""
    U, _, Vt = np.linalg.svd(np.mean(np.asarray(rotations, dtype=np.float64), axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def robust_mean_rotation(
    rotations: Sequence[np.ndarray], reject_deg: float = 25.0, iters: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Chordal mean with iterative rejection of outlying rotations."""
    keep = np.ones(len(rotations), dtype=bool)
    R = chordal_mean(rotations)
    for _ in range(iters):
        dev = np.array([geodesic_deg(R, Ri) for Ri in rotations])
        new_keep = dev <= reject_deg
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 3 else keep
            break
        keep = new_keep
        R = chordal_mean([Ri for Ri, k in zip(rotations, keep) if k])
    return R, keep


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Similarity fit ``Q ~= s * R @ P + t`` over corresponding point sets."""
    Pc, Qc = P - P.mean(axis=0), Q - Q.mean(axis=0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = float((S[:2].sum() + d * S[2]) / max((Pc**2).sum(), 1e-12))
    return R, s, Q.mean(axis=0) - s * R @ P.mean(axis=0)


# ------------------------------ data sampling -------------------------------


def _camera_frame_3d(output: Dict[str, Any], focal_calib: float) -> np.ndarray:
    """SAM3D joints in that view's camera frame, in metres.

    SAM3D projects with its own assumed focal length (typically ~1350-1550 here,
    against a calibrated ~1710), and monocular depth scales with that assumption.
    ``pred_cam_t`` is therefore rescaled to the calibrated focal; the body's own
    geometry comes from the metric shape prior and is left untouched.
    """
    focal_sam3d = float(np.asarray(output["focal_length"]).reshape(-1)[0])
    ratio = focal_calib / focal_sam3d if focal_sam3d > 0 else 1.0
    return np.asarray(output["pred_keypoints_3d"], dtype=np.float64) + (
        np.asarray(output["pred_cam_t"], dtype=np.float64) * ratio
    )


def collect_person_samples(
    record_path: Path,
    sam3d_root: Path,
    max_frames: int,
    focal_face: float,
    focal_side: float,
) -> Dict[str, Any] | None:
    """Sample synchronized face/side frame pairs for one person.

    Pairs are formed exactly as ``sam3d_from_split_cycle.process_cycle`` does, so
    the recovered extrinsics apply to the frames the triangulator will use.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    person_id = str(record["metadata"]["person_id"])

    pairs: List[Tuple[int, int]] = []
    for cycle in record.get("cycles", []):
        face_range, side_range = cycle["face_video_frames"], cycle["side_video_frames"]
        fs, fe = int(face_range["start"]), int(face_range["end"])
        ss, se = int(side_range["start"]), int(side_range["end"])
        pairs.extend((fs + i, ss + i) for i in range(min(fe - fs, se - ss)))
    if not pairs:
        return None

    stride = max(1, len(pairs) // max_frames)
    pairs = pairs[::stride][:max_frames]

    k2a, k2b, x3a, x3b = [], [], [], []
    for face_idx, side_idx in pairs:
        pa = sam3d_frame_path(sam3d_root, person_id, "face", face_idx)
        pb = sam3d_frame_path(sam3d_root, person_id, "side", side_idx)
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
        k2a.append(ka)
        k2b.append(kb)
        x3a.append(_camera_frame_3d(oa, focal_face))
        x3b.append(_camera_frame_3d(ob, focal_side))

    if len(k2a) < 8:
        return None
    return {
        "person_id": person_id,
        "k2a": np.stack(k2a),
        "k2b": np.stack(k2b),
        "x3a": np.stack(x3a),
        "x3b": np.stack(x3b),
    }


def _collect_worker(
    args: Tuple[str, str, int, float, float]
) -> Dict[str, Any] | None:
    record_path, sam3d_root, max_frames, focal_face, focal_side = args
    return collect_person_samples(
        Path(record_path), Path(sam3d_root), max_frames, focal_face, focal_side
    )


# ---------------------------- extrinsic estimation --------------------------


def _valid_mask(k2: np.ndarray) -> np.ndarray:
    return np.isfinite(k2).all(axis=-1) & ~(k2 == 0).all(axis=-1)


def undistort(k2: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    return cv2.undistortPoints(
        k2.reshape(-1, 1, 2).astype(np.float64), calib["K"], calib["dist"]
    ).reshape(-1, 2)


def estimate_relative_pose(
    k2a: np.ndarray,
    k2b: np.ndarray,
    calib_a: Dict[str, np.ndarray],
    calib_b: Dict[str, np.ndarray],
    threshold_px: float,
) -> Tuple[np.ndarray, np.ndarray, float] | None:
    """Essential-matrix relative pose. Returns ``(R, unit_t, inlier_ratio)``."""
    mask = _valid_mask(k2a) & _valid_mask(k2b)
    if mask.sum() < 64:
        return None
    na = undistort(k2a[mask], calib_a)
    nb = undistort(k2b[mask], calib_b)
    focal = float(calib_a["K"][0, 0])
    E, inliers = cv2.findEssentialMat(
        na,
        nb,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.99999,
        threshold=threshold_px / focal,
    )
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=inliers.copy())
    return R, t.ravel(), float(inliers.mean())


def metric_scale(x3a: np.ndarray, x3b: np.ndarray) -> float:
    """Baseline length in metres, from per-frame Kabsch fits of the monocular 3D.

    The essential matrix fixes rotation and translation direction but not length,
    so the two views' metric monocular predictions supply the missing scale.
    """
    lengths = []
    for A, B in zip(x3a, x3b):
        m = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
        if m.sum() < 8:
            continue
        lengths.append(float(np.linalg.norm(kabsch(A[m], B[m])[2])))
    if not lengths:
        return float("nan")
    return float(np.median(lengths))


def reprojection_error(
    R: np.ndarray,
    t: np.ndarray,
    k2a: np.ndarray,
    k2b: np.ndarray,
    calib_a: Dict[str, np.ndarray],
    calib_b: Dict[str, np.ndarray],
) -> float:
    """Mean two-view reprojection error (px) after triangulating with ``R, t``."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    rvec, _ = cv2.Rodrigues(R)
    errors = []
    for ka, kb in zip(k2a, k2b):
        mask = _valid_mask(ka) & _valid_mask(kb)
        if mask.sum() < 8:
            continue
        na, nb = undistort(ka[mask], calib_a), undistort(kb[mask], calib_b)
        X = cv2.triangulatePoints(P1, P2, na.T, nb.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (X[:3] / X[3]).T
        if not np.isfinite(X).all():
            continue
        pa, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), calib_a["K"], calib_a["dist"])
        pb, _ = cv2.projectPoints(X, rvec, t.reshape(3, 1), calib_b["K"], calib_b["dist"])
        errors.append(
            0.5
            * (
                np.mean(np.linalg.norm(pa.reshape(-1, 2) - ka[mask], axis=1))
                + np.mean(np.linalg.norm(pb.reshape(-1, 2) - kb[mask], axis=1))
            )
        )
    return float(np.mean(errors)) if errors else float("nan")


# MHR-70 segments that stay rigid on a real body, used for the independent
# check that reprojection error cannot make.
RIGID_BONES: List[Tuple[int, int]] = [
    (5, 6),
    (9, 10),
    (5, 7),
    (6, 8),
    (9, 11),
    (10, 12),
    (11, 13),
    (12, 14),
    (5, 9),
    (6, 10),
]


def bone_length_cv(
    R: np.ndarray,
    t: np.ndarray,
    k2a: np.ndarray,
    k2b: np.ndarray,
    calib_a: Dict[str, np.ndarray],
    calib_b: Dict[str, np.ndarray],
) -> float:
    """Mean coefficient of variation of rigid bone lengths across frames, in %.

    A correct reconstruction keeps limb lengths constant.  A pose that satisfies
    the epipolar constraint while distorting depth does not, and reprojection
    error cannot tell the two apart -- two-view geometry is scale-free and a
    warped-but-consistent solution reprojects just as well.
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    rows = []
    for ka, kb in zip(k2a, k2b):
        mask = _valid_mask(ka) & _valid_mask(kb)
        if mask.sum() < 8:
            continue
        index = np.full(ka.shape[0], -1, dtype=int)
        index[mask] = np.arange(int(mask.sum()))
        na, nb = undistort(ka[mask], calib_a), undistort(kb[mask], calib_b)
        X = cv2.triangulatePoints(P1, P2, na.T, nb.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (X[:3] / X[3]).T
        rows.append(
            [
                np.linalg.norm(X[index[i]] - X[index[j]])
                if index[i] >= 0 and index[j] >= 0
                else np.nan
                for i, j in RIGID_BONES
            ]
        )
    if not rows:
        return float("nan")
    lengths = np.asarray(rows, dtype=np.float64)
    lengths[~np.isfinite(lengths)] = np.nan
    if np.all(np.isnan(lengths)):
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.nanstd(lengths, axis=0) / np.abs(np.nanmean(lengths, axis=0))
    return float(np.nanmean(cv) * 100.0)


def fit_person(
    sample: Dict[str, Any],
    calib_face: Dict[str, np.ndarray],
    calib_side: Dict[str, np.ndarray],
    threshold_px: float,
) -> Dict[str, Any] | None:
    """Fit the face->side pose for one person.

    The pose scored in ``holdout_px`` is fitted on even-indexed samples and
    scored on the odd-indexed half.  ``R_full``/``t_full`` are refit on every
    sample and are what gets stored for downstream triangulation.
    """
    k2a, k2b = sample["k2a"], sample["k2b"]
    x3a, x3b = sample["x3a"], sample["x3b"]

    fit, hold = slice(0, None, 2), slice(1, None, 2)
    pose = estimate_relative_pose(k2a[fit], k2b[fit], calib_face, calib_side, threshold_px)
    if pose is None:
        return None
    R, unit_t, inlier_ratio = pose
    scale = metric_scale(x3a[fit], x3b[fit])
    if not np.isfinite(scale) or scale <= 0:
        return None
    t = unit_t * scale

    full = estimate_relative_pose(k2a, k2b, calib_face, calib_side, threshold_px)
    R_full, t_full = R, t
    if full is not None:
        scale_full = metric_scale(x3a, x3b)
        if np.isfinite(scale_full) and scale_full > 0:
            R_full, t_full = full[0], full[1] * scale_full

    return {
        "R": R,
        "t": t,
        "R_full": R_full,
        "t_full": t_full,
        "inlier_ratio": inlier_ratio,
        "baseline_m": float(np.linalg.norm(t)),
        "angle_deg": rotation_angle_deg(R),
        "holdout_px": reprojection_error(
            R, t, k2a[hold], k2b[hold], calib_face, calib_side
        ),
        "num_frames": int(len(k2a)),
    }


# ------------------------------ rig clustering ------------------------------


def cluster_rigs(
    rotations: Dict[str, np.ndarray], iters: int = 20
) -> Tuple[Dict[str, int], np.ndarray]:
    """Split persons into the two mirror-image rig configurations.

    The side camera sat on the subject's left in some sessions and on their right
    in others.  Those two layouts differ by reflecting the side camera across the
    face camera's axis, which makes one group's face->side rotation close to the
    transpose of the other's.  Each person is assigned to whichever of ``R`` /
    ``R.T`` is nearer a running reference, which is then re-estimated; cluster 0
    is the majority configuration by convention.
    """
    pids = sorted(rotations, key=lambda p: int(p))
    reference = rotations[pids[0]]
    labels: Dict[str, int] = {p: 0 for p in pids}
    for _ in range(iters):
        labels = {
            p: int(
                geodesic_deg(reference, rotations[p].T)
                < geodesic_deg(reference, rotations[p])
            )
            for p in pids
        }
        aligned = [rotations[p].T if labels[p] else rotations[p] for p in pids]
        updated, _ = robust_mean_rotation(aligned)
        if geodesic_deg(reference, updated) < 1e-3:
            reference = updated
            break
        reference = updated
    if sum(labels.values()) * 2 > len(labels):
        labels = {p: 1 - v for p, v in labels.items()}
    return labels, reference


def consensus_per_cluster(
    results: Dict[str, Dict[str, Any]], labels: Dict[str, int]
) -> Dict[int, Dict[str, np.ndarray]]:
    """Robust consensus pose for each rig cluster, from its members' own fits."""
    consensus: Dict[int, Dict[str, np.ndarray]] = {}
    for cid in sorted(set(labels.values())):
        members = [
            results[p]
            for p, l in labels.items()
            if l == cid and p in results and np.isfinite(results[p]["holdout_px"])
        ]
        if len(members) < 3:
            continue
        R, keep = robust_mean_rotation([m["R"] for m in members])
        kept = [m for m, k in zip(members, keep) if k] or members
        direction = np.median(
            np.stack([m["t"] / max(np.linalg.norm(m["t"]), 1e-9) for m in kept]), axis=0
        )
        direction /= max(np.linalg.norm(direction), 1e-9)
        scale = float(np.median([m["baseline_m"] for m in kept]))
        consensus[cid] = {"R": R, "t": direction * scale}
    return consensus


def draw_recovered_rigs(
    clusters: Dict[int, Dict[str, np.ndarray]],
    labels: Dict[str, int],
    K: np.ndarray,
    image_size: Sequence[int],
    output_dir: Path,
) -> None:
    """Render the recovered rig geometry, one figure per cluster.

    The synthetic layout figures produced by ``camera_position_mapping`` describe
    the assumed rig, not the one the data actually came from, so a dataset built
    on estimated extrinsics needs its own pictures.
    """
    from gymnastics.triangulation.camera_position_mapping import Extrinsics, draw_cameras_matplotlib

    output_dir.mkdir(parents=True, exist_ok=True)
    for cid, pose in sorted(clusters.items()):
        R, t = pose["R"], pose["t"]
        face = Extrinsics(
            R_wc=np.eye(3), t_wc=np.zeros(3), R_cw=np.eye(3), t_cw=np.zeros(3), C=np.zeros(3)
        )
        side = Extrinsics(R_wc=R, t_wc=t, R_cw=R.T, t_cw=-R.T @ t, C=-R.T @ t)
        extr = {0: face, 1: side}
        fig, ax = draw_cameras_matplotlib(
            extr,
            K_map={0: K, 1: K},
            img_size=tuple(image_size),
            frustum_depth=0.6,
            axis_len=0.25,
            save_path=str(output_dir / f"recovered_rig_cluster{cid}.png"),
        )
        ax.set_title(
            f"rig cluster {cid} consensus - {sum(1 for l in labels.values() if l == cid)} persons, "
            f"{rotation_angle_deg(R):.1f} deg, {np.linalg.norm(t):.2f} m baseline\n"
            "camera 0 = face (world origin), camera 1 = side"
        )
        fig.savefig(
            output_dir / f"recovered_rig_cluster{cid}.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)


# ---------------------------------- driver ----------------------------------


def assumed_relative_pose(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Relative face->side pose implied by the legacy synthetic layout."""
    from gymnastics.triangulation.camera_position_mapping import (
        angles_to_position,
        build_extrinsics_map,
    )

    pos = cfg["camera_position"]
    target = tuple(float(v) for v in pos["T"])
    layout = {
        int(cid): {
            "pos": angles_to_position(
                target, float(pos["r"]), float(yaw), 0.0, float(pos["z"])
            ),
            "target": target,
        }
        for cid, yaw in pos["yaws"].items()
    }
    extr = build_extrinsics_map(layout)
    face, side = extr[int(cfg["view_camera"]["face"])], extr[int(cfg["view_camera"]["side"])]
    R = side.R_wc @ face.R_wc.T
    return R, side.t_wc - R @ face.t_wc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate per-person face/side extrinsics from SAM3D correspondences."
    )
    parser.add_argument("--config", default="configs/triangulation/sam3d_triangulation.yaml")
    parser.add_argument("--person", nargs="*", help="Optional person ids to restrict to.")
    parser.add_argument("--max-frames", type=int, default=120, help="Samples per person.")
    parser.add_argument("--threshold-px", type=float, default=1.5, help="RANSAC threshold.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--max-bone-cv-pct",
        type=float,
        default=4.5,
        help="Bone-length CV above which a per-person fit is rejected as non-rigid.",
    )
    parser.add_argument(
        "--bone-cv-ratio",
        type=float,
        default=1.8,
        help="Reject a fit whose bone-length CV exceeds this multiple of its cluster median.",
    )
    parser.add_argument(
        "--output", default="local/runs/analysis/extrinsics/estimated_extrinsics.json"
    )
    parser.add_argument(
        "--camera-figure-dir",
        default=None,
        help="Also render the recovered rig geometry here (e.g. the dataset _camera dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    sam3d_root = Path(cfg["paths"]["sam3d_person_root"])
    split_root = Path(cfg["paths"]["split_cycle_root"])
    calib_face = load_calibration(Path(cfg["calibration"]["face"]))
    calib_side = load_calibration(Path(cfg["calibration"]["side"]))
    R_assumed, t_assumed = assumed_relative_pose(cfg)

    records = sorted(split_root.glob("person_*/alignment_record_*.json"))
    if args.person:
        wanted = {str(p) for p in args.person}
        records = [p for p in records if p.parent.name.removeprefix("person_") in wanted]

    print(f"[INFO] Sampling up to {args.max_frames} frame pairs for {len(records)} persons")
    tasks = [
        (
            str(p),
            str(sam3d_root),
            args.max_frames,
            float(calib_face["K"][0, 0]),
            float(calib_side["K"][0, 0]),
        )
        for p in records
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        samples = [s for s in pool.map(_collect_worker, tasks) if s is not None]
    print(f"[INFO] Collected samples for {len(samples)} persons")

    results: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        fit = fit_person(sample, calib_face, calib_side, args.threshold_px)
        if fit is None:
            continue
        fit["assumed_holdout_px"] = reprojection_error(
            R_assumed,
            t_assumed,
            sample["k2a"][1::2],
            sample["k2b"][1::2],
            calib_face,
            calib_side,
        )
        results[sample["person_id"]] = fit
    if not results:
        raise RuntimeError("No person could be fitted; check inputs.")

    # Group persons by rig configuration.  Kabsch rotations are far steadier
    # across a person than per-frame essential-matrix ones, so cluster on those.
    kabsch_R: Dict[str, np.ndarray] = {}
    for sample in samples:
        if sample["person_id"] not in results:
            continue
        rots = []
        for A, B in zip(sample["x3a"], sample["x3b"]):
            m = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
            if m.sum() >= 8:
                rots.append(kabsch(A[m], B[m])[0])
        if rots:
            kabsch_R[sample["person_id"]] = chordal_mean(rots)

    labels, reference = cluster_rigs(kabsch_R)
    clusters = consensus_per_cluster(results, labels)
    for cid, pose in sorted(clusters.items()):
        members = [p for p, l in labels.items() if l == cid]
        print(
            f"[INFO] rig cluster {cid}: {len(members)} persons, "
            f"consensus {rotation_angle_deg(pose['R']):.1f} deg / "
            f"{np.linalg.norm(pose['t']):.2f} m baseline"
        )

    # Score every per-person fit on rigidity as well.  A pose can beat the
    # assumed layout on reprojection while distorting depth, and only the bone
    # lengths expose that.  This scores the pose that actually ships
    # (``R_full``/``t_full``) over every sampled frame -- unlike reprojection,
    # bone-length stability is not a quantity the fit optimises, so measuring it
    # on the fitting frames is not self-confirming.
    by_id = {s["person_id"]: s for s in samples}
    for pid, r in results.items():
        r["rig_cluster"] = int(labels.get(pid, -1))
        r["method"] = "per_person"
        sample = by_id[pid]
        r["bone_cv_pct"] = bone_length_cv(
            r["R_full"],
            r["t_full"],
            sample["k2a"],
            sample["k2b"],
            calib_face,
            calib_side,
        )
    cluster_cv: Dict[int, float] = {}
    for cid in sorted(set(labels.values())):
        values = [
            r["bone_cv_pct"]
            for p, r in results.items()
            if labels.get(p) == cid and np.isfinite(r["bone_cv_pct"])
        ]
        if values:
            cluster_cv[cid] = float(np.median(values))

    # Fall back to the person's own cluster consensus -- fitted over many people
    # and far better conditioned -- when their individual fit either loses to the
    # assumed layout or is anomalously non-rigid for its cluster.
    fallback = 0
    for pid, r in results.items():
        cid = labels.get(pid)
        pose = clusters.get(cid)
        if pose is None:
            continue
        limit = max(
            args.max_bone_cv_pct, args.bone_cv_ratio * cluster_cv.get(cid, args.max_bone_cv_pct)
        )
        loses = (
            not np.isfinite(r["holdout_px"]) or r["holdout_px"] > r["assumed_holdout_px"]
        )
        non_rigid = np.isfinite(r["bone_cv_pct"]) and r["bone_cv_pct"] > limit
        if not (loses or non_rigid):
            continue

        sample = by_id[pid]
        err = reprojection_error(
            pose["R"],
            pose["t"],
            sample["k2a"][1::2],
            sample["k2b"][1::2],
            calib_face,
            calib_side,
        )
        cv = bone_length_cv(
            pose["R"], pose["t"], sample["k2a"], sample["k2b"], calib_face, calib_side
        )
        if not np.isfinite(err):
            continue
        current = r["holdout_px"] if np.isfinite(r["holdout_px"]) else np.inf
        better_geometry = err < min(r["assumed_holdout_px"], current)
        steadier = (
            non_rigid
            and np.isfinite(cv)
            and cv < r["bone_cv_pct"]
            and err < r["assumed_holdout_px"]
        )
        if better_geometry or steadier:
            r.update(
                R_full=pose["R"],
                t_full=pose["t"],
                holdout_px=err,
                bone_cv_pct=cv,
                method="cluster_consensus",
            )
            fallback += 1
    print(
        f"[INFO] {fallback} persons fell back to their cluster consensus pose "
        f"(cluster bone-CV medians: "
        + ", ".join(f"{c}:{v:.1f}%" for c, v in sorted(cluster_cv.items()))
        + ")"
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "max_frames_per_person": args.max_frames,
        "ransac_threshold_px": args.threshold_px,
        "calibration": {
            "face": str(calib_face["path"]),
            "side": str(calib_side["path"]),
        },
        "rig_clusters": {
            str(cid): {
                "num_persons": sum(1 for l in labels.values() if l == cid),
                "rotation_deg": rotation_angle_deg(pose["R"]),
                "baseline_m": float(np.linalg.norm(pose["t"])),
                "R": pose["R"].tolist(),
                "t": pose["t"].tolist(),
            }
            for cid, pose in sorted(clusters.items())
        },
        "assumed": {
            "rotation_deg": rotation_angle_deg(R_assumed),
            "baseline_m": float(np.linalg.norm(t_assumed)),
        },
        "persons": {
            pid: {
                "R": r["R_full"].tolist(),
                "t": r["t_full"].tolist(),
                "rig_cluster": r["rig_cluster"],
                "method": r["method"],
                "angle_deg": rotation_angle_deg(r["R_full"]),
                "baseline_m": float(np.linalg.norm(r["t_full"])),
                "inlier_ratio": r["inlier_ratio"],
                "num_frames": r["num_frames"],
                "holdout_reproj_px": r["holdout_px"],
                "assumed_holdout_reproj_px": r["assumed_holdout_px"],
                "bone_cv_pct": r["bone_cv_pct"],
            }
            for pid, r in sorted(results.items(), key=lambda kv: int(kv[0]))
        },
    }
    out = Path(args.output)
    save_frame_json(out, payload)

    figure_dir = Path(args.camera_figure_dir) if args.camera_figure_dir else out.parent
    draw_recovered_rigs(
        clusters, labels, calib_face["K"], calib_face["image_size"].tolist(), figure_dir
    )
    print(f"[INFO] Rig figures written to {figure_dir}")

    new = np.array([r["holdout_px"] for r in results.values()])
    old = np.array([r["assumed_holdout_px"] for r in results.values()])
    ok = np.isfinite(new) & np.isfinite(old)
    print(f"\n[RESULT] held-out reprojection over {ok.sum()} persons")
    print(f"  assumed layout : median {np.median(old[ok]):6.2f} px   mean {old[ok].mean():6.2f} px")
    print(f"  estimated      : median {np.median(new[ok]):6.2f} px   mean {new[ok].mean():6.2f} px")
    print(f"  improved       : {(new[ok] < old[ok]).sum()}/{ok.sum()} persons")
    print(f"[DONE] wrote {out}")


if __name__ == "__main__":
    main()
