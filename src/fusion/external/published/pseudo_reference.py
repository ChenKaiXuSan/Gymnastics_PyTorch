"""How much does a two-view triangulated pseudo-reference flatter two-view fusion?

The private dataset has no marker-based ground truth: its reference is
triangulated from the *same* two SAM3D views the methods fuse, so the
reference and the input share their detector, their image-plane noise and
their skeleton convention. That is why the depth-aware rule looks so much
stronger there (10 mm ahead of the plain body average) than on FreeMan
(2 mm), and it is the first thing a reviewer will challenge.

This module measures the artefact directly. On FreeMan, where an
independent 8-camera reference exists, it builds a pseudo-reference the
private way -- DLT triangulation of the SAM3D 2D keypoints of the two
selected views with the release's own calibration, which is the *best case*
for the pseudo-reference -- and writes it as a drop-in reference cache. Any
method can then be scored against both references with the same evaluator:

    python -m fusion external-published pseudo-reference build     # cache
    python -m fusion external-published model --dataset freeman --run <sweep> \
        --override data.options.reference_source=two_view_triangulated

The difference between the two rankings is the bias the private column
carries by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.paths import PROJECT_ROOT

CACHE_ROOT = PROJECT_ROOT / "local" / "runs" / "external_published" / "pseudo_reference" / "freeman"


def camera_matrices(camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(K, R, t)`` of a FreeMan camera (its rotation is stored as a Rodrigues vector)."""
    import cv2

    rotation = np.asarray(camera.rotation, dtype=np.float64)
    R = cv2.Rodrigues(rotation)[0] if rotation.size == 3 else rotation
    return np.asarray(camera.matrix, dtype=np.float64), R, np.asarray(camera.translation, dtype=np.float64).reshape(3)


def triangulate(points_a: np.ndarray, points_b: np.ndarray, valid: np.ndarray, cam_a, cam_b, *, scale_to_m: float) -> tuple[np.ndarray, np.ndarray]:
    """DLT triangulation of ``[F, J, 2]`` image points from two calibrated views."""
    import cv2

    Ka, Ra, ta = camera_matrices(cam_a)
    Kb, Rb, tb = camera_matrices(cam_b)
    Pa = Ka @ np.hstack([Ra, ta.reshape(3, 1)])
    Pb = Kb @ np.hstack([Rb, tb.reshape(3, 1)])
    frames, joints = points_a.shape[:2]
    out = np.zeros((frames, joints, 3), dtype=np.float32)
    ok = np.zeros((frames, joints), dtype=bool)
    flat = valid.reshape(-1)
    if flat.any():
        xa = points_a.reshape(-1, 2)[flat].T.astype(np.float64)
        xb = points_b.reshape(-1, 2)[flat].T.astype(np.float64)
        homogeneous = cv2.triangulatePoints(Pa, Pb, xa, xb)
        points = (homogeneous[:3] / homogeneous[3]).T * scale_to_m
        finite = np.isfinite(points).all(axis=-1)
        buffer = np.zeros((frames * joints, 3), dtype=np.float32)
        buffer[flat] = np.where(finite[:, None], points, 0.0)
        out = buffer.reshape(frames, joints, 3)
        mask = np.zeros(frames * joints, dtype=bool)
        mask[flat] = finite
        ok = mask.reshape(frames, joints)
    return out, ok


def build_freeman(benchmark_root: Path | None = None, *, subjects: Sequence[int] | None = None, scale_to_m: float = 0.01) -> dict[str, Any]:
    """Write ``<cache>/<session>.npz`` with the two-view triangulated reference of every session."""
    from fusion.benchmarks.freeman.dataset import _load_cameras
    from fusion.benchmarks.freeman.training import load_manifest_sessions

    from .keypoints2d import freeman_view
    from .mapping import COCO17_FROM_MHR70

    root = Path(benchmark_root or PROJECT_ROOT / "local" / "runs" / "freeman_benchmark_cluster")
    ids = list(subjects) if subjects else sorted(int(p.stem.split("_")[1]) for p in (root / "manifests").glob("subject_*_sessions.json"))
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for subject in ids:
        for session in load_manifest_sessions(root, subject):
            out = CACHE_ROOT / f"{session.session_id}.npz"
            if out.is_file():
                written.append(session.session_id)
                continue
            try:
                cameras = _load_cameras(Path(session.keypoints3d_path).parents[1] / "cameras" / f"{session.session_id}.json")
                views = [freeman_view(root, subject, session.session_id, view) for view in (session.pair.view_a, session.pair.view_b)]
                frames = min(len(views[0].frame_ids), len(views[1].frame_ids))
                index = list(COCO17_FROM_MHR70)
                points = [v.points[:frames][:, index] for v in views]
                valid = views[0].valid[:frames][:, index] & views[1].valid[:frames][:, index]
                joints, ok = triangulate(points[0], points[1], valid, cameras[session.pair.view_a], cameras[session.pair.view_b], scale_to_m=scale_to_m)
                np.savez_compressed(out, keypoints3d=joints, valid=ok, frame_ids=views[0].frame_ids[:frames], views=[session.pair.view_a, session.pair.view_b])
                written.append(session.session_id)
            except Exception as error:  # a session without a SAM3D cache is simply skipped
                skipped.append({"session": session.session_id, "error": str(error)})
    manifest = {"sessions": len(written), "skipped": skipped, "cache": str(CACHE_ROOT), "scale_to_m": scale_to_m,
                "note": "COCO17 joints triangulated from the two selected views' SAM3D 2D keypoints with the release calibration"}
    (CACHE_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[pseudo-reference] {len(written)} sessions cached, {len(skipped)} skipped -> {CACHE_ROOT}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("build",))
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--scale-to-m", type=float, default=0.01)
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_freeman(args.benchmark_root, subjects=args.subjects, scale_to_m=args.scale_to_m)
    return 0
