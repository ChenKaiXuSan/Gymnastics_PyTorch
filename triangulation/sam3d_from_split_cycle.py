#!/usr/bin/env python3
"""Triangulate SAM3D-Body 2D keypoints using split-cycle alignment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from triangulation.camera_position_mapping import prepare_camera_position


MHR70_SKELETON: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 62),
    (6, 8),
    (8, 41),
    (5, 9),
    (6, 10),
    (9, 10),
    (9, 11),
    (11, 13),
    (10, 12),
    (12, 14),
    (13, 15),
    (13, 16),
    (13, 17),
    (14, 18),
    (14, 19),
    (14, 20),
    (5, 69),
    (6, 69),
    (7, 63),
    (8, 64),
]


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def load_sam3d_output(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    output = data["output"]
    return output.item() if output.shape == () else output.tolist()


def load_keypoints_2d(path: Path) -> np.ndarray:
    output = load_sam3d_output(path)
    if "pred_keypoints_2d" not in output:
        raise KeyError(f"pred_keypoints_2d not found in {path}")
    kpts = _as_array(output["pred_keypoints_2d"])
    if kpts.ndim != 2 or kpts.shape[1] != 2:
        raise ValueError(f"Expected (J,2) keypoints in {path}, got {kpts.shape}")
    return kpts


def sam3d_frame_path(root: Path, person_id: str, view: str, frame_idx: int) -> Path:
    return root / person_id / view / f"{frame_idx:06d}_sam3d_body.npz"


def load_calibration(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "path": np.asarray(str(path)),
        "K": np.asarray(data["camera_matrix"], dtype=np.float32).reshape(3, 3),
        "dist": np.asarray(data["dist_coeffs"], dtype=np.float32).reshape(-1),
        "image_size": np.asarray(data["image_size"], dtype=np.int32),
    }


def undistort_keypoints(kpts: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    if len(kpts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.undistortPoints(
        kpts.reshape(-1, 1, 2).astype(np.float32),
        calib["K"],
        calib["dist"],
    ).reshape(-1, 2)


def triangulate_keypoints(
    face_kpts: np.ndarray,
    side_kpts: np.ndarray,
    face_calib: Dict[str, np.ndarray],
    side_calib: Dict[str, np.ndarray],
    face_rt: Dict[str, np.ndarray],
    side_rt: Dict[str, np.ndarray],
) -> np.ndarray:
    if face_kpts.shape != side_kpts.shape:
        raise ValueError(f"Shape mismatch: {face_kpts.shape} vs {side_kpts.shape}")

    P_face = np.hstack([face_rt["R"], face_rt["t"].reshape(3, 1)]).astype(np.float32)
    P_side = np.hstack([side_rt["R"], side_rt["t"].reshape(3, 1)]).astype(np.float32)

    valid = np.isfinite(face_kpts).all(axis=1) & np.isfinite(side_kpts).all(axis=1)
    valid &= ~((face_kpts == 0).all(axis=1) | (side_kpts == 0).all(axis=1))

    joints_3d = np.full((face_kpts.shape[0], 3), np.nan, dtype=np.float32)
    if valid.sum() < 2:
        return joints_3d

    face_norm = undistort_keypoints(face_kpts[valid], face_calib)
    side_norm = undistort_keypoints(side_kpts[valid], side_calib)
    pts4d = cv2.triangulatePoints(P_face, P_side, face_norm.T, side_norm.T)
    pts3d = (pts4d[:3] / pts4d[3]).T.astype(np.float32)
    pts3d[~np.isfinite(pts3d)] = np.nan
    joints_3d[valid] = pts3d
    return joints_3d


def reprojection_errors(
    joints_3d: np.ndarray,
    kpts_2d: np.ndarray,
    calib: Dict[str, np.ndarray],
    rt: Dict[str, np.ndarray],
) -> np.ndarray:
    valid = np.isfinite(joints_3d).all(axis=1) & np.isfinite(kpts_2d).all(axis=1)
    errors = np.full((joints_3d.shape[0],), np.nan, dtype=np.float32)
    if not valid.any():
        return errors

    rvec, _ = cv2.Rodrigues(np.asarray(rt["R"], dtype=np.float32))
    projected, _ = cv2.projectPoints(
        joints_3d[valid].astype(np.float32),
        rvec,
        np.asarray(rt["t"], dtype=np.float32).reshape(3, 1),
        calib["K"],
        calib["dist"],
    )
    projected = projected.reshape(-1, 2)
    errors[valid] = np.linalg.norm(projected - kpts_2d[valid], axis=1)
    return errors


def save_frame_json(
    path: Path,
    frame_data: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(frame_data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def plot_3d_keypoints(
    joints_3d: np.ndarray,
    save_path: Path,
    title: str,
    skeleton: Iterable[Tuple[int, int]] = MHR70_SKELETON,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    valid = np.isfinite(joints_3d).all(axis=1)
    if valid.any():
        pts = joints_3d[valid]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, c="tab:blue")
        for i, j in skeleton:
            if i < len(joints_3d) and j < len(joints_3d) and valid[i] and valid[j]:
                seg = joints_3d[[i, j]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="tab:orange", lw=1.2)
        center = np.nanmean(pts, axis=0)
        radius = max(float(np.nanmax(np.ptp(pts, axis=0))) * 0.6, 0.5)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def frames_to_video(frame_dir: Path, output_path: Path, fps: int) -> None:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    try:
        for frame in frames:
            img = cv2.imread(str(frame))
            if img is not None:
                writer.write(img)
    finally:
        writer.release()


def process_cycle(
    person_id: str,
    cycle: Dict[str, Any],
    sam3d_root: Path,
    output_person_root: Path,
    face_calib: Dict[str, np.ndarray],
    side_calib: Dict[str, np.ndarray],
    face_rt: Dict[str, np.ndarray],
    side_rt: Dict[str, np.ndarray],
    vis_stride: int,
    save_vis: bool,
    make_video: bool,
    fps: int,
    max_frames: int | None = None,
) -> Dict[str, Any]:
    cycle_idx = int(cycle["cycle_index"])
    face_range = cycle["face_video_frames"]
    side_range = cycle["side_video_frames"]
    face_start, face_end = int(face_range["start"]), int(face_range["end"])
    side_start, side_end = int(side_range["start"]), int(side_range["end"])
    length = min(face_end - face_start, side_end - side_start)
    if max_frames is not None:
        length = min(length, max_frames)

    cycle_root = output_person_root / f"cycle_{cycle_idx:03d}"
    joints_dir = cycle_root / "joints_3d"
    vis_dir = cycle_root / "visualization"
    joints_seq: List[np.ndarray] = []
    frame_records: List[Dict[str, Any]] = []
    missing = 0
    face_error_means: List[float] = []
    side_error_means: List[float] = []

    for local_idx in range(length):
        face_idx = face_start + local_idx
        side_idx = side_start + local_idx
        face_path = sam3d_frame_path(sam3d_root, person_id, "face", face_idx)
        side_path = sam3d_frame_path(sam3d_root, person_id, "side", side_idx)

        if not face_path.exists() or not side_path.exists():
            missing += 1
            continue

        face_kpts = load_keypoints_2d(face_path)
        side_kpts = load_keypoints_2d(side_path)
        joints_3d = triangulate_keypoints(
            face_kpts,
            side_kpts,
            face_calib=face_calib,
            side_calib=side_calib,
            face_rt=face_rt,
            side_rt=side_rt,
        )
        face_errors = reprojection_errors(joints_3d, face_kpts, face_calib, face_rt)
        side_errors = reprojection_errors(joints_3d, side_kpts, side_calib, side_rt)
        if np.isfinite(face_errors).any():
            face_error_means.append(float(np.nanmean(face_errors)))
        if np.isfinite(side_errors).any():
            side_error_means.append(float(np.nanmean(side_errors)))
        joints_seq.append(joints_3d)

        record = {
            "person_id": person_id,
            "cycle_index": cycle_idx,
            "cycle_frame_index": local_idx,
            "face_frame_index": face_idx,
            "side_frame_index": side_idx,
            "face_sam3d_path": str(face_path),
            "side_sam3d_path": str(side_path),
            "num_joints": int(joints_3d.shape[0]),
            "valid_joints": int(np.isfinite(joints_3d).all(axis=1).sum()),
            "face_reprojection_error_px": face_errors.tolist(),
            "side_reprojection_error_px": side_errors.tolist(),
            "face_reprojection_error_mean_px": (
                float(np.nanmean(face_errors)) if np.isfinite(face_errors).any() else None
            ),
            "side_reprojection_error_mean_px": (
                float(np.nanmean(side_errors)) if np.isfinite(side_errors).any() else None
            ),
            "joints_3d": joints_3d.tolist(),
        }
        frame_records.append(record)
        save_frame_json(joints_dir / f"{local_idx:06d}_joints_3d.json", record)

        if save_vis and local_idx % max(vis_stride, 1) == 0:
            plot_3d_keypoints(
                joints_3d,
                vis_dir / f"{local_idx:06d}.png",
                title=f"person {person_id} cycle {cycle_idx} frame {local_idx}",
            )

    sequence = np.stack(joints_seq, axis=0) if joints_seq else np.empty((0, 70, 3))
    cycle_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cycle_root / "joints_3d_sequence.npz",
        joints_3d=sequence,
        frame_records=np.asarray(frame_records, dtype=object),
    )

    summary = {
        "person_id": person_id,
        "cycle_index": cycle_idx,
        "face_video_frames": face_range,
        "side_video_frames": side_range,
        "processed_frames": int(sequence.shape[0]),
        "missing_pairs": int(missing),
        "num_joints": int(sequence.shape[1]) if sequence.size else 70,
        "face_reprojection_error_mean_px": (
            float(np.mean(face_error_means)) if face_error_means else None
        ),
        "side_reprojection_error_mean_px": (
            float(np.mean(side_error_means)) if side_error_means else None
        ),
    }
    save_frame_json(cycle_root / "summary.json", summary)

    if save_vis and make_video:
        frames_to_video(vis_dir, cycle_root / f"cycle_{cycle_idx:03d}_3d.mp4", fps=fps)

    return summary


def process_person(
    record_path: Path,
    sam3d_root: Path,
    output_root: Path,
    face_calib: Dict[str, np.ndarray],
    side_calib: Dict[str, np.ndarray],
    rt_info: Dict[int, Dict[str, np.ndarray]],
    face_camera_id: int,
    side_camera_id: int,
    vis_stride: int,
    save_vis: bool,
    make_video: bool,
    fps: int,
    max_cycles: int | None = None,
    max_frames: int | None = None,
) -> Dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    person_id = str(record["metadata"]["person_id"])
    cycles = record.get("cycles", [])
    if max_cycles is not None:
        cycles = cycles[:max_cycles]

    output_person_root = output_root / f"person_{person_id}"
    output_person_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        process_cycle(
            person_id=person_id,
            cycle=cycle,
            sam3d_root=sam3d_root,
            output_person_root=output_person_root,
            face_calib=face_calib,
            side_calib=side_calib,
            face_rt=rt_info[face_camera_id],
            side_rt=rt_info[side_camera_id],
            vis_stride=vis_stride,
            save_vis=save_vis,
            make_video=make_video,
            fps=fps,
            max_frames=max_frames,
        )
        for cycle in cycles
    ]

    person_summary = {
        "person_id": person_id,
        "alignment_record": str(record_path),
        "face_camera_id": face_camera_id,
        "side_camera_id": side_camera_id,
        "face_calibration": str(face_calib["path"]),
        "side_calibration": str(side_calib["path"]),
        "cycles": summaries,
    }
    save_frame_json(output_person_root / "summary.json", person_summary)
    return person_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triangulate SAM3D-Body 2D keypoints using split_cycle alignment."
    )
    parser.add_argument("--config", default="configs/sam3d_triangulation.yaml")
    parser.add_argument("--person", nargs="*", help="Optional person ids, e.g. 1 2 3")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    sam3d_root = Path(cfg["paths"]["sam3d_person_root"])
    split_root = Path(cfg["paths"]["split_cycle_root"])
    output_root = Path(cfg["paths"]["output_root"])
    calibration_cfg = cfg.get("calibration", {})
    if calibration_cfg:
        face_calib = load_calibration(Path(calibration_cfg["face"]))
        side_calib = load_calibration(Path(calibration_cfg["side"]))
        K_for_camera_vis = face_calib["K"]
        image_size = face_calib["image_size"].tolist()
    else:
        K_for_camera_vis = np.asarray(cfg["camera_K"]["K"], dtype=np.float32).reshape(3, 3)
        dist = np.asarray(
            cfg["camera_K"].get("distortion_coefficients", np.zeros(5)),
            dtype=np.float32,
        ).reshape(-1)
        face_calib = {"path": np.asarray("camera_K"), "K": K_for_camera_vis, "dist": dist}
        side_calib = {"path": np.asarray("camera_K"), "K": K_for_camera_vis, "dist": dist}
        image_size = cfg["camera_K"]["image_size"]

    camera_output = output_root / "_camera"
    cam = prepare_camera_position(
        K=K_for_camera_vis,
        yaws=cfg["camera_position"]["yaws"],
        T=cfg["camera_position"]["T"],
        r=cfg["camera_position"]["r"],
        z=cfg["camera_position"]["z"],
        output_path=str(camera_output),
        img_size=image_size,
    )

    wanted = set(str(p) for p in args.person) if args.person else None
    records = sorted(split_root.glob("person_*/alignment_record_*.json"))
    if wanted is not None:
        records = [
            p
            for p in records
            if p.parent.name.removeprefix("person_") in wanted
        ]

    summaries = []
    for record_path in records:
        print(f"[INFO] Processing {record_path}")
        summaries.append(
            process_person(
                record_path=record_path,
                sam3d_root=sam3d_root,
                output_root=output_root,
                face_calib=face_calib,
                side_calib=side_calib,
                rt_info=cam["rt_info"],
                face_camera_id=int(cfg["view_camera"]["face"]),
                side_camera_id=int(cfg["view_camera"]["side"]),
                vis_stride=int(cfg["visualization"]["vis_stride"]),
                save_vis=bool(cfg["visualization"]["save_frames"]) and not args.no_vis,
                make_video=bool(cfg["visualization"]["make_video"]) and not args.no_video,
                fps=int(cfg["visualization"]["fps"]),
                max_cycles=args.max_cycles,
                max_frames=args.max_frames,
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    save_frame_json(
        output_root / "summary.json",
        {
            "sam3d_person_root": str(sam3d_root),
            "split_cycle_root": str(split_root),
            "output_root": str(output_root),
            "face_calibration": str(face_calib["path"]),
            "side_calibration": str(side_calib["path"]),
            "num_persons": len(summaries),
            "persons": summaries,
        },
    )
    print(f"[DONE] Saved triangulated results to {output_root}")


if __name__ == "__main__":
    main()
