#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Camera calibration script (refactored & optimized)
- Auto-detect input: directory of images or a video file
- Robust evaluation metrics & optional auto-prune of bad samples
- Saves both .npz and OpenCV .yml, plus per-image error CSV and visualizations

Author: Kaixu Chen <chenkaixusan@gmail.com>
Refactor: ChatGPT (optimization, structure, metrics, CLI)
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

# ---------------------- Logging ----------------------
LOG = logging.getLogger("calib")


def setup_logging(verbosity: int = 1) -> None:
    level = (
        logging.WARNING
        if verbosity <= 0
        else (logging.INFO if verbosity == 1 else logging.DEBUG)
    )
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=level)


# ---------------------- Config ----------------------
@dataclass
class CalibConfig:
    board_cols: int = 9  # inner corners along columns
    board_rows: int = 6  # inner corners along rows
    square_size: float = 25.0  # same unit across all images
    rational_model: bool = True  # enable k4..k6
    zero_tangent: bool = False  # set True if you know p1,p2≈0
    fast_check: bool = False  # speed-up chessboard detection
    rotate_deg: int = 0  # rotate frames/images by deg (0/90/180/270)
    prune_top_ratio: float = 0.1  # auto-drop worst x% images once; set 0 to disable
    output_dir: str = "logs/calibration_vis"

    @property
    def board_size(self) -> Tuple[int, int]:
        return (self.board_cols, self.board_rows)


# ---------------------- Utilities ----------------------
VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".MP4",
    ".MOV",
    ".AVI",
    ".MKV",
    ".M4V",
}
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")


def is_video_file(path: str | os.PathLike) -> bool:
    p = Path(path)
    return p.is_file() and (p.suffix in VIDEO_EXTS)


def is_image_dir(path: str | os.PathLike) -> bool:
    return Path(path).is_dir()


def is_video_folder(path: str | os.PathLike, recursive: bool = True) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False

    it = p.rglob("*") if recursive else p.iterdir()
    for fp in it:
        # 某些路径可能因权限报错，忽略继续
        try:
            if fp.is_file() and fp.suffix.lower() in VIDEO_EXTS:
                return True
        except OSError:
            continue
    return False


def prepare_object_points(
    board_size: Tuple[int, int], square_size: float
) -> np.ndarray:
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(square_size)


def find_chessboard_corners(
    image: np.ndarray, board_size: Tuple[int, int], *, fast_check: bool = False
) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast_check:
        flags |= cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not ret:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners_refined


def save_visualization(
    image: np.ndarray,
    corners: np.ndarray,
    board_size: Tuple[int, int],
    save_path: str | os.PathLike,
) -> None:
    img_vis = cv2.drawChessboardCorners(image.copy(), board_size, corners, True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_vis)


def save_undistortion_comparison(
    original_img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    save_path: str | os.PathLike,
) -> None:
    undistorted = cv2.undistort(original_img, camera_matrix, dist_coeffs)
    concat = np.hstack((original_img, undistorted))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), concat)


def rotate_image(img: np.ndarray, deg: int) -> np.ndarray:
    """Rotate by multiples of 90 quickly; arbitrary angle via warp (keeps size)."""
    d = deg % 360
    if d == 0:
        return img
    if d == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # arbitrary angle: rotate around center, keep canvas size
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), d, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def load_images_from_dir(
    image_dir: str | os.PathLike, patterns: Sequence[str] = IMG_EXTS
) -> List[str]:
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(str(image_dir), pat)))
    return sorted(paths)


def sample_frames_from_video(
    video_path: str | os.PathLike,
    rotate_deg: int = 0,  # <-- 新增：抽帧阶段旋转
) -> List[Tuple[str, np.ndarray]]:
    """Sample frames from a video. If rotate_deg!=0, rotate each sampled frame here."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[Tuple[str, np.ndarray]] = []
    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        if rotate_deg % 360 != 0:
            frame = rotate_image(frame, rotate_deg)

        frames.append(frame.copy())

    cap.release()

    return frames


# ---------------------- Evaluation helpers ----------------------


def _as_xy(arr: np.ndarray) -> np.ndarray:
    """Return Nx2 view regardless of coming as (N,2) or (N,1,2)."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[1] == 1 and a.shape[2] == 2:
        return a.reshape(-1, 2)
    return a.reshape(-1, 2)


def compute_reproj_stats(
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    objpoints: Sequence[np.ndarray],
    imgpoints: Sequence[np.ndarray],
) -> Tuple[List[dict], dict]:
    per_img: List[dict] = []
    all_errs: List[np.ndarray] = []
    for obj, img, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        per_img.append(
            {
                "mean_px": float(e.mean()),
                "median_px": float(np.median(e)),
                "p95_px": float(np.percentile(e, 95)),
            }
        )
        all_errs.append(e)
    if all_errs:
        cat = np.concatenate(all_errs)
        global_stat = {
            "global_mean_px": float(cat.mean()),
            "global_median_px": float(np.median(cat)),
            "global_p95_px": float(np.percentile(cat, 95)),
        }
    else:
        global_stat = {
            "global_mean_px": None,
            "global_median_px": None,
            "global_p95_px": None,
        }
    return per_img, global_stat


def compute_edge_center_ratio(
    imgpoints: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    border_ratio: float = 0.2,
) -> dict:
    """Robust coverage stats that accept points as (N,2) or (N,1,2)."""
    w, h = image_size
    xs, ys = [], []
    for ip in imgpoints:
        xy = _as_xy(ip)
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
    if not xs:
        return {"coverage_ratio": 0.0, "edge_points_ratio": 0.0}
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull) if hull is not None else 0.0
    coverage_ratio = float(hull_area / (w * h))
    x0, x1 = w * border_ratio, w * (1 - border_ratio)
    y0, y1 = h * border_ratio, h * (1 - border_ratio)
    at_edge = ((xs < x0) | (xs > x1) | (ys < y0) | (ys > y1)).mean()
    return {"coverage_ratio": coverage_ratio, "edge_points_ratio": float(at_edge)}


def compute_fov_and_principal(K: np.ndarray, image_size: Tuple[int, int]) -> dict:
    w, h = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    hfov = float(2 * np.degrees(np.arctan(w / (2 * fx))))
    vfov = float(2 * np.degrees(np.arctan(h / (2 * fy))))
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "hfov_deg": hfov,
        "vfov_deg": vfov,
        "principal_point_offset_px": (cx - w / 2, cy - h / 2),
        "aspect_fx_fy": (fx / fy) if fy else None,
    }


def line_straightness_on_corners(
    imgpoints: Sequence[np.ndarray],
    board_size: Tuple[int, int],
    K: np.ndarray,
    dist: np.ndarray,
) -> dict:
    cols, rows = board_size
    pts_concat = np.concatenate(imgpoints, axis=0).reshape(-1, 1, 2).astype(np.float32)
    pts_ud = cv2.undistortPoints(pts_concat, K, dist, P=K).reshape(-1, 2)

    def rms_line_fit(pts_2d: np.ndarray, cols: int, rows: int) -> float:
        pts = pts_2d.reshape(-1, rows * cols, 2)
        errs = []
        for P in pts:
            for r in range(rows):
                row_pts = P[r * cols : (r + 1) * cols]
                x, y = row_pts[:, 0], row_pts[:, 1]
                A = np.c_[x, np.ones_like(x)]
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                d = np.abs(m * x - y + c) / np.sqrt(m * m + 1)
                errs.append(np.mean(d * d))
            for cidx in range(cols):
                col_pts = P[cidx::cols]
                x, y = col_pts[:, 0], col_pts[:, 1]
                A = np.c_[x, np.ones_like(x)]
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                d = np.abs(m * x - y + c) / np.sqrt(m * m + 1)
                errs.append(np.mean(d * d))
        return float(np.sqrt(np.mean(errs))) if errs else float("nan")

    before = rms_line_fit(pts_concat.reshape(-1, 2), cols, rows)
    after = rms_line_fit(pts_ud, cols, rows)
    return {"straightness_rms_before_px": before, "straightness_rms_after_px": after}


def valid_roi_fraction(
    K: np.ndarray, dist: np.ndarray, image_size: Tuple[int, int], alpha: float = 0.0
) -> dict:
    w, h = image_size
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    x, y, ww, hh = roi
    frac = float((ww * hh) / (w * h)) if ww > 0 and hh > 0 else 0.0
    return {"valid_roi_fraction_alpha0": frac, "new_camera_matrix": newK}


# ---------------------- Core calibration ----------------------


def calibrate_camera(
    images_or_frames: Sequence[np.ndarray], cfg: CalibConfig, file_name: str
) -> dict:
    out_dir = Path(cfg.output_dir) / file_name
    out_dir.mkdir(parents=True, exist_ok=True)
    corners_dir = out_dir / "corners"
    corners_dir.mkdir(parents=True, exist_ok=True)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    used_names: List[str] = []
    used_names_full: List[str] | None = None
    objp = prepare_object_points(cfg.board_size, cfg.square_size)

    LOG.info("Detecting chessboard corners (%dx%d)...", cfg.board_cols, cfg.board_rows)
    for i, img in enumerate(images_or_frames):
        name = file_name
        if img is None:
            LOG.warning("Empty image: %s", name)
            continue
        img_r = img
        corners = find_chessboard_corners(
            img_r, cfg.board_size, fast_check=cfg.fast_check
        )
        if corners is None:
            LOG.debug("Chessboard not detected: %s", name)
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used_names.append(str(i))

        # 保存检测出来的frame
        save_visualization(
            img_r, corners, cfg.board_size, corners_dir / f"corners_{i+1:04d}.jpg"
        )

    # Keep a copy of full list before pruning
    used_names_full = list(used_names)

    if not objpoints:
        return {"status": "Failed", "reason": "No corners detected", "num_images": 0}

    # determine image size after rotation
    h, w = images_or_frames[0][1].shape[:2]

    flags = 0
    if cfg.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if cfg.zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    LOG.info("Running calibrateCamera with flags=%s", flags)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None, flags=flags
    )

    # Evaluation metrics
    per_img_stats, global_stats = compute_reproj_stats(
        K, dist, rvecs, tvecs, objpoints, imgpoints
    )
    cov_stats = compute_edge_center_ratio(imgpoints, (w, h), border_ratio=0.2)
    param_stats = compute_fov_and_principal(K, (w, h))
    straight_stats = line_straightness_on_corners(imgpoints, cfg.board_size, K, dist)
    roi_stats = valid_roi_fraction(K, dist, (w, h), alpha=0.0)

    # Optionally prune worst X% images by mean reproj error and re-calibrate once
    dropped = []
    if cfg.prune_top_ratio > 0 and len(per_img_stats) >= 20:
        means = np.array([s["mean_px"] for s in per_img_stats])
        thresh = np.quantile(means, 1 - cfg.prune_top_ratio)
        keep_idx = np.where(means <= thresh)[0]
        drop_idx = np.where(means > thresh)[0]
        if drop_idx.size > 0 and keep_idx.size >= 10:
            LOG.info(
                "Pruning top %.1f%% worst images (%d dropped) and recalibrating...",
                cfg.prune_top_ratio * 100,
                drop_idx.size,
            )
            dropped = [used_names[i] for i in drop_idx.tolist()]
            objpoints2 = [objpoints[i] for i in keep_idx]
            imgpoints2 = [imgpoints[i] for i in keep_idx]
            kept_names = [used_names[i] for i in keep_idx]
            prev_ret = ret
            ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
                objpoints2, imgpoints2, (w, h), None, None, flags=flags
            )
            # Accept if RMS improved
            if ret2 < prev_ret:
                ret, K, dist, rvecs, tvecs = ret2, K2, dist2, rvecs2, tvecs2
                objpoints, imgpoints, used_names = objpoints2, imgpoints2, kept_names
                per_img_stats, global_stats = compute_reproj_stats(
                    K, dist, rvecs, tvecs, objpoints, imgpoints
                )
                cov_stats = compute_edge_center_ratio(
                    imgpoints, (w, h), border_ratio=0.2
                )
                straight_stats = line_straightness_on_corners(
                    imgpoints, cfg.board_size, K, dist
                )
                LOG.info(
                    "RMS improved from %.4f to %.4f after pruning.", prev_ret, ret2
                )

    # Save parameters (.npz + .yml)
    output_npz = out_dir / "calibration_parameters.npz"
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_npz,
        camera_matrix=K,
        dist_coeffs=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=(w, h),
        used=used_names,
        used_full=used_names_full,
        dropped=dropped,
        config=asdict(cfg),
    )

    try:
        output_yml = out_dir / "calibration_parameters.yaml"
        fs = cv2.FileStorage(output_yml, cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", int(w))
        fs.write("image_height", int(h))
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", dist)
        fs.write("flags", int(flags))
        fs.release()
    except Exception as e:
        LOG.warning("Failed to write YML: %s", e)

    # Save per-image errors CSV
    per_img_df = pd.DataFrame(per_img_stats)
    names_for_csv = (
        used_names
        if len(used_names) == len(per_img_df)
        else (used_names[: len(per_img_df)] if used_names else [])
    )
    per_img_df["image_name"] = names_for_csv
    per_img_csv = str(Path(cfg.output_dir) / "per_image_errors.csv")
    per_img_df.to_csv(per_img_csv, index=False)

    # Undistortion previews (limit number for speed)
    vis_limit = min(len(used_names), 30)
    for i, (name, img) in enumerate(images_or_frames[:vis_limit]):
        if name in used_names[:vis_limit]:
            img_r = rotate_image(img, cfg.rotate_deg)
            save_undistortion_comparison(
                img_r, K, dist, Path(cfg.output_dir) / f"undistort_{i+1:03d}.jpg"
            )

    result = {
        "status": "Success" if ret else "Failed",
        "rms_reproj_error": float(ret),
        "camera_matrix": K,
        "dist_coeffs": dist,
        "num_images": len(used_names),
        "image_size": (w, h),
        "used_names": used_names,
        "dropped_names": dropped,
        "per_image_csv": per_img_csv,
        **global_stats,
        **cov_stats,
        **param_stats,
        **straight_stats,
        **roi_stats,
    }
    return result


def load_images_from_dir_recursive(
    image_dir: str | os.PathLike, patterns: Sequence[str] = IMG_EXTS
) -> List[str]:
    """递归搜集目录及子目录中的所有图片路径"""
    paths: List[str] = []
    base = str(image_dir)
    for pat in patterns:
        # ** 匹配子目录
        paths.extend(glob.glob(os.path.join(base, "**", pat), recursive=True))
    return sorted(paths)


def list_videos_in_dir(
    dir_path: str | os.PathLike, recursive: bool = True
) -> List[str]:
    """（递归）列出目录中的所有视频文件"""
    base = str(dir_path)
    vids: List[str] = []
    for ext in VIDEO_EXTS:
        pattern = (
            os.path.join(base, "**", f"*{ext}")
            if recursive
            else os.path.join(base, f"*{ext}")
        )
        vids.extend(glob.glob(pattern, recursive=recursive))
    return sorted(vids)


def calibrate_from_input(input_path: str | os.PathLike, cfg: CalibConfig) -> dict:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if is_image_dir(input_path):
        # 先尝试当前目录非递归搜图
        img_paths = load_images_from_dir(input_path)
        # 若为空，改用递归（支持子文件夹）
        if not img_paths:
            img_paths = load_images_from_dir_recursive(input_path)

        if img_paths:
            images = []
            for p in img_paths:
                im = cv2.imread(p)
                if im is None:
                    LOG.warning("Cannot read image: %s", p)
                    continue
                images.append((p, im))
            return calibrate_camera(images, cfg)

        # 没有图片？那就递归找视频
        video_paths = list_videos_in_dir(input_path, recursive=True)
        if video_paths:
            for vp in video_paths:
                vname = os.path.splitext(os.path.basename(vp))[0]
                frames = sample_frames_from_video(
                    video_path=vp,
                    rotate_deg=cfg.rotate_deg,
                )

                return calibrate_camera(frames, cfg, file_name=vname)

        # 既没图也没视频
        raise RuntimeError(f"No images or videos found under: {input_path}")

    else:
        raise RuntimeError(
            f"Invalid input_path. Not a directory or a known video file: {input_path}"
        )


# ---------------------- CLI ----------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera calibration (images or video)")
    p.add_argument(
        "--input",
        type=str,
        default="/workspace/code/camera_calibration/input_video",
        help="Directory of images or a video file path",
    )
    p.add_argument(
        "--board", type=str, default="9x6", help="Inner corners as CxR, e.g., 9x6"
    )
    p.add_argument(
        "--square", type=float, default=25.0, help="Square size in your length unit"
    )
    p.add_argument("--out_dir", type=str, default="logs/calibration_vis")

    p.add_argument("--fast_check", action="store_true")
    p.add_argument(
        "--no_rational", action="store_true", help="Disable rational distortion model"
    )
    p.add_argument("--zero_tangent", action="store_true")
    p.add_argument(
        "--prune",
        type=float,
        default=0.1,
        help="Drop top x ratio of worst images; 0=disable",
    )
    p.add_argument(
        "--rotate_deg",
        type=int,
        default=90,
        help="Rotate frames/images by deg (0/90/180/270)",
    )
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    cols, rows = map(int, args.board.lower().split("x"))
    cfg = CalibConfig(
        board_cols=cols,
        board_rows=rows,
        square_size=args.square,
        rational_model=not args.no_rational,
        zero_tangent=args.zero_tangent,
        fast_check=args.fast_check,
        prune_top_ratio=max(0.0, min(0.9, args.prune)),
        output_dir=args.out_dir,
        rotate_deg=args.rotate_deg,
    )
    LOG.info("Config: %s", cfg)

    result = calibrate_from_input(args.input, cfg)

    # Pretty print summary
    summary = pd.DataFrame(
        [
            {
                "status": result.get("status"),
                "rms_reproj_error": result.get("rms_reproj_error"),
                "global_mean_px": result.get("global_mean_px"),
                "global_p95_px": result.get("global_p95_px"),
                "coverage_ratio": result.get("coverage_ratio"),
                "edge_points_ratio": result.get("edge_points_ratio"),
                "hfov_deg": result.get("hfov_deg"),
                "vfov_deg": result.get("vfov_deg"),
                "pp_off_x": (
                    result.get("principal_point_offset_px")[0]
                    if result.get("principal_point_offset_px")
                    else None
                ),
                "pp_off_y": (
                    result.get("principal_point_offset_px")[1]
                    if result.get("principal_point_offset_px")
                    else None
                ),
                "straight_before_px": result.get("straightness_rms_before_px"),
                "straight_after_px": result.get("straightness_rms_after_px"),
                "valid_roi_frac@alpha0": result.get("valid_roi_fraction_alpha0"),
                "num_used_images": result.get("num_images"),
                "image_size": result.get("image_size"),
                "dropped": len(result.get("dropped_names", [])),
            }
        ]
    )
    print("\n📌 Calibration Result Summary:")
    print(summary)
    print(f"Per-image error CSV: {result.get('per_image_csv')}")
    print(f"Params saved: {cfg.output_npz} , {cfg.output_yml}")


if __name__ == "__main__":
    main()
