#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Camera calibration script for multiple videos using 9x6 chessboard images.

Author: Kaixu Chen <chenkaixusan@gmail.com>
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd


def extract_frames_from_video(video_path, output_dir, step=1, rotate_cw=False):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    idx, kept = -1, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % step != 0:
            continue
        if rotate_cw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 向右旋转90°
        cv2.imwrite(os.path.join(output_dir, f"frame_{idx:06d}.jpg"), frame)
        kept += 1
    cap.release()
    return kept


def extract_frames_from_multiple_videos(
    video_paths, base_output_dir, step=1, rotate_cw=False
):
    frame_dirs = []
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(base_output_dir, f"{video_name}_frames")
        extract_frames_from_video(video_path, output_dir, step, rotate_cw=rotate_cw)
        frame_dirs.append(output_dir)
    return frame_dirs


def prepare_object_points(board_size, square_size):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    return objp * square_size


def find_chessboard_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not ret:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def save_visualization(image, corners, board_size, save_path):
    vis = cv2.drawChessboardCorners(image.copy(), board_size, corners, True)
    cv2.imwrite(save_path, vis)


def save_undistortion_comparison(image, camera_matrix, dist_coeffs, save_path):
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    combined = np.hstack((image, undistorted))
    cv2.imwrite(save_path, combined)


def calibrate_camera_from_images(
    image_dir,
    board_size,
    square_size,
    output_file,
    vis_dir,
):
    os.makedirs(vis_dir, exist_ok=True)
    objp = prepare_object_points(board_size, square_size)
    objpoints, imgpoints, valid_images = [], [], []

    images = sorted(glob.glob(os.path.join(image_dir, "*.JPG")))
    if not images:
        print(f"⚠️ No images found in: {image_dir}")
        return {"status": "Failed", "reason": "No images", "num_images": 0}

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            continue
        corners = find_chessboard_corners(img, board_size)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_images.append(fname)
            save_visualization(
                img, corners, board_size, os.path.join(vis_dir, f"corners_{i:04d}.jpg")
            )
        else:
            print(f"🔍 Chessboard not found: {os.path.basename(fname)}")

    if not objpoints:
        print(f"❌ Calibration failed: No corners detected in {image_dir}")
        return {"status": "Failed", "reason": "No corners detected", "num_images": 0}

    h, w = cv2.imread(valid_images[0]).shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=(w, h),
    )

    for i, fname in enumerate(valid_images):
        img = cv2.imread(fname)
        save_undistortion_comparison(
            img,
            camera_matrix,
            dist_coeffs,
            os.path.join(vis_dir, f"undistort_{i:04d}.jpg"),
        )

    return {
        "status": "Success",
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "num_images": len(valid_images),
        "image_size": (w, h),
        "output_file": output_file,
    }


if __name__ == "__main__":
    # === Calibration Settings ===
    CHESSBOARD_SIZE = (8, 5)
    SQUARE_SIZE_MM = 25.0
    STEP = 1  # Frame extraction step

    # === Directory Paths ===
    VIDEO_DIR = "camera_calibration/input_video"
    FRAME_BASE_DIR = "logs/camera_calibration/extracted_frames"
    OUTPUT_BASE_DIR = "logs/camera_calibration/calibration_results"

    # Ensure directories exist
    os.makedirs(FRAME_BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # === Find All Videos ===
    video_list = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.MOV")))
    if not video_list:
        raise FileNotFoundError(f"No videos found in: {VIDEO_DIR}")

    # === Process Each Video ===
    all_frame_dirs = extract_frames_from_multiple_videos(
        video_list, FRAME_BASE_DIR, step=STEP, rotate_cw=True
    )

    for frame_dir in all_frame_dirs:
        video_name = os.path.basename(frame_dir).replace("_frames", "")
        vis_dir = os.path.join(OUTPUT_BASE_DIR, f"{video_name}_vis")
        calib_file = os.path.join(OUTPUT_BASE_DIR, f"{video_name}_calibration.npz")

        result = calibrate_camera_from_images(
            image_dir=frame_dir,
            board_size=CHESSBOARD_SIZE,
            square_size=SQUARE_SIZE_MM,
            output_file=calib_file,
            vis_dir=vis_dir,
        )

        print(f"\n📌 Calibration Result for [{video_name}]")
        print(pd.DataFrame([result]))
