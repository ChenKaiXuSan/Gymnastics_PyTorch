#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: stereo_triangulation_from_video.py
Author: Kaixu Chen
Last Modified: 2025-08-04
Description: 对左右视角视频逐帧进行姿态估计与三角测量。
"""

import cv2
import numpy as np

# ---------- 相机内参 ----------
K = np.array([
    [1675.1430, 0.0, 880.9680],
    [0.0, 1286.3486, 1025.9397],
    [0.0, 0.0, 1.0]
], dtype=np.float32)


def load_videos(left_path, right_path):
    capL = cv2.VideoCapture(left_path)
    capR = cv2.VideoCapture(right_path)
    return capL, capR


def extract_features(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(gray, None)


def match_features(des1, des2, matcher, ratio=0.75):
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good


def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask_pose


def triangulate_points(R, t, pts1, pts2, K):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T  # (N, 3)
    return pts3D


def process_frame(frameL, frameR, detector, matcher, K):
    kpL, desL = extract_features(frameL, detector)
    kpR, desR = extract_features(frameR, detector)

    if desL is None or desR is None or len(kpL) < 8 or len(kpR) < 8:
        return None, "Insufficient features"

    good_matches = match_features(desL, desR, matcher)
    if len(good_matches) < 8:
        return None, "Not enough good matches"

    ptsL = np.float32([kpL[m.queryIdx].pt for m in good_matches])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in good_matches])

    R, t, mask_pose = estimate_pose(ptsL, ptsR, K)
    if R is None:
        return None, "Essential matrix failed"

    inliersL = ptsL[mask_pose.ravel() == 1]
    inliersR = ptsR[mask_pose.ravel() == 1]

    if len(inliersL) < 8:
        return None, "Not enough inliers"

    pts3D = triangulate_points(R, t, inliersL, inliersR, K)
    return pts3D, None


def main():
    # ---------- 视频输入路径 ----------
    video_left_path = "left.mp4"
    video_right_path = "right.mp4"

    capL, capR = load_videos(video_left_path, video_right_path)
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    frame_idx = 0
    triangulated_frames = []

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        pts3D, error = process_frame(frameL, frameR, sift, matcher, K)

        if pts3D is not None:
            triangulated_frames.append(pts3D)
            print(f"Frame {frame_idx}: {pts3D.shape[0]} points triangulated.")
        else:
            print(f"Frame {frame_idx}: {error}")

        frame_idx += 1

    capL.release()
    capR.release()
    print(f"\n总共处理了 {len(triangulated_frames)} 帧。")


if __name__ == "__main__":
    main()
