# -*- coding: utf-8 -*-
"""
estimate_camera_pose.py

通过图像特征点匹配、基础矩阵与本质矩阵，恢复两个摄像机之间的相对姿态（R, T）。

用法：
    替换图像路径和内参矩阵后运行：
    python estimate_camera_pose.py

作者：ChatGPT
"""

import cv2
import numpy as np

def estimate_camera_pose(img1_path, img2_path, K):
    # 加载图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("图像读取失败，请检查路径是否正确。")

    # ORB 特征检测与匹配
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("未检测到足够特征点。")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:
        raise RuntimeError("匹配点太少，无法估计基础矩阵。")

    # 提取匹配点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 计算基础矩阵 F（使用 RANSAC）
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None:
        raise RuntimeError("基础矩阵估计失败。")

    # 计算本质矩阵 E
    E = K.T @ F @ K

    # 归一化点坐标并恢复姿态
    pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
    _, R, T, _ = cv2.recoverPose(E, pts1_norm, pts2_norm)

    return F, E, R, T

if __name__ == "__main__":
    # 示例相机内参（请替换为你自己的相机内参）
    K = np.array([
        [1.53905292e+03, 0.00000000e+00, 8.67619960e+02],
        [0.00000000e+00, 1.54153123e+03, 5.72745232e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ], dtype=np.float32)

    # 示例图像路径（请替换为你自己的图像）
    img1_path = "left.jpg"
    img2_path = "right.jpg"

    try:
        F, E, R, T = estimate_camera_pose(img1_path, img2_path, K)
        print("Fundamental Matrix (F):\\n", F)
        print("Essential Matrix (E):\\n", E)
        print("Rotation Matrix (R):\\n", R)
        print("Translation Vector (T):\\n", T)
    except Exception as e:
        print("错误：", e)
