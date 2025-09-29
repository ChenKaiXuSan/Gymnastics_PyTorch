#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/kpt_generation/camera_position.py
Project: /workspace/code/kpt_generation
Created Date: Tuesday August 5th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday August 5th 2025 10:11:30 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import numpy as np
import cv2
import torch
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- 姿态估计 ----------
def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask_pose


def to_gray_cv_image(tensor_img):
    """
    将 torch 格式的 HWC 图像（0~1 或 0~255）转换为 OpenCV 灰度图像（uint8）

    参数:
        tensor_img: torch.Tensor, shape=(H, W, C), dtype=torch.float32 或 torch.uint8

    返回:
        img_gray: np.ndarray, shape=(H, W), dtype=uint8，适用于 OpenCV
    """
    if not isinstance(tensor_img, torch.Tensor):
        tensor_img = torch.tensor(tensor_img)
        
    if tensor_img.dim() != 3 or tensor_img.shape[2] != 3:
        raise ValueError("输入必须是 (H, W, 3) 的 RGB 图像")

    # 如果是 float，转换到 [0, 255]
    if tensor_img.dtype == torch.float32:
        img_np = tensor_img.clamp(0, 1).mul(255).byte().cpu().numpy()
    elif tensor_img.dtype == torch.uint8:
        img_np = tensor_img.cpu().numpy()
    else:
        raise TypeError("只支持 float32 或 uint8 类型的输入")

    # 转灰度（OpenCV 格式）
    import cv2

    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return img_gray


def estimate_camera_pose_from_sift_imgs(img1, img2, K, ratio_thresh=0.75):
    """
    使用 SIFT 特征从两幅图像中估计相机相对姿态（R, T）

    参数:
        img1: np.ndarray, 图像1（灰度图）
        img2: np.ndarray, 图像2（灰度图）
        K: np.ndarray, 相机内参矩阵 (3x3)
        ratio_thresh: float, Lowe's ratio 阈值（默认0.75）

    返回:
        R: np.ndarray, 旋转矩阵 (3x3)
        T: np.ndarray, 平移向量 (3x1)
        pts1: np.ndarray, 匹配点（图像1） Nx2
        pts2: np.ndarray, 匹配点（图像2） Nx2
        mask_pose: np.ndarray, 有效匹配掩码
    """

    if img1 is None or img2 is None or len(img1.shape) != 2 or len(img2.shape) != 2:
        raise ValueError("输入图像必须是灰度图，并且不能为空")

    # Step 1: 提取 SIFT 特征
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Step 2: 匹配特征
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Step 3: Lowe’s ratio test
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < 8:
        raise ValueError(f"匹配点太少（仅有{len(pts1)}个），无法估计姿态。")

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    # Step 4: 估计本质矩阵与姿态
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    if E is None:
        raise ValueError("Essential matrix 计算失败。")

    _, R, T, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    return R, T, pts1, pts2, mask_pose


def save_sift_keypoints_image(img_gray, keypoints, save_path, figsize=(10, 8)):
    """
    可视化并保存 SIFT 关键点图像

    参数:
        img_gray: np.ndarray, 灰度图像
        keypoints: list of cv2.KeyPoint, SIFT 提取的关键点
        save_path: str, 要保存的文件路径（如 'keypoints.jpg'）
        figsize: tuple, 图像大小（用于 matplotlib）
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img_kp = cv2.drawKeypoints(
        img_gray,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )

    plt.figure(figsize=figsize)
    plt.imshow(img_kp, cmap="gray")
    plt.title("SIFT Keypoints")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_sift_matches_image(
    img1, img2, kp1, kp2, matches, save_path, top_n=50, figsize=(16, 8)
):
    """
    可视化并保存 SIFT 匹配结果图像

    参数:
        img1, img2: np.ndarray, 原始图像（可为彩色或灰度）
        kp1, kp2: list of cv2.KeyPoint, 图像1和图像2的关键点
        matches: list of cv2.DMatch, 过滤后的匹配结果（如 ratio test 后）
        save_path: str, 要保存的文件路径（如 'matches.jpg'）
        top_n: int, 显示前多少个匹配点
        figsize: tuple, 图像大小
    """
    img_match = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:top_n],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=figsize)
    plt.imshow(img_match)
    plt.title(f"SIFT Matches (Top {top_n})")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def visualize_SIFT_matches(img1, img2, kp1, kp2, save_path, top_n=50, figsize=(16, 8)):
    """
    可视化并保存 SIFT 匹配结果图像

    参数:
        img1, img2: np.ndarray, 原始图像（可为彩色或灰度）
        kp1, kp2: list of cv2.KeyPoint, 图像1和图像2的关键点
        matches: list of cv2.DMatch, 过滤后的匹配结果（如 ratio test 后）
        save_path: str, 要保存的文件路径（如 'matches.jpg'）
        top_n: int, 显示前多少个匹配点
        figsize: tuple, 图像大小
    """

    # SIFT 特征提取
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 保存关键点图
    # save_sift_keypoints_image(img1, kp1, save_path)
    # save_sift_keypoints_image(img2, kp2, save_path)

    # 匹配与 ratio test
    bf = cv2.BFMatcher()
    matches_all = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches_all:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 保存匹配图
    save_sift_matches_image(img1, img2, kp1, kp2, good_matches, save_path)
