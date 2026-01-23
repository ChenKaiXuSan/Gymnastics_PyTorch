#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/utils.py
Project: /workspace/code/sam3d_body
Created Date: Thursday December 4th 2025
Author: Kaixu Chen
-----
Comment:
Visualization utilities for SAM3Dbody results.

Have a good code time :)
-----
Last Modified: Thursday December 4th 2025 4:24:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import json
import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


from .sam_3d_body.visualization.renderer import Renderer
from .sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from .tools.vis_utils import visualize_sample, visualize_sample_together

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        # Draw bounding box
        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2,
        )

        # Add person ID text
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    results = []

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # 1. Original image
        img_orig = img_cv2.copy()

        # 2. Mesh overlay on original image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 3. Mesh on white background (front view)
        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 4. Side view
        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        # Combine all views
        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def vis_results(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
    visualizer: SkeletonVisualizer,
):
    """Save 3D mesh results to files and return PLY file paths"""

    os.makedirs(save_dir, exist_ok=True)
    ply_files = []

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Save focal length
    if outputs:
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        logger.info(f"Saved focal length: {focal_length_path}")

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # Store individual mesh
        tmesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
        )
        mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
        mesh_path = os.path.join(save_dir, mesh_filename)
        tmesh.export(mesh_path)
        ply_files.append(mesh_path)

        # Save individual overlay image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)

        # Save bbox image
        img_bbox = img_cv2.copy()
        bbox = person_output["bbox"]
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            4,
        )
        bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)

        logger.info(f"Saved mesh: {mesh_path}")
        logger.info(f"Saved overlay: {os.path.join(save_dir, overlay_filename)}")
        logger.info(f"Saved bbox: {os.path.join(save_dir, bbox_filename)}")

        # 2D 结果可视化
        vis_results = visualize_2d_results(img_cv2, outputs, visualizer)
        cv2.imwrite(
            os.path.join(save_dir, f"{image_name}_2d_visualization.png"),
            vis_results[pid],
        )
        logger.info(
            f"Saved 2D visualization: {os.path.join(save_dir, f'{image_name}_2d_visualization.png')}"
        )

        # 3D 网格可视化
        mesh_results = visualize_3d_mesh(img_cv2, outputs, faces)
        # Display results

        cv2.imwrite(
            os.path.join(save_dir, f"{image_name}_3d_mesh_visualization_{pid}.png"),
            mesh_results[pid],
        )

        logger.info(
            f"Saved 3D mesh visualization: {os.path.join(save_dir, f'{image_name}_3d_mesh_visualization_{pid}.png')}"
        )

        # 综合可视化
        together_img = visualize_sample_together(
            img_cv2=img_cv2,
            outputs=outputs,
            faces=faces,
        )
        cv2.imwrite(
            os.path.join(save_dir, f"{image_name}_together_visualization.png"),
            together_img,
        )
        logger.info(
            f"Saved together visualization: {os.path.join(save_dir, f'{image_name}_together_visualization.png')}"
        )
