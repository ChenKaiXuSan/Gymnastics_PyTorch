#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from project.train.map_config import PersonInfo

logger = logging.getLogger(__name__)


class LabeledPersonDataset(Dataset):
    """
    Multi-view labeled video dataset.
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: PersonInfo,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        load_frames: bool = True,
        load_2d_kpt: bool = True,
        load_3d_kpt: bool = True,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = self._prepare_index_mapping(index_mapping)
        self._transform = transform
        self._load_frames = bool(load_frames)
        self._load_2d_kpt = bool(load_2d_kpt)
        self._load_3d_kpt = bool(load_3d_kpt)
        if not self._load_frames and not self._load_2d_kpt and not self._load_3d_kpt:
            raise ValueError(
                "At least one of load_frames/load_2d_kpt/load_3d_kpt must be enabled."
            )
        self._source_index_cache: Dict[int, List[int]] = {}

    def __len__(self) -> int:
        return len(self._index_mapping)

    @staticmethod
    def _prepare_index_mapping(index_mapping: PersonInfo) -> List[PersonInfo]:
        """Convert dict-based index mapping to list for indexing."""
        if isinstance(index_mapping, dict):
            # Sort by key to ensure consistent order
            sorted_items = sorted(index_mapping.items(), key=lambda x: x[0])
            return [item[1] for item in sorted_items]
        elif isinstance(index_mapping, list):
            return index_mapping
        else:
            raise TypeError(
                f"Expected index_mapping to be dict or list, got {type(index_mapping)}"
            )

    @staticmethod
    def _load_frames_dir(path: Path) -> torch.Tensor:
        """Load image sequence directory into (T,C,H,W)."""
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {path}")

        frame_files = sorted(path.glob("*.png"))
        if len(frame_files) == 0:
            frame_files = sorted(path.glob("*.jpg"))
        if len(frame_files) == 0:
            raise RuntimeError(f"No frame files found in: {path}")

        frames = []
        for p in frame_files:
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to read frame: {p}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1)
            )
        return torch.stack(frames, dim=0)

    @staticmethod
    def _load_sam3d_file(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load SAM3D 2D/3D keypoints from one npz file.

        Returns:
            (sam_2d, sam_3d)
        """
        data = np.load(npz_path, allow_pickle=True)
        if "output" not in data.files:
            raise KeyError(f"Missing 'output' in SAM npz: {npz_path}")
        output = data["output"]
        if isinstance(output, np.ndarray) and output.shape == ():
            output = output.item()

        if not isinstance(output, dict):
            raise TypeError(f"Unexpected SAM output type in {npz_path}: {type(output)}")

        if "pred_keypoints_3d" in output:
            arr_3d = output["pred_keypoints_3d"]
        elif "pred_joint_coords" in output:
            arr_3d = output["pred_joint_coords"]
        else:
            raise KeyError(
                f"No 3D keypoint key found in SAM output: {npz_path}, keys={list(output.keys())}"
            )

        if "pred_keypoints_2d" in output:
            arr_2d = output["pred_keypoints_2d"]
        else:
            # fallback: keep first 2 dims from 3d keypoints for compatibility
            arr_2d = np.asarray(arr_3d, dtype=np.float32)[..., :2]

        return np.asarray(arr_2d, dtype=np.float32), np.asarray(
            arr_3d, dtype=np.float32
        )

    def _apply_transform(self, video_tchw: torch.Tensor) -> torch.Tensor:
        """
        Apply transform on a segment.

        Expect transform: (T,C,H,W) -> (T,C,H,W) or compatible.
        """
        if self._transform is None:
            return video_tchw
        return self._transform(video_tchw)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # index with index from index_mapping dict
        raw_item = self._index_mapping[index]

        # load 3d kpt from given path
        return self._load_item(raw_item)


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[Any],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    load_frames: bool = True,
    load_2d_kpt: bool = True,
    load_3d_kpt: bool = True,
) -> LabeledPersonDataset:
    return LabeledPersonDataset(
        experiment=experiment,
        transform=transform,
        index_mapping=dataset_idx,
        load_frames=load_frames,
        load_2d_kpt=load_2d_kpt,
        load_3d_kpt=load_3d_kpt,
    )
