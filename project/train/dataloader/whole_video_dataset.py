#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from project.train.map_config import PersonInfo
from project.train.dataloader.utils import uniform_temporal_subsample

logger = logging.getLogger(__name__)


class LabeledPersonDataset(Dataset):
    """
    Multi-view labeled video dataset.
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: List[PersonInfo],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        load_frames: bool = True,
        load_2d_kpt: bool = True,
        load_3d_kpt: bool = True,
        temporal_subsample_num_samples: int = 32,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._transform = transform
        self._load_frames = bool(load_frames)
        self._load_2d_kpt = bool(load_2d_kpt)
        self._load_3d_kpt = bool(load_3d_kpt)
        self._temporal_subsample_num_samples = int(temporal_subsample_num_samples)
        if not self._load_frames and not self._load_2d_kpt and not self._load_3d_kpt:
            raise ValueError(
                "At least one of load_frames/load_2d_kpt/load_3d_kpt must be enabled."
            )
        self._source_index_cache: Dict[int, List[int]] = {}

    def __len__(self) -> int:
        return len(self._index_mapping)

    @staticmethod
    def _load_fused_kpt_sequence(
        fused_kpt_dir: Path,
        frame_start: int,
        frame_end: int,
    ) -> torch.Tensor:
        """Load fused 3D keypoints from per-frame npz files and stack into (T,J,3)."""
        if not fused_kpt_dir.exists() or not fused_kpt_dir.is_dir():
            raise FileNotFoundError(f"Fused kpt directory not found: {fused_kpt_dir}")

        frame_files = sorted(fused_kpt_dir.glob("frame_*.npz"))
        if not frame_files:
            raise FileNotFoundError(f"No fused npz files found in: {fused_kpt_dir}")

        start = 0 if frame_start is None or frame_start < 0 else int(frame_start)
        end = len(frame_files) if frame_end is None or frame_end < 0 else int(frame_end)
        start = max(0, min(start, len(frame_files)))
        end = max(start, min(end, len(frame_files)))

        kpts_list: List[np.ndarray] = []
        for t in range(start, end):
            frame_path = fused_kpt_dir / f"frame_{t:06d}.npz"
            if not frame_path.exists():
                raise FileNotFoundError(f"Missing fused frame file: {frame_path}")

            data = np.load(frame_path, allow_pickle=True)
            kpts = data["kpts_body"] if "kpts_body" in data.files else None

            kpts_list.append(np.asarray(kpts, dtype=np.float32))

        if not kpts_list:
            raise ValueError(
                f"Empty fused kpt sequence after slicing [{start}, {end}) in {fused_kpt_dir}"
            )

        return torch.from_numpy(np.stack(kpts_list, axis=0)).float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # index with index from index_mapping dict
        raw_item = self._index_mapping[index]

        sample: Dict[str, Any] = {
            "person_id": raw_item.person_id,
            "turn_id": raw_item.turn_id,
            "labels": {
                "twist": int(raw_item.label_twist_3class),
                "posture": int(raw_item.label_posture_3class),
                "relax": int(raw_item.label_relax_3class),
                "total": int(raw_item.label_total_3class),
            },
        }

        # 按照 turn 的 start/end 加载 fused 3D kpt，并堆成 tensor
        if self._load_3d_kpt:
            fused_3d_kpt_dir = Path(raw_item.fused_kpt_path)
            fused_turn_start = int(raw_item.fused_kpt_turn_frame_start)
            fused_turn_end = int(raw_item.fused_kpt_turn_frame_end)

            fused_3d_kpt = self._load_fused_kpt_sequence(
                fused_kpt_dir=fused_3d_kpt_dir,
                frame_start=fused_turn_start,
                frame_end=fused_turn_end,
            )  # (T,J,3)

            sample["fused_turn_frame_start"] = fused_turn_start
            sample["fused_turn_frame_end"] = fused_turn_end

            # 抽帧
            transformed_fused_3d_kpt = uniform_temporal_subsample(
                fused_3d_kpt, num_samples=self._temporal_subsample_num_samples, dim=0
            )
            sample["fused_kpt3d"] = transformed_fused_3d_kpt

        return sample


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[PersonInfo],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    load_frames: bool = True,
    load_2d_kpt: bool = True,
    load_3d_kpt: bool = True,
    temporal_subsample_num_samples: int = 32,
) -> LabeledPersonDataset:
    return LabeledPersonDataset(
        experiment=experiment,
        transform=transform,
        index_mapping=dataset_idx,
        load_frames=load_frames,
        load_2d_kpt=load_2d_kpt,
        load_3d_kpt=load_3d_kpt,
        temporal_subsample_num_samples=temporal_subsample_num_samples,
    )
