#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/MultiView_DriverAction_PyTorch/project/dataloader/data_loader.py
Project: /workspace/MultiView_DriverAction_PyTorch/project/dataloader
Created Date: Saturday January 24th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday January 24th 2026 10:51:04 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
)

from project.train.dataloader.whole_video_dataset import whole_video_dataset
from project.train.dataloader.utils import Div255


class PersonDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size
        self._load_frames = bool(getattr(opt.data, "load_frames", True))
        self._load_2d_kpt = bool(getattr(opt.data, "load_2d_kpt", True))
        self._load_3d_kpt = bool(getattr(opt.data, "load_3d_kpt", True))
        self._temporal_subsample_num_samples = int(
            getattr(opt.model, "temporal_kernel_size", 32)
        )
        if not self._load_frames and not self._load_2d_kpt and not self._load_3d_kpt:
            raise ValueError(
                "At least one of data.load_frames/data.load_2d_kpt/data.load_3d_kpt must be true."
            )

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self._experiment = opt.experiment

        self.mapping_transform = Compose(
            [
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
            ]
        )

    @staticmethod
    def _merge_bt_pose(x: torch.Tensor, name: str) -> torch.Tensor:
        """Merge sample/time dims: (1,T,J,C) or (T,J,C) -> (T,1,J,C)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected {name} shape (T,J,C), got {tuple(x.shape)}")
        return x.unsqueeze(1)

    @staticmethod
    def _merge_bt_video(x: torch.Tensor, name: str) -> torch.Tensor:
        """Merge sample/time dims: (1,C,T,H,W) -> (T,C,H,W)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(f"Expected {name} shape (1,C,T,H,W), got {tuple(x.shape)}")
        return x[0].permute(1, 0, 2, 3)

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        # train dataset
        self.train_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["train"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            temporal_subsample_num_samples=self._temporal_subsample_num_samples,
        )

        # val dataset
        self.val_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            temporal_subsample_num_samples=self._temporal_subsample_num_samples,
        )

        # test dataset
        self.test_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["test"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            temporal_subsample_num_samples=self._temporal_subsample_num_samples,
        )

    def colln_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将不同的batch重新合成一个batch"""

        person_id = []
        turn_id = []
        labels = {
            "twist": [],
            "posture": [],
            "relax": [],
            "total": [],
        }
        fused_kpt3d_list = []

        for b in batch:
            fused_kpt3d = b.get("fused_kpt3d", None)
            twist_label = b["labels"]["twist"]
            posture_label = b["labels"]["posture"]
            relax_label = b["labels"]["relax"]
            total_label = b["labels"]["total"]

            bth, *_ = fused_kpt3d.shape

            for _ in range(bth):
                person_id.append(b["person_id"])
                turn_id.append(b["turn_id"])
                labels["twist"].append(twist_label)
                labels["posture"].append(posture_label)
                labels["relax"].append(relax_label)
                labels["total"].append(total_label)

            fused_kpt3d_list.append(fused_kpt3d)

        collated = {
            "person_id": person_id,
            "turn_id": turn_id,
            "labels": {
                "twist": torch.tensor(labels["twist"], dtype=torch.long),
                "posture": torch.tensor(labels["posture"], dtype=torch.long),
                "relax": torch.tensor(labels["relax"], dtype=torch.long),
                "total": torch.tensor(labels["total"], dtype=torch.long),
            },
            "fused_kpt3d": (
                torch.cat(fused_kpt3d_list, dim=0) if fused_kpt3d_list else None
            ),
        }
        return collated

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self.colln_fn,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,  # 🚀 GPU内存传输加速（改自False）
            shuffle=False,
            drop_last=True,
            collate_fn=self.colln_fn,
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,  # 🚀 GPU内存传输加速（改自False）
            shuffle=False,
            drop_last=True,
            collate_fn=self.colln_fn,
        )

        return test_data_loader
