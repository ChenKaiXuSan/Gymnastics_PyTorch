#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/Skiing_Analysis_PyTorch/prepare_dataset/preprocess.py
Project: /workspace/code/Skiing_Analysis_PyTorch/prepare_dataset
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday April 23rd 2025 12:56:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from pathlib import Path

import torch

from prepare_dataset.model.detectron2 import Detectron2Wrapper

logger = logging.getLogger(__name__)


class PreprocessD2:

    def __init__(self, config, person: str) -> None:
        super(PreprocessD2, self).__init__()

        self.d2_model = Detectron2Wrapper(config, person=person)

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        d2_pose, d2_pose_score, d2_bboxes_xyxy, d2_none_index = self.d2_model(
            vframes, video_path
        )

        return (
            d2_pose,
            d2_pose_score,
            d2_bboxes_xyxy,
            d2_none_index,
        )
