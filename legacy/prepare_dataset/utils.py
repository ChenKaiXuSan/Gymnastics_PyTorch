"""
File: utils.py
Project: utils
Created Date: 2023-09-03 13:02:25
Author: chenkaixu
-----
Comment:

Have a good code time!
-----
Last Modified: 2023-09-03 13:03:05
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

import os

import cv2
from tqdm import tqdm

from pathlib import Path

import torch

import logging

logger = logging.getLogger(__name__)


def merge_frame_to_video(
    save_path: Path, person: str, video_name: str, flag: str, filter: bool = False
) -> None:

    if filter:
        _save_path = save_path / "vis" / "filter_img" / flag / person / video_name
        _out_path = save_path / "vis" / "filter_video" / flag / person
    else:
        _save_path = save_path / "vis" / "img" / flag / person / video_name
        _out_path = save_path / "vis" / "video" / flag / person

    frames = sorted(list(_save_path.iterdir()), key=lambda x: int(x.stem.split("_")[0]))

    if not _out_path.exists():
        _out_path.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(_out_path / video_name) + ".mp4", fourcc, 30.0, (width, height)
    )

    for f in tqdm(frames, desc=f"Save {flag}-{video_name}", total=len(frames)):
        img = cv2.imread(str(f))
        out.write(img)

    out.release()

    logger.info(f"Video saved to {_out_path / video_name}.mp4")


def process_none(batch_Dict: dict[torch.Tensor], none_index: list):
    """
    process_none, where from batch_Dict to instead the None value with next frame tensor (or froward frame tensor).

    Args:
        batch_Dict (dict): batch in Dict, where include the None value when yolo dont work.
        none_index (list): none index list map to batch_Dict, here not use this.

    Returns:
        list: list include the replace value for None value.
    """

    boundary = len(batch_Dict) - 1
    filter_batch = batch_Dict.copy()

    for i in none_index:

        # * if the index is None, we need to replace it with next frame.
        if batch_Dict[i] is None:
            next_idx = i + 1

            if next_idx < boundary:
                filter_batch[i] = batch_Dict[next_idx]
            else:
                filter_batch[i] = batch_Dict[boundary - 1]

    return filter_batch
