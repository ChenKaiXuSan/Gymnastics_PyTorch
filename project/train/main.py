#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
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
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from project.train.dataloader.data_loader import PersonDataModule
from project.train.map_config import PersonInfo


#####################################
# select different experiment trainer
#####################################

from project.train.trainer.train_STGCN import STGCNTrainer
from project.train.trainer.train_SSM import SSMTrainer
from project.train.trainer.train_TCN import TCNTrainer

# from project.train.trainer.train_fusion_SSM import FusionSSMTrainer

logger = logging.getLogger(__name__)


def _resolve_trainer_requirements(hparams: DictConfig) -> Dict[str, object]:
    """Infer required input modalities from current trainer selection."""
    backbone = str(hparams.model.backbone)
    backbone_key = backbone.lower()
    fuse_method = str(hparams.model.fuse_method)

    if backbone_key in {"st_gcn"}:
        return {
            "trainer_name": "STGCNTrainer",
            "requires_frames": False,
            "requires_2d_kpt": False,
            "requires_3d_kpt": True,
        }

    raise ValueError(
        "Cannot infer trainer input requirements for "
        f"backbone={backbone}, fuse_method={fuse_method}"
    )


def _validate_input_loading_config(hparams: DictConfig) -> Dict[str, object]:
    """Fail fast when dataloader switches do not satisfy trainer inputs."""
    req = _resolve_trainer_requirements(hparams)

    load_frames = bool(getattr(hparams.data, "load_frames", True))
    load_2d_kpt = bool(getattr(hparams.data, "load_2d_kpt", True))
    load_3d_kpt = bool(getattr(hparams.data, "load_3d_kpt", True))

    missing: List[str] = []
    if bool(req["requires_frames"]) and not load_frames:
        missing.append("frames")
    if bool(req["requires_2d_kpt"]) and not load_2d_kpt:
        missing.append("2D keypoints")
    if bool(req["requires_3d_kpt"]) and not load_3d_kpt:
        missing.append("3D keypoints")

    logger.info(
        "Trainer=%s | required inputs: frames=%s, 2d=%s, 3d=%s | enabled loading: frames=%s, 2d=%s, 3d=%s",
        req["trainer_name"],
        req["requires_frames"],
        req["requires_2d_kpt"],
        req["requires_3d_kpt"],
        load_frames,
        load_2d_kpt,
        load_3d_kpt,
    )

    if missing:
        raise ValueError(
            f"Current trainer {req['trainer_name']} requires: {', '.join(missing)}, "
            "but the corresponding dataloader switches are disabled. "
            f"Current config: load_frames={load_frames}, load_2d_kpt={load_2d_kpt}, load_3d_kpt={load_3d_kpt}"
        )

    return req


def load_fold_dataset_idx_from_fold_json(
    config: DictConfig, fold: int
) -> Dict[str, List[PersonInfo]]:
    """加载指定fold的JSON文件

    Args:
        config: Hydra配置对象
        fold: fold号 (0-4 for 5-fold, etc.)

    Returns:
        Dict[str, List[UnityDataConfig]]: {"train": [...], "val": [...], "test": [...]}
    """

    index_file_path = Path(str(config.data.index_mapping_path))

    fold_file = index_file_path / f"fold_{fold:02d}.json"

    with open(fold_file, "r", encoding="utf-8") as f:
        fold_data = json.load(f)

    fold_data.pop("_metadata", None)

    dataset_idx: Dict[str, List[PersonInfo]] = {"train": [], "val": [], "test": []}

    # 处理三种split
    for split in ["train", "val", "test"]:
        src_list = fold_data.get(split, [])

        for item in src_list:
            dataset_idx[split].append(
                PersonInfo(
                    person_id=item["person_id"],
                    turn_id=item["turn_id"],
                    cam1_video_path=item["cam1_video_path"],
                    cam2_video_path=item["cam2_video_path"],
                    sam3d_cam1_results_path=item["sam3d_cam1_results_path"],
                    sam3d_cam2_results_path=item["sam3d_cam2_results_path"],
                    cam1_turn_frame_start=item["cam1_turn_frame_start"],
                    cam1_turn_frame_end=item["cam1_turn_frame_end"],
                    cam2_turn_frame_start=item["cam2_turn_frame_start"],
                    cam2_turn_frame_end=item["cam2_turn_frame_end"],
                    label_twist_3class=int(item.get("label_twist_3class", -1)),
                    label_posture_3class=int(item.get("label_posture_3class", -1)),
                    label_relax_3class=int(item.get("label_relax_3class", -1)),
                    label_total_3class=int(item.get("label_total_3class", -1)),
                    fused_kpt_path=str(item.get("fused_kpt_path", "")),
                    fused_kpt_turn_frame_start=int(
                        item.get("fused_turn_frame_start", -1)
                    ),
                    fused_kpt_turn_frame_end=int(item.get("fused_turn_frame_end", -1)),
                )
            )

    logger.info(
        "Loaded fold %s: train=%s, val=%s, test=%s",
        fold,
        len(dataset_idx["train"]),
        len(dataset_idx["val"]),
        len(dataset_idx["test"]),
    )
    return dataset_idx


def train(hparams: DictConfig, dataset_idx, fold: int):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    logger.info("Using trainer %s for fold %s", hparams.model.backbone, fold)

    # * select experiment
    trainer_name = hparams.model.backbone

    if trainer_name == "st_gcn":
        classification_module = STGCNTrainer(hparams)
        monitor_metric = "val/loss"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}"
    elif trainer_name == "ssm":
        classification_module = SSMTrainer(hparams)
        monitor_metric = "val/loss"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}"
    elif trainer_name == "tcn":
        classification_module = TCNTrainer(hparams)
        monitor_metric = "val/loss"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}"
    else:
        raise ValueError(f"Unsupported trainer name: {trainer_name}")

    # * prepare data module
    data_module = PersonDataModule(hparams, dataset_idx=dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tb_logs"),
        name="fold_" + str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=10)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_path, "checkpoints", "fold_" + str(fold)),
        filename=ckpt_filename,
        auto_insert_metric_name=False,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_top_k=2,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=[tb_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            lr_monitor,
        ],
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches=2,
    )

    trainer.fit(classification_module, data_module)

    # save the metrics to file
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our checkpoints include OmegaConf objects, so force full load for trusted local ckpts.
    try:
        trainer.test(
            classification_module,
            data_module,
            ckpt_path="best",
            weights_only=False,
        )
    except TypeError:
        # Backward compatibility for Lightning versions without `weights_only` arg.
        trainer.test(
            classification_module,
            data_module,
            ckpt_path="best",
        )


@hydra.main(
    version_base=None,
    config_path="../../configs",  # * the config_path is relative to location of the python script
    config_name="train.yaml",
)
def init_params(config):
    #########
    # K fold
    #########
    # 使用预生成的单fold JSON文件（每个fold文件必须存在）

    n_splits = int(config.data.cross_validation.n_splits)
    for fold in range(n_splits):
        # 加载单个fold的JSON文件
        dataset_value = load_fold_dataset_idx_from_fold_json(config, fold)
        logger.info("#" * 50)
        logger.info("Start train fold: %s", fold)
        logger.info("#" * 50)

        train(config, dataset_value, fold)

        logger.info("#" * 50)
        logger.info("finish train fold: %s", fold)
        logger.info("#" * 50)

    logger.info("#" * 50)
    logger.info("finish train folds: total=%s", n_splits)
    logger.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
