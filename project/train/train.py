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
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from dataloader.data_loader import PersonDataModule
from map_config import PersonInfo

#####################################
# select different experiment trainer
#####################################

from trainer.train_STGCN import STGCNTrainer
from trainer.train_SSM import SSMTrainer
from trainer.train_TCN import TCNTrainer

logger = logging.getLogger(__name__)


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
                    label_total_5class=int(item.get("label_total_5class", -1)),
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
    elif trainer_name == "body_part_mamba":
        from trainer.train_body_part_mamba import BodyPartMambaClassificationModule

        classification_module = BodyPartMambaClassificationModule(hparams)
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

    csv_logger = CSVLogger(
        save_dir=os.path.join(hparams.log_path, "csv_logs"),
        name="fold_" + str(fold),
    )

    # some callbacks
    progress_bar = RichProgressBar(refresh_rate=10, leave=True)
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
        logger=[tb_logger, csv_logger],
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
        test_metrics = trainer.test(
            classification_module,
            data_module,
            ckpt_path="best",
            weights_only=False,
        )
    except TypeError:
        # Backward compatibility for Lightning versions without `weights_only` arg.
        test_metrics = trainer.test(
            classification_module,
            data_module,
            ckpt_path="best",
        )

    # write test metrics to txt file
    metrics_save_path = os.path.join(
        hparams.log_path, "metrics", f"fold_{fold}_test_metrics.txt"
    )
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(test_metrics, indent=4))


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

        train(config, dataset_value, fold)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
