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
    EarlyStopping,
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
from project.train.trainer.train_fusion_SSM import FusionSSMTrainer

logger = logging.getLogger(__name__)


def _resolve_trainer_requirements(hparams: DictConfig) -> Dict[str, object]:
    """Infer required input modalities from current trainer selection."""
    backbone = str(hparams.model.backbone)
    fuse_method = str(hparams.model.fuse_method)

    if backbone == "3dcnn":
        if fuse_method in ["ssm", "mamba", "mamba_ssm"]:
            return {
                "trainer_name": "FusionSSMTrainer",
                "requires_frames": False,
                "requires_2d_kpt": False,
                "requires_3d_kpt": True,
            }
    elif backbone == "stgcn":
        return {
            "trainer_name": "STGCNTrainer",
            "requires_frames": False,
            "requires_2d_kpt": True,
            "requires_3d_kpt": False,
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
                    turn_id=item["action_id"],
                    cam1_video_path=item["cam1_path"],
                    cam2_video_path=item["cam2_path"],
                    sam3d_cam1_results_path=item["sam3d_cam1_kpt3d_dir"],
                    sam3d_cam2_results_path=item["sam3d_cam2_kpt3d_dir"],
                    cam1_turn_frame_start=item["cam1_turn_frame_start"],
                    cam1_turn_frame_end=item["cam1_turn_frame_end"],
                    cam2_turn_frame_start=item["cam2_turn_frame_start"],
                    cam2_turn_frame_end=item["cam2_turn_frame_end"],
                )
            )

    logger.info(
        f"✓ Loaded fold {fold}: train={len(dataset_idx['train'])}, val={len(dataset_idx['val'])}, test={len(dataset_idx['test'])}"
    )
    return dataset_idx


def load_fold_dataset_idx_from_index_mapping(config: DictConfig):
    """Load precomputed fold mapping from index json file.

    This removes CV split preparation from training entry.
    """
    index_mapping_cfg = Path(str(config.data.index_mapping))
    index_file_name = str(config.data.index_mapping_file)

    # Backward/forward compatible:
    # 1) data.index_mapping points to directory + data.index_mapping_file
    # 2) data.index_mapping points directly to a json file
    if index_mapping_cfg.suffix == ".json":
        index_file = index_mapping_cfg
    else:
        index_file = index_mapping_cfg / index_file_name

    if not index_file.exists():
        raise FileNotFoundError(
            f"Index mapping file not found: {index_file}. "
            f"Please generate it first (e.g. cross_validation/generate_cv_index.py)."
        )

    with open(index_file, "r", encoding="utf-8") as f:
        serial = json.load(f)

    # Skip metadata entry if exists.
    serial.pop("_metadata", None)

    fold_dataset_idx: Dict[int, Dict[str, List[UnityDataConfig]]] = {}
    for kfold, d in serial.items():
        if not isinstance(d, dict):
            raise ValueError(f"Fold {kfold} must be a dict, got {type(d)}")

        fold = int(kfold)
        fold_dataset_idx[fold] = {"train": [], "val": [], "test": []}

        # Accept both val/valid and test aliases from different generators.
        split_aliases = {
            "train": ["train"],
            "val": ["val", "valid"],
            "test": ["test", "eval", "holdout"],
        }

        for split, aliases in split_aliases.items():
            src_list = None
            for alias in aliases:
                if alias in d:
                    src_list = d[alias]
                    break
            if src_list is None:
                # Backward compatibility: old index files may not have test split.
                if split == "test" and "val" in d:
                    src_list = d["val"]
                else:
                    raise KeyError(
                        f"Fold {kfold} missing split '{split}' (aliases: {aliases})"
                    )
            if not isinstance(src_list, list):
                raise TypeError(
                    f"Fold {kfold} split '{split}' must be a list, got {type(src_list)}"
                )

            for item in src_list:
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Index item in fold {kfold}/{split} must be dict, got {type(item)}"
                    )

                # camera-pair index format: build UnityDataConfig directly.
                if "cam1_frames_dir" in item and "cam2_frames_dir" in item:
                    required_fields = [
                        "person_id",
                        "action_id",
                        "cam1_id",
                        "cam2_id",
                        "cam1_path",
                        "cam2_path",
                        "label_path",
                        "cam1_frames_dir",
                        "cam2_frames_dir",
                        "cam1_kpt2d_dir",
                        "cam2_kpt2d_dir",
                        "kpt3d_dir",
                        "sam3d_cam1_kpt2d_dir",
                        "sam3d_cam2_kpt2d_dir",
                        "sam3d_cam1_kpt3d_dir",
                        "sam3d_cam2_kpt3d_dir",
                        "sequence_meta_path",
                        "joint_names_path",
                    ]
                    missing = [k for k in required_fields if k not in item]
                    if missing:
                        raise KeyError(
                            f"Fold {kfold}/{split} missing required UnityDataConfig keys: {missing}"
                        )

                    fold_dataset_idx[fold][split].append(
                        UnityDataConfig(
                            person_id=str(item["person_id"]),
                            action_id=str(item["action_id"]),
                            cam1_id=str(item["cam1_id"]),
                            cam2_id=str(item["cam2_id"]),
                            cam1_path=str(item["cam1_path"]),
                            cam2_path=str(item["cam2_path"]),
                            label_path=str(item["label_path"]),
                            cam1_frames_dir=str(item["cam1_frames_dir"]),
                            cam2_frames_dir=str(item["cam2_frames_dir"]),
                            cam1_kpt2d_dir=str(item["cam1_kpt2d_dir"]),
                            cam2_kpt2d_dir=str(item["cam2_kpt2d_dir"]),
                            kpt3d_dir=str(item["kpt3d_dir"]),
                            sam3d_cam1_kpt2d_dir=str(item["sam3d_cam1_kpt2d_dir"]),
                            sam3d_cam2_kpt2d_dir=str(item["sam3d_cam2_kpt2d_dir"]),
                            sam3d_cam1_kpt3d_dir=str(item["sam3d_cam1_kpt3d_dir"]),
                            sam3d_cam2_kpt3d_dir=str(item["sam3d_cam2_kpt3d_dir"]),
                            sequence_meta_path=str(item["sequence_meta_path"]),
                            joint_names_path=str(item["joint_names_path"]),
                            annotation_path=str(item.get("annotation_path", ""))
                            or None,
                            label_twist_3class=(
                                int(item["label_twist_3class"])
                                if item.get("label_twist_3class") is not None
                                else None
                            ),
                            label_posture_3class=(
                                int(item["label_posture_3class"])
                                if item.get("label_posture_3class") is not None
                                else None
                            ),
                            label_relax_3class=(
                                int(item["label_relax_3class"])
                                if item.get("label_relax_3class") is not None
                                else None
                            ),
                            label_total_3class=(
                                int(item["label_total_3class"])
                                if item.get("label_total_3class") is not None
                                else None
                            ),
                        )
                    )
                    continue

                raise ValueError(
                    "Unsupported index item format. "
                    f"Expected camera-pair fields, got keys: {list(item.keys())}"
                )

    return fold_dataset_idx


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
    req = _validate_input_loading_config(hparams)
    logger.info("Using trainer %s for fold %s", req["trainer_name"], fold)

    # * select experiment
    monitor_metric = "val/video_acc"
    monitor_mode = "max"
    ckpt_filename = "{epoch}-{val/loss:.2f}-{val/video_acc:.4f}"

    if hparams.model.backbone == "st_gcn":

        classification_module = STGCNTrainer(hparams)
        monitor_metric = "val/loss"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}"

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

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=5,
        mode=monitor_mode,
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
            # early_stopping,
            lr_monitor,
        ],
        # limit_train_batches=10,
        # limit_val_batches=10,
        # limit_test_batches=10,
    )

    trainer.fit(classification_module, data_module)

    # save the metrics to file
    trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
    )


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="train.yaml",
)
def init_params(config):
    # Load precomputed fold mapping only; do not prepare CV splits here.
    # 使用预生成的单fold JSON文件（每个fold文件必须存在）
    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold in range(config.data.n_splits):
        # 加载单个fold的JSON文件
        dataset_value = load_fold_dataset_idx_from_fold_json(config, fold)
        logger.info("#" * 50)
        logger.info(f"Start train fold: {fold}")
        logger.info("#" * 50)

        train(config, dataset_value, fold)

        logger.info("#" * 50)
        logger.info(f"finish train fold: {fold}")
        logger.info("#" * 50)

    logger.info("#" * 50)
    logger.info("finish train folds: %s", fold)
    logger.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
