#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/eval.py
Project: /workspace/code/project
Created Date: Tuesday November 11th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday November 11th 2025 1:34:34 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import re
import glob
import json
import time
import math
import logging
from typing import Dict, List, Tuple, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor

# DataModule
from project.dataloader.data_loader import UnityDataModule

# Trainers (LightningModules)
from project.trainer.baseline.train_3dcnn import Res3DCNNTrainer
from project.trainer.mid.train_pose_attn import PoseAttnTrainer
from project.trainer.early.train_early_fusion import EarlyFusion3DCNNTrainer
from project.trainer.late.train_late_fusion import LateFusion3DCNNTrainer

# K-fold splitter
from project.cross_validation import DefineCrossValidation

logger = logging.getLogger(__name__)


def _select_module(hparams: DictConfig):
    """Mirror the selection logic used in main.py"""
    if hparams.model.backbone != "3dcnn":
        raise ValueError("Only backbone='3dcnn' is supported in this eval script.")

    fm = hparams.model.fuse_method
    if fm == "pose_atn":
        return PoseAttnTrainer(hparams)
    elif fm in ["add", "mul", "concat", "avg"]:
        return EarlyFusion3DCNNTrainer(hparams)
    elif fm == "late":
        return LateFusion3DCNNTrainer(hparams)
    elif fm == "none":
        return Res3DCNNTrainer(hparams)
    else:
        raise ValueError(f"Unsupported fuse_method: {fm}")


def _parse_ckpt_metric(path: str) -> Optional[Tuple[int, float, float]]:
    """Parse epoch, val_loss, val_acc from checkpoint filename."""

    base = os.path.basename(path)
    epoch, vloss, vacc = base.split("-")[0:3]
    vacc = vacc.replace(".ckpt", "")

    return int(epoch), float(vloss), float(vacc)


def _find_best_ckpt_for_fold(log_path: str, fold: str | int) -> Optional[str]:
    """
    Search all version_* for this fold's checkpoints and pick the highest val/video_acc.
    Fallback order:
      1) best by parsed val/video_acc
      2) last.ckpt
      3) None (caller will run without pretrained weights)
    """
    fold_dir = os.path.join(log_path, str(fold))
    if not os.path.isdir(fold_dir):
        return None

    # 1) collect all candidate ckpts with metrics in filename
    pattern = os.path.join(fold_dir, "version_*", "checkpoints", "*.ckpt")
    candidates = glob.glob(pattern)
    best_path = None
    best_acc = -math.inf

    for p in candidates:
        if "last.ckpt" in os.path.basename(p).lower():
            continue
        parsed = _parse_ckpt_metric(p)
        if parsed is None:
            continue
        _, _, vacc = parsed
        if vacc > best_acc:
            best_acc = vacc
            best_path = p

    if best_path is not None:
        return best_path

    # 2) try last.ckpt
    last_candidates = [
        p for p in candidates if os.path.basename(p).lower() == "last.ckpt"
    ]
    if last_candidates:
        # if multiple, choose the newest by mtime
        last_candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return last_candidates[0]

    # 3) nothing found
    return None


def _aggregate(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate a list of test result dicts (one per fold).
    Returns a dict: metric -> {"mean": ..., "std": ...}
    Only aggregates keys starting with 'test/' and whose values are numbers.
    """
    from collections import defaultdict
    import numpy as np

    buckets = defaultdict(list)
    for r in results:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k.startswith("test/"):
                buckets[k].append(float(v))

    agg: Dict[str, Dict[str, float]] = {}
    for k, arr in buckets.items():
        if len(arr) == 0:
            continue
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=0))
        agg[k] = {"mean": m, "std": s}
    return agg


def _save_outputs(
    out_dir: str,
    fold_results: Dict[str, Dict[str, float]],
    aggregate_stats: Dict[str, Dict[str, float]],
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"eval_results_{ts}.json")
    csv_path = os.path.join(out_dir, f"eval_results_{ts}.csv")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_fold": fold_results,
                "aggregate": aggregate_stats,
                "created_at": ts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # CSV (flatten)
    import csv

    # collect header
    metric_names = set()
    for _fold, metrics in fold_results.items():
        metric_names.update([k for k in metrics.keys() if k.startswith("test/")])
    metric_names = sorted(metric_names)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold"] + metric_names)
        for fld, metrics in sorted(fold_results.items(), key=lambda x: str(x[0])):
            row = [fld] + [metrics.get(m, "") for m in metric_names]
            writer.writerow(row)
        # add a blank line and aggregate
        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])
        for m in metric_names:
            ms = aggregate_stats.get(m, {})
            writer.writerow([m, ms.get("mean", ""), ms.get("std", "")])

    return json_path, csv_path


def _make_trainer(hparams: DictConfig) -> Trainer:
    """Minimal trainer for evaluation."""
    return Trainer(
        devices=[int(hparams.train.gpu)],
        accelerator="gpu",
        logger=CSVLogger(
            save_dir=os.path.join(hparams.train.log_path, "eval_csv_logs"),
            name="eval",
        ),
        callbacks=[DeviceStatsMonitor()],
    )


def _eval_one_fold(hparams: DictConfig, dataset_idx, fold: int) -> Dict[str, float]:
    """Run test() for one fold and return the metrics dict."""
    seed_everything(42, workers=True)

    # module and data
    module = _select_module(hparams)
    datamodule = WalkDataModule(hparams, dataset_idx)

    # locate ckpt
    ckpt = _find_best_ckpt_for_fold(hparams.eval.input_path, fold)
    if ckpt:
        logger.info(f"[fold {fold}] Using checkpoint: {ckpt}")
    else:
        logger.warning(
            f"[fold {fold}] No checkpoint found. Running test() with randomly initialized weights."
        )

    trainer = _make_trainer(hparams)

    # Run test
    if ckpt:
        test_out = trainer.test(module, datamodule, ckpt_path=ckpt)
    else:
        test_out = trainer.test(module, datamodule)

    # PL returns a list[dict]; typically len==1 unless multiple test loaders
    if not test_out:
        logger.warning(f"[fold {fold}] Empty test result; returning empty dict.")
        return {}

    # If multiple dicts, merge keys by later overwriting (usually fine)
    merged: Dict[str, float] = {}
    for d in test_out:
        merged.update(d)
    return merged


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="eval.yaml",
)
def main(config: DictConfig):
    """
    K-fold evaluation:
    - Split folds using DefineCrossValidation(config)()
    - For each fold, load best ckpt if available and run trainer.test
    - Save per-fold results and aggregate mean/std to log_path
    """
    # Prepare folds
    fold_dataset_idx = DefineCrossValidation(config)()
    logger.info("#" * 60)
    logger.info("Start EVALUATION over all folds")
    logger.info("#" * 60)

    per_fold_results: Dict[str, Dict[str, float]] = {}

    for fold, dataset_value in fold_dataset_idx.items():
        logger.info("#" * 60)
        logger.info(f"Evaluating fold: {fold}")
        logger.info("#" * 60)

        metrics = _eval_one_fold(config, dataset_value, fold)
        per_fold_results[str(fold)] = metrics

        # Pretty print small summary
        if metrics:
            nice = {
                k: round(v, 6)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            logger.info(f"[fold {fold}] test metrics: {nice}")
        else:
            logger.info(f"[fold {fold}] No metrics returned.")

    # Aggregate
    aggregate_stats = _aggregate(list(per_fold_results.values()))

    logger.info("#" * 60)
    logger.info("Aggregate (mean ± std) for test/* metrics:")
    for m, s in sorted(aggregate_stats.items()):
        logger.info(f"  {m}: mean={s['mean']:.6f}, std={s['std']:.6f}")
    logger.info("#" * 60)

    # Save
    out_dir = config.eval.log_path
    json_path, csv_path = _save_outputs(out_dir, per_fold_results, aggregate_stats)
    logger.info(f"Saved evaluation results:\n  JSON: {json_path}\n  CSV : {csv_path}")
    logger.info("Finished EVALUATION over all folds.")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
