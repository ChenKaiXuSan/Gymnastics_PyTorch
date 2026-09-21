"""Evaluate a published method through the model protocol.

The method's trial transform is plugged into the standard DataModule of the
dataset (Hydra ``data=<dataset>`` config, one fold file at a time), so the
windows, phase normalisation, test subjects and reference handling are the
ones every learned run uses. The metric is the per-frame Procrustes-aligned
error of view A on the 20 major joints, aggregated exactly like the
Lightning module logs ``test/pa_mpjpe_face`` (joint-frame weighted mean per
batch, batch-size weighted over the epoch).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from common.paths import PROJECT_ROOT
from fusion.keypoints.schema import PosePairTrial
from fusion.losses import masked_mean
from fusion.metrics import per_joint_error

FOLD_DIRS = {
    "gymnastics": "src/configs/fusion/folds/gymnastics",
    "freeman": "src/configs/fusion/folds/freeman_all40",
    "sportspose": "src/configs/fusion/folds/sportspose",
}


def data_overrides(dataset: str, fold_json: Path, extra: Sequence[str] = ()) -> list[str]:
    """Hydra overrides that reproduce a training run's data config for evaluation only."""
    return [
        f"data={dataset}",
        f"data.fold_json={fold_json}",
        "data.cache_dir=null",
        "data.attach_reference=true",
        "data.test_with_corruption=false",
        "data.validate_with_corruption=false",
        "data.num_workers=0",
        "data.cycle_target.enabled=false",
        *extra,
    ]


def evaluate_fold(dataset: str, fold_json: Path, transform: Callable[[PosePairTrial], PosePairTrial] | None, *, extra_overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Per-frame PA-MPJPE of view A over the fold's test windows."""
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    cfg = compose_config(data_overrides(dataset, fold_json, extra_overrides))
    datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True), trial_transform=transform)  # type: ignore[arg-type]
    datamodule.setup("test")
    total, weight = 0.0, 0
    joints_used: set[int] = set()
    windows = 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            frame_mask = batch["frame_mask"][..., None]
            usable = batch["reference_valid"] & batch["valid_a"] & frame_mask
            errors, mask = per_joint_error(batch["pose_a"], batch["reference"], usable, align="procrustes")
            value = float(masked_mean(errors, mask))
            size = int(batch["pose_a"].shape[0])
            total += value * size
            weight += size
            windows += size
            joints_used.update(int(j) for j in torch.nonzero(mask.any(dim=(0, 1))).flatten().tolist())
    return {
        "fold": fold_json.stem,
        "pa_mpjpe": total / max(weight, 1),
        "test_windows": windows,
        "test_subjects": list(datamodule.split.test),
        "joints_evaluated": sorted(joints_used),
        "joint_names": [datamodule.skeleton.joint_names[j] for j in sorted(joints_used)],
    }


def evaluate_folds(dataset: str, transform_factory: Callable[[], Callable[[PosePairTrial], PosePairTrial] | None], *, folds_dir: Path | None = None, extra_overrides: Sequence[str] = ()) -> dict[str, Any]:
    """All folds of ``dataset``; ``transform_factory`` builds the transform once per fold."""
    folds = sorted((folds_dir or PROJECT_ROOT / FOLD_DIRS[dataset]).glob("fold_*.json"))
    if not folds:
        raise FileNotFoundError(f"no fold files for {dataset}")
    results = [evaluate_fold(dataset, fold, transform_factory(), extra_overrides=extra_overrides) for fold in folds]
    values = [r["pa_mpjpe"] for r in results]
    return {
        "dataset": dataset,
        "folds": results,
        "summary": {"pa_mpjpe_mean": statistics.fmean(values), "pa_mpjpe_sd": statistics.pstdev(values) if len(values) > 1 else 0.0, "folds": len(values), "joint_names": results[0]["joint_names"]},
    }


def write_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)), encoding="utf-8")
    return path
