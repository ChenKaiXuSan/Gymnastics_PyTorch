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


# The joints every method in the comparison predicts: the COCO-style body
# joints without nose (CanonPose's MPII-16 layout has no face joint that maps
# to it) and without neck (the H36M-17 methods' thorax does not survive the
# FreeMan reference's COCO-17 layout). Used for the cross-method table.
COMPARISON_JOINTS: tuple[str, ...] = (
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow", "left-wrist", "right-wrist",
    "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle", "right-ankle",
)

Predictor = Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor]]


def view_a_predictor(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """The default prediction: view A, which the trial transform has replaced with the method's pose."""
    return batch["pose_a"], batch["valid_a"]


def evaluate_fold(dataset: str, fold_json: Path, transform: Callable[[PosePairTrial], PosePairTrial] | None, *, extra_overrides: Sequence[str] = (), joints: Sequence[str] | None = None, predictor: Predictor | None = None) -> dict[str, Any]:
    """Per-frame PA-MPJPE of the prediction over the fold's test windows.

    Args:
        dataset: ``gymnastics``, ``freeman`` or ``sportspose``.
        fold_json: The fold file (its test subjects are scored).
        transform: Trial transform that inserts the method's poses, or ``None``.
        extra_overrides: Extra Hydra data overrides.
        joints: Restrict the metric to these joint names -- both the
            Procrustes alignment and the error then use exactly this set, so
            methods with different skeletons become comparable.
        predictor: ``batch -> (pose, valid)``; the default reads view A.
    """
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    cfg = compose_config(data_overrides(dataset, fold_json, extra_overrides))
    datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True), trial_transform=transform)  # type: ignore[arg-type]
    datamodule.setup("test")
    predict = predictor or view_a_predictor
    subset = None
    if joints is not None:
        names = list(datamodule.skeleton.joint_names)
        missing = [name for name in joints if name not in names]
        if missing:
            raise ValueError(f"{dataset}: the skeleton has no joints {missing}")
        subset = torch.zeros(len(names), dtype=torch.bool)
        subset[[names.index(name) for name in joints]] = True
    total, weight = 0.0, 0
    joints_used: set[int] = set()
    windows = 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            frame_mask = batch["frame_mask"][..., None]
            pose, valid = predict(batch)
            usable = batch["reference_valid"] & valid & frame_mask
            if subset is not None:
                usable = usable & subset
            errors, mask = per_joint_error(pose, batch["reference"], usable, align="procrustes")
            value = float(masked_mean(errors, mask))
            size = int(pose.shape[0])
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


def fold_files(dataset: str, folds_dir: Path | None = None) -> list[Path]:
    folds = sorted((folds_dir or PROJECT_ROOT / FOLD_DIRS[dataset]).glob("fold_*.json"))
    if not folds:
        raise FileNotFoundError(f"no fold files for {dataset}")
    return folds


def evaluate_folds(dataset: str, transform_factory: Callable[[Path], Callable[[PosePairTrial], PosePairTrial] | None], *, folds_dir: Path | None = None, extra_overrides: Sequence[str] = (), joints: Sequence[str] | None = None, predictor_factory: Callable[[Path], Predictor] | None = None) -> dict[str, Any]:
    """All folds of ``dataset``; ``transform_factory(fold_json)`` builds the transform of each fold
    (trained methods load that fold's checkpoint, released methods ignore the argument)."""
    folds = fold_files(dataset, folds_dir)
    results = [evaluate_fold(dataset, fold, transform_factory(fold), extra_overrides=extra_overrides, joints=joints, predictor=predictor_factory(fold) if predictor_factory else None) for fold in folds]
    values = [r["pa_mpjpe"] for r in results]
    return {
        "dataset": dataset,
        "folds": results,
        "summary": {"pa_mpjpe_mean": statistics.fmean(values), "pa_mpjpe_sd": statistics.pstdev(values) if len(values) > 1 else 0.0, "folds": len(values), "joint_names": results[0]["joint_names"], "joint_subset": list(joints) if joints else None},
    }


def write_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)), encoding="utf-8")
    return path
