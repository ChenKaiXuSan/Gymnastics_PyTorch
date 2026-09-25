"""Calibrate the depth discount ``alpha`` of the closed-form rule, globally and per joint.

``P_base = (Lambda_A + Lambda_B)^-1 (Lambda_A P_A + Lambda_B P_B)`` with
``Lambda_v = I - alpha d_v d_v^T``: ``alpha`` is the share of a view's
precision removed along its own optical axis. It is the rule's only
hyper-parameter and was chosen once as a global 0.8 on FreeMan. This module
re-selects it the leak-free way and per joint:

* fold-wise: for every fold, the value (or the 20-joint table) that
  minimises the rule's error on the fold's *validation* subjects against the
  dataset's own reference is applied to the fold's *test* subjects;
* dataset-wide tables (all subjects) are also written, to be *transferred*
  to the other datasets -- cross-dataset use touches no reported subject;
* on the private dataset the reference is a two-view triangulation of the
  same detections and rewards ``alpha -> 1`` by construction, so its own
  calibration is written only as a diagnostic, never applied.

    python -m fusion external-published alpha --dataset freeman [--joints comparison12|all]
    python -m fusion external-published alpha --dataset gymnastics --tables freeman,fit3d

Outputs under ``local/runs/external_published/alpha/<dataset>/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from common.paths import PROJECT_ROOT
from fusion.metrics import per_joint_error
from fusion.modules.weighted_fusion import depth_aware_pose_fusion

GRID: tuple[float, ...] = (0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.95, 0.98)
DEFAULT_ALPHA = 0.8
MIN_PRECISION = 0.5
OUTPUT_ROOT = PROJECT_ROOT / "local" / "runs" / "external_published" / "alpha"


def collect(dataset: str, fold_json: Path, split: str, *, extra: Sequence[str] = ()) -> dict[str, Any]:
    """Every window of ``split`` as CPU tensors plus the subject of each window."""
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    from .evaluate import data_overrides

    cfg = compose_config(data_overrides(dataset, fold_json, extra))
    datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
    datamodule.setup("validate" if split == "val" else "test")
    loader = datamodule.val_dataloader() if split == "val" else datamodule.test_dataloader()
    keys = ("pose_a", "pose_b", "valid_a", "valid_b", "depth_a", "depth_b", "reference", "reference_valid", "frame_mask")
    stacked: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    subjects: list[str] = []
    for batch in loader:
        for k in keys:
            stacked[k].append(batch[k])
        subjects.extend(str(s) for s in batch["subject_id"])
    out: dict[str, Any] = {k: torch.cat(v) for k, v in stacked.items()}
    out["subjects"] = subjects
    out["joint_names"] = list(datamodule.skeleton.joint_names)
    return out


def rule(data: dict[str, Any], alpha: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    """The equal-weight rule with a global ``alpha`` or a per-joint ``[J]`` table (fused joint by joint)."""
    pose_a, pose_b = data["pose_a"], data["pose_b"]
    frame_mask = data["frame_mask"][..., None]
    valid_a, valid_b = data["valid_a"] & frame_mask, data["valid_b"] & frame_mask
    half = torch.full_like(pose_a[..., :1], 0.5)
    if not torch.is_tensor(alpha):
        return depth_aware_pose_fusion(pose_a, pose_b, half, half, valid_a, valid_b, data["depth_a"], data["depth_b"], alpha=float(alpha), min_precision=MIN_PRECISION)
    fused = torch.zeros_like(pose_a)
    valid = torch.zeros(pose_a.shape[:-1], dtype=torch.bool)
    for value in torch.unique(alpha):
        joints = torch.nonzero(alpha == value).flatten()
        f, v = depth_aware_pose_fusion(pose_a[:, :, joints], pose_b[:, :, joints], half[:, :, joints], half[:, :, joints], valid_a[:, :, joints], valid_b[:, :, joints], data["depth_a"], data["depth_b"], alpha=float(value), min_precision=MIN_PRECISION)
        fused[:, :, joints], valid[:, :, joints] = f, v
    return fused, valid


def per_joint_errors(data: dict[str, Any], fused: torch.Tensor, valid: torch.Tensor, subset: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray]:
    """``(sum of error per joint [J], count per joint [J])`` after the metric's per-frame Procrustes alignment."""
    usable = data["reference_valid"] & valid & data["frame_mask"][..., None]
    if subset is not None:
        usable = usable & subset
    errors, mask = per_joint_error(fused, data["reference"], usable, align="procrustes")
    return torch.where(mask, errors, torch.zeros_like(errors)).sum(dim=(0, 1)).numpy(), mask.sum(dim=(0, 1)).numpy()


def sweep(data: dict[str, Any], subset: torch.Tensor | None) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Per-joint error sums and counts of the rule for every alpha of the grid."""
    return {alpha: per_joint_errors(data, *rule(data, alpha), subset) for alpha in GRID}


def select(curves: dict[float, tuple[np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Global and per-joint argmin over the grid; also the curves in mm."""
    alphas = list(curves)
    totals = np.array([curves[a][0].sum() / max(curves[a][1].sum(), 1) for a in alphas])
    per_joint = np.array([curves[a][0] / np.maximum(curves[a][1], 1) for a in alphas])  # [A, J]
    global_alpha = alphas[int(np.argmin(totals))]
    joint_alpha = np.array([alphas[i] for i in np.argmin(per_joint, axis=0)])
    return global_alpha, joint_alpha, {"alphas": alphas, "global_mm": (1000 * totals).tolist(), "per_joint_mm": (1000 * per_joint).tolist()}


def score(data: dict[str, Any], alpha: torch.Tensor | float, subset: torch.Tensor | None) -> float:
    total, count = per_joint_errors(data, *rule(data, alpha), subset)
    return float(1000 * total.sum() / max(count.sum(), 1))


def run(dataset: str, *, joints: Sequence[str] | None, tables: dict[str, np.ndarray] | None = None, folds_dir: Path | None = None, extra: Sequence[str] = (), calibrate: bool = True) -> dict[str, Any]:
    from .evaluate import fold_files

    folds = fold_files(dataset, folds_dir)
    results: dict[str, Any] = {"dataset": dataset, "grid": list(GRID), "folds": [], "joint_names": None}
    dataset_curves: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {a: [] for a in GRID}
    for fold in folds:
        test = collect(dataset, fold, "test", extra=extra)
        names = test["joint_names"]
        results["joint_names"] = names
        subset = None
        if joints is not None:
            subset = torch.zeros(len(names), dtype=torch.bool)
            subset[[names.index(n) for n in joints]] = True
        row: dict[str, Any] = {"fold": fold.stem, "test_subjects": sorted(set(test["subjects"]), key=lambda s: (len(s), s))}
        row["test_default"] = score(test, DEFAULT_ALPHA, subset)
        row["test_hard"] = score(test, GRID[-1], subset)
        row["test_average"] = score(test, 0.0, subset)
        if calibrate:
            val = collect(dataset, fold, "val", extra=extra)
            curves = sweep(val, subset)
            global_alpha, joint_alpha, curve_report = select(curves)
            row.update({"val_global_alpha": global_alpha, "val_joint_alpha": joint_alpha.tolist(), "val_curves": curve_report,
                        "test_val_global": score(test, global_alpha, subset), "test_val_joint": score(test, torch.tensor(joint_alpha, dtype=torch.float32), subset)})
            # The dataset-wide table pools every fold's *test* split = all subjects once (for transfer only).
            test_curves = sweep(test, subset)
            for a in GRID:
                dataset_curves[a].append(test_curves[a])
        for name, table in (tables or {}).items():
            row[f"test_table_{name}"] = score(test, torch.tensor(table, dtype=torch.float32), subset)
        results["folds"].append(row)
        print(f"[alpha] {dataset} {fold.stem}: default {row['test_default']:.2f}  hard {row['test_hard']:.2f}  average {row['test_average']:.2f}"
              + (f"  val-global(alpha={row['val_global_alpha']}) {row['test_val_global']:.2f}  val-joint {row['test_val_joint']:.2f}" if calibrate else "")
              + "".join(f"  table[{n}] {row[f'test_table_{n}']:.2f}" for n in (tables or {})), flush=True)
    if calibrate:
        pooled = {a: (np.sum([c[0] for c in dataset_curves[a]], axis=0), np.sum([c[1] for c in dataset_curves[a]], axis=0)) for a in GRID}
        global_alpha, joint_alpha, curve_report = select(pooled)
        results["dataset_table"] = {"global_alpha": global_alpha, "joint_alpha": joint_alpha.tolist(), "curves": curve_report}
        print(f"[alpha] {dataset} dataset-wide: global alpha {global_alpha}, per joint {dict(zip(results['joint_names'], joint_alpha.tolist()))}", flush=True)
    keys = [k for k in results["folds"][0] if k.startswith("test_")]
    results["summary"] = {k: {"mean": statistics.fmean(r[k] for r in results["folds"]), "sd": statistics.pstdev([r[k] for r in results["folds"]])} for k in keys}
    return results


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from .evaluate import COMPARISON_JOINTS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    parser.add_argument("--joints", default="all", help="'all' (the 20 major joints) or 'comparison12'")
    parser.add_argument("--tables", default="", help="comma/plus-separated datasets whose dataset-wide alpha tables are transferred")
    parser.add_argument("--no-calibrate", action="store_true", help="only score the default / hard / transferred tables (private data)")
    parser.add_argument("--folds-dir", type=Path, default=None)
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    joints = list(COMPARISON_JOINTS) if args.joints == "comparison12" else None
    tables: dict[str, np.ndarray] = {}
    for name in [t for t in args.tables.replace("+", ",").split(",") if t]:
        path = OUTPUT_ROOT / name / f"calibration_{args.joints}.json"
        tables[name] = np.array(json.loads(path.read_text(encoding="utf-8"))["dataset_table"]["joint_alpha"], dtype=np.float32)
    results = run(args.dataset, joints=joints, tables=tables, folds_dir=args.folds_dir, extra=list(args.override or []), calibrate=not args.no_calibrate)
    out = OUTPUT_ROOT / args.dataset / (f"calibration_{args.joints}.json" if not args.no_calibrate else f"transfer_{args.joints}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[alpha] {args.dataset} summary (fold mean ± sd, mm):")
    for k, v in results["summary"].items():
        print(f"   {k:22s} {v['mean']:6.2f} ± {v['sd']:4.2f}")
    print(f"-> {out}")
    return 0
