"""Two analyses the millimetre table cannot answer.

**Stratified error** -- where does the learned model beat its closed-form
rule? The comparison is run per frame and split by conditions that are
visible without the reference: how much of the two views' input is missing
(occlusion proxy), whether the frame sits at a turn-around (phase extremes)
or in mid-swing, how fast the body moves, and the cohort (elderly / student
on the private data). The learned parts lose on clean averages, so the
question is whether a hard stratum exists where they win.

**Measurement error** -- the application measures trunk rotation, not
millimetres. For each cycle the range of motion and the peak angular
velocity of the shoulder-versus-hip twist (`fusion.measurement`) are
compared with the reference's, for the model, its rule and each view.

    python -m fusion external-published analysis --dataset gymnastics \
        --run local/runs/cycle_aware/gymnastics_v1_1_5fold_seed0

Writes ``local/runs/external_published/analysis/<run>/{strata,measurement}.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from common.paths import PROJECT_ROOT
from fusion.measurement import per_cycle_extremes, trunk_twist
from fusion.metrics import per_joint_error

COHORT_ELDERLY_MIN = 58  # private ids: 1-57 students, 58-137 elderly (src/configs/shared/folds)


def frame_errors(pose: torch.Tensor, reference: torch.Tensor, usable: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``([B, T] mean error, [B, T] valid)``: the metric aligns each frame separately."""
    errors, mask = per_joint_error(pose, reference, usable, align="procrustes")
    count = mask.sum(dim=-1)
    total = torch.where(mask, errors, torch.zeros_like(errors)).sum(dim=-1)
    return total / count.clamp_min(1), count > 0


def frame_strata(batch: dict[str, Any], dataset: str) -> dict[str, list[list[str]]]:
    """Per-frame strata labels (``[B][T]``) that never look at the reference.

    The window-level version of this was useless: a window spans two whole
    cycles, so every window averages to the same phase and the same input
    coverage. These labels are per frame.
    """
    frame_mask = batch["frame_mask"]
    batch_size, frames = frame_mask.shape
    labels: dict[str, list[list[str]]] = {}
    joints = batch["valid_a"].shape[-1]
    observed = (batch["valid_a"].float().sum(-1) + batch["valid_b"].float().sum(-1)) / (2 * joints)
    labels["observed_input"] = [["<80 %" if v < 0.8 else ("80-95 %" if v < 0.95 else ">=95 %") for v in row] for row in observed.tolist()]
    phase = batch.get("phase")
    if phase is not None:
        centre = phase[..., 0] if phase.dim() == 3 else phase
        # phase is the cycle coordinate in [0, 1): 0 and 1 are the turn-arounds, 0.5 mid-swing.
        distance = torch.minimum(centre % 1.0, 1.0 - (centre % 1.0))
        labels["phase_region"] = [["turn-around" if v < 0.15 else ("mid-swing" if v > 0.35 else "between") for v in row] for row in distance.tolist()]
    speed = None
    delta_t = batch.get("delta_t")
    if delta_t is not None:
        motion = torch.zeros_like(frame_mask, dtype=torch.float32)
        motion[:, 1:] = torch.linalg.vector_norm(batch["pose_a"][:, 1:] - batch["pose_a"][:, :-1], dim=-1).mean(-1) / delta_t[:, 1:].clamp_min(1e-4)
        speed = motion
        quantiles = torch.quantile(motion[frame_mask].flatten(), torch.tensor([0.33, 0.66])) if bool(frame_mask.any()) else torch.tensor([0.0, 0.0])
        labels["speed"] = [["slow" if v < float(quantiles[0]) else ("fast" if v > float(quantiles[1]) else "medium") for v in row] for row in speed.tolist()]
    if dataset == "gymnastics":
        labels["cohort"] = [[("elderly" if int(s) >= COHORT_ELDERLY_MIN else "student")] * frames for s in batch["subject_id"]]
    return labels


def stratified(dataset: str, run_dir: Path, *, joints: Sequence[str] | None, device: str = "cuda", folds_dir: Path | None = None, extra: Sequence[str] = ()) -> dict[str, Any]:
    """Model minus rule, per frame, grouped by stratum."""
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    from .evaluate import data_overrides, fold_files
    from .model_rows import fold_checkpoint, load_module, model_predictor

    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for fold in fold_files(dataset, folds_dir):
        cfg = compose_config(data_overrides(dataset, fold, extra))
        datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
        datamodule.setup("test")
        module = load_module(fold_checkpoint(Path(run_dir), fold.stem), device)
        predict_model = model_predictor(module, device, "model")
        predict_rule = model_predictor(module, device, "rule")
        names = list(datamodule.skeleton.joint_names)
        subset = None
        if joints is not None:
            subset = torch.zeros(len(names), dtype=torch.bool)
            subset[[names.index(n) for n in joints]] = True
        for batch in datamodule.test_dataloader():
            frame_mask = batch["frame_mask"]
            reference, reference_valid = batch["reference"], batch["reference_valid"]
            model_pose, model_valid = predict_model(batch)
            rule_pose, rule_valid = predict_rule(batch)
            usable = reference_valid & model_valid & rule_valid & frame_mask[..., None]
            if subset is not None:
                usable = usable & subset
            model_error, ok_m = frame_errors(model_pose, reference, usable)
            rule_error, ok_r = frame_errors(rule_pose, reference, usable)
            ok = (ok_m & ok_r & frame_mask).tolist()
            model_list, rule_list = model_error.tolist(), rule_error.tolist()
            for name, labels in frame_strata(batch, dataset).items():
                for b, row in enumerate(labels):
                    for t, label in enumerate(row):
                        if ok[b][t]:
                            groups[(name, label)].append((model_list[b][t], rule_list[b][t]))
    rows = []
    for (name, label), values in sorted(groups.items()):
        model_values = [1000 * v[0] for v in values]
        rule_values = [1000 * v[1] for v in values]
        differences = [r - m for m, r in zip(model_values, rule_values)]
        rows.append({
            "stratum": name, "level": label, "frames": len(values),
            "model_mm": statistics.fmean(model_values), "rule_mm": statistics.fmean(rule_values),
            "diff_mm": statistics.fmean(differences), "model_better_frames": sum(1 for d in differences if d > 0),
        })
    return {"dataset": dataset, "run": str(run_dir), "strata": rows}


def measurement(dataset: str, run_dir: Path, *, device: str = "cuda", folds_dir: Path | None = None, extra: Sequence[str] = ()) -> dict[str, Any]:
    """Per-cycle range-of-motion and peak angular velocity error against the reference."""
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    from .evaluate import data_overrides, fold_files
    from .model_rows import fold_checkpoint, load_module, model_predictor

    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fold in fold_files(dataset, folds_dir):
        cfg = compose_config(data_overrides(dataset, fold, extra))
        datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
        datamodule.setup("test")
        module = load_module(fold_checkpoint(Path(run_dir), fold.stem), device)
        skeleton = datamodule.skeleton
        predictors = {variant: model_predictor(module, device, variant) for variant in ("model", "rule", "face", "side")}
        for batch in datamodule.test_dataloader():
            frame_mask = batch["frame_mask"]
            reference, reference_valid = batch["reference"], batch["reference_valid"]
            theta_r, theta_r_valid = trunk_twist(reference, reference_valid & frame_mask[..., None], skeleton)
            rom_r, peak_r, cycle_r = per_cycle_extremes(theta_r, theta_r_valid, batch["cycle_index"], batch["delta_t"])
            for variant, predict in predictors.items():
                pose, valid = predict(batch)
                theta, theta_valid = trunk_twist(pose, valid & frame_mask[..., None], skeleton)
                rom, peak, cycle = per_cycle_extremes(theta, theta_valid, batch["cycle_index"], batch["delta_t"])
                usable = cycle & cycle_r & (rom_r > 0.05)
                if not bool(usable.any()):
                    continue
                collected[variant]["rom_error_deg"].extend((180 / math.pi * (rom - rom_r).abs()[usable]).tolist())
                collected[variant]["rom_relative"].extend(((rom / rom_r.clamp_min(1e-6))[usable]).tolist())
                collected[variant]["peak_error_deg_s"].extend((180 / math.pi * (peak - peak_r).abs()[usable]).tolist())
                collected[variant]["peak_relative"].extend(((peak / peak_r.clamp_min(1e-6))[usable]).tolist())
    rows = []
    for variant, metrics in collected.items():
        row: dict[str, Any] = {"variant": variant, "cycles": len(metrics["rom_error_deg"])}
        for key, values in metrics.items():
            row[key] = statistics.fmean(values)
            row[key + "_sd"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return {"dataset": dataset, "run": str(run_dir), "measurement": sorted(rows, key=lambda r: r["variant"])}


def output_root(run_dir: Path) -> Path:
    return PROJECT_ROOT / "local" / "runs" / "external_published" / "analysis" / Path(run_dir).name


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from .evaluate import COMPARISON_JOINTS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--what", default="strata,measurement", help="comma/plus-separated: strata, measurement, failures")
    parser.add_argument("--joints", default="comparison12")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds-dir", type=Path, default=None)
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    joints = None if args.joints in ("all", "") else (list(COMPARISON_JOINTS) if args.joints == "comparison12" else [j for j in args.joints.replace("+", ",").split(",") if j])
    wanted = [w for w in args.what.replace("+", ",").split(",") if w]
    root = output_root(args.run)
    root.mkdir(parents=True, exist_ok=True)
    if "strata" in wanted:
        payload = stratified(args.dataset, args.run, joints=joints, device=args.device, folds_dir=args.folds_dir, extra=list(args.override or []))
        (root / "strata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[analysis] {args.dataset} strata (model - rule, positive = model better):")
        for row in payload["strata"]:
            print(f"   {row['stratum']:16s} {row['level']:12s} n={row['frames']:7d}  model {row['model_mm']:6.2f}  rule {row['rule_mm']:6.2f}  diff {row['diff_mm']:+5.2f}  model better {row['model_better_frames']}/{row['frames']}")
    if "measurement" in wanted:
        payload = measurement(args.dataset, args.run, device=args.device, folds_dir=args.folds_dir, extra=list(args.override or []))
        (root / "measurement.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[analysis] {args.dataset} measurement (trunk twist, per cycle):")
        for row in payload["measurement"]:
            print(f"   {row['variant']:6s} cycles={row['cycles']:5d}  ROM err {row['rom_error_deg']:5.2f}°  ROM ratio {row['rom_relative']:.3f}  peak err {row['peak_error_deg_s']:6.2f}°/s  peak ratio {row['peak_relative']:.3f}")
    if "failures" in wanted:
        from .failure_strata import run as failure_run

        payload = failure_run(args.dataset, args.run, joints=joints, device=args.device, folds_dir=args.folds_dir, extra=list(args.override or []))
        (root / "failures.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[analysis] {args.dataset} real-failure strata (rule - model, positive = model better), {payload['frames']} frames:")
        for row in payload["strata"]:
            print(f"   {row['stratum']:18s} {row['level']:9s} share {100 * row['frame_share']:5.1f} %  model {row['model_mm']:6.2f}  rule {row['rule_mm']:6.2f}"
                  f"  face {row['face_mm']:6.2f}  side {row['side_mm']:6.2f}  diff {row['diff_mm']:+5.2f}  subjects {row['subjects_model_better']}/{row['subjects']}  p {row['wilcoxon_p']:.2g}")
    return 0
