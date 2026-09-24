"""Model, its closed-form rule and the ablations under increasing input corruption.

On clean inputs the learning-free rule is better than the learned model
(gymnastics: 18.0 vs 19.3 mm on the 12 comparison joints), so the learned
parts have to justify themselves where a view degrades. This sweep evaluates
finished checkpoints -- no retraining -- with the test-time corruption of
``src/configs/fusion/corruption/default.yaml`` scaled by a single factor:

    level 0.0   clean (corruption disabled)
    level x     every family's probability multiplied by x, the noise and
                depth magnitudes left at their released values

    python -m fusion external-published corruption --dataset gymnastics \
        --run local/runs/cycle_aware/gymnastics_v1_1_5fold_seed0 \
        --levels 0,0.5,1,2 --variants model,rule,base

Results: ``local/runs/external_published/corruption/<run>/level_<x>_<variant>.json``
(the same summaries the other evaluations write, per-subject values included).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from common.paths import PROJECT_ROOT

# Corruption probabilities scaled by the level; magnitudes stay at the release values.
PROBABILITIES = ("view_probability", "joint_mask_probability", "distal_mask_probability", "gaussian_noise_probability", "depth_probability", "temporal_dropout_probability", "contiguous_dropout_probability")
BASELINE = {"view_probability": 0.5, "joint_mask_probability": 0.05, "distal_mask_probability": 0.3, "gaussian_noise_probability": 0.3, "depth_probability": 0.3, "temporal_dropout_probability": 0.03, "contiguous_dropout_probability": 0.2}


def level_overrides(level: float) -> list[str]:
    """Hydra overrides that set the test-time corruption to ``level`` times the released one."""
    if level <= 0:
        return ["data.test_with_corruption=false"]
    scaled = [f"corruption.{name}={min(1.0, BASELINE[name] * level):.4f}" for name in PROBABILITIES]
    return ["data.test_with_corruption=true", *scaled]


def output_root(run_dir: Path) -> Path:
    return PROJECT_ROOT / "local" / "runs" / "external_published" / "corruption" / Path(run_dir).name


def run_sweep(dataset: str, run_dir: Path, *, levels: Sequence[float], variants: Sequence[str], joints: Sequence[str] | None, device: str = "cuda", folds_dir: Path | None = None, extra: Sequence[str] = ()) -> list[dict[str, Any]]:
    from .evaluate import write_summary
    from .model_rows import evaluate_run

    rows: list[dict[str, Any]] = []
    for level in levels:
        for variant in variants:
            payload = evaluate_run(dataset, Path(run_dir), joints=joints, device=device, folds_dir=folds_dir, extra_overrides=[*level_overrides(level), *extra], variant=variant)
            payload["method"]["corruption_level"] = level
            out = write_summary(output_root(run_dir) / f"level_{level:g}_{variant}.json", payload)
            summary = payload["summary"]
            rows.append({"run": Path(run_dir).name, "level": level, "variant": variant, "pa_mpjpe_mm": 1000 * summary["pa_mpjpe_mean"], "sd_mm": 1000 * summary["pa_mpjpe_sd"], "file": str(out)})
            print(f"[corruption] {dataset} {Path(run_dir).name} level {level:g} {variant:5s}: {rows[-1]['pa_mpjpe_mm']:6.2f} ± {rows[-1]['sd_mm']:4.2f} mm -> {out.name}", flush=True)
    (output_root(run_dir) / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from .evaluate import COMPARISON_JOINTS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--levels", default="0,0.5,1,2", help="comma/plus-separated corruption levels (1 = the released test-time corruption)")
    parser.add_argument("--variants", default="model,rule", help="comma/plus-separated: model, base, rule, face, side")
    parser.add_argument("--joints", default="comparison12")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds-dir", type=Path, default=None)
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    levels = [float(v) for v in args.levels.replace("+", ",").split(",") if v]
    variants = [v for v in args.variants.replace("+", ",").split(",") if v]
    joints = None if args.joints in ("all", "") else (list(COMPARISON_JOINTS) if args.joints == "comparison12" else [j for j in args.joints.replace("+", ",").split(",") if j])
    run_sweep(args.dataset, args.run, levels=levels, variants=variants, joints=joints, device=args.device, folds_dir=args.folds_dir, extra=list(args.override or []))
    return 0
