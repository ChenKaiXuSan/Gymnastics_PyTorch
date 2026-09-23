"""Score the cycle-aware model through the external-baseline evaluator.

The external rows are computed on the joints each published method's
skeleton covers (12-14 of the 20 major joints), so a comparison table needs
the model on exactly the same joints, windows and folds. This module runs a
finished sweep's per-fold checkpoints through
:func:`fusion.external.published.evaluate.evaluate_folds` with a predictor
that returns the model's fused pose instead of view A, so every number in
the table comes out of the same code path.

    python -m fusion external-published model --dataset gymnastics \
        --run local/runs/cycle_aware/gymnastics_v1_1_5fold_seed0 [--joints comparison12]

``--checkpoint last`` (default) matches what the sweep reports: ``run_fold``
calls ``trainer.test`` without ``ckpt_path``, i.e. on the final-epoch
weights that ``last.ckpt`` holds; ``--checkpoint best`` uses the
``val/total`` checkpoint instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from common.paths import PROJECT_ROOT


def fold_checkpoint(run_dir: Path, fold: str, *, which: str = "last") -> Path:
    """The fold's checkpoint: ``last.ckpt`` (what the sweep tested) or the best ``val/total`` one."""
    directory = Path(run_dir) / fold / "checkpoints"
    if not directory.is_dir():
        raise FileNotFoundError(f"no checkpoints for {fold} in {run_dir}")
    if which == "last":
        path = directory / "last.ckpt"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; use --checkpoint best")
        return path
    candidates = sorted(p for p in directory.glob("*.ckpt") if p.name != "last.ckpt")
    if not candidates:
        raise FileNotFoundError(f"no monitored checkpoint in {directory}")
    return candidates[-1]


def load_module(checkpoint: Path, device: str = "cuda"):
    """The Lightning module with the checkpoint's own architecture, in eval mode."""
    from fusion.lightning_module import CycleAwareFusionModule

    module = CycleAwareFusionModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
    module.to(torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")).eval()
    return module


VARIANTS = ("model", "base", "rule", "face", "side")


def model_predictor(module, device: str = "cuda", variant: str = "model"):
    """``batch -> (pose, valid)`` on the CPU, like the evaluator's other predictions.

    Variants (the quantities the Lightning module also logs):
        model  the fused output,
        base   its closed-form base pose with the *learned* reliability weights,
        rule   the same closed form with equal weights -- the learning-free rule,
        face / side  the two input views.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    target = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    def predict(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if variant == "face":
            return batch["pose_a"], batch["valid_a"]
        if variant == "side":
            return batch["pose_b"], batch["valid_b"]
        moved = {k: (v.to(target) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            output = module(moved)
            if variant == "rule":
                frame_mask = moved["frame_mask"][..., None]
                half = torch.full_like(moved["pose_a"][..., :1], 0.5)
                pose = module.model.fuse_base(moved["pose_a"], moved["pose_b"], half, half, moved["valid_a"] & frame_mask, moved["valid_b"] & frame_mask, moved.get("depth_a"), moved.get("depth_b"))
                pose = pose[0] if isinstance(pose, tuple) else pose
            else:
                pose = output.pose if variant == "model" else output.base_pose
        return pose.float().cpu(), output.valid.cpu()

    return predict


def evaluate_run(dataset: str, run_dir: Path, *, joints: Sequence[str] | None = None, which: str = "last", device: str = "cuda", folds_dir: Path | None = None, extra_overrides: Sequence[str] = (), variant: str = "model") -> dict[str, Any]:
    from .evaluate import evaluate_folds

    modules: dict[str, Any] = {}

    def predictor_factory(fold: Path):
        if fold.stem not in modules:
            modules[fold.stem] = load_module(fold_checkpoint(Path(run_dir), fold.stem, which=which), device)
        return model_predictor(modules[fold.stem], device, variant)

    payload = evaluate_folds(dataset, lambda fold: None, folds_dir=folds_dir, extra_overrides=extra_overrides, joints=joints, predictor_factory=predictor_factory)
    payload["method"] = {"name": f"cycle_aware_{variant}", "run": str(run_dir), "checkpoint": which, "variant": variant}
    return payload


def output_path(run_dir: Path, joints: Sequence[str] | None, variant: str = "model") -> Path:
    suffix = "all_joints" if not joints else f"{len(list(joints))}joints"
    root = "model" if variant == "model" else "internal"
    return PROJECT_ROOT / "local" / "runs" / "external_published" / root / Path(run_dir).name / f"summary_{variant}_{suffix}.json"


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    from .evaluate import COMPARISON_JOINTS, write_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    parser.add_argument("--run", type=Path, required=True, help="sweep directory with fold_XX/checkpoints")
    parser.add_argument("--joints", default="comparison12", help="'comparison12', 'all', or comma/plus-separated joint names")
    parser.add_argument("--checkpoint", default="last", choices=("last", "best"))
    parser.add_argument("--variant", default="model", choices=VARIANTS, help="model | base (learned weights) | rule (equal weights) | face | side")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds-dir", type=Path, default=None)
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.joints == "all":
        joints = None
    elif args.joints == "comparison12":
        joints = list(COMPARISON_JOINTS)
    else:
        joints = [j for j in args.joints.replace("+", ",").split(",") if j]
    payload = evaluate_run(args.dataset, args.run, joints=joints, which=args.checkpoint, device=args.device, folds_dir=args.folds_dir, extra_overrides=list(args.override or []), variant=args.variant)
    out = write_summary(output_path(args.run, joints, args.variant), payload)
    s = payload["summary"]
    print(f"[{args.variant}] {args.dataset} {Path(args.run).name} ({args.checkpoint}): PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    print("  per fold: " + ", ".join(f"{1000 * f['pa_mpjpe']:.1f}" for f in payload["folds"]))
    return 0
