"""What each method costs: parameters, inference time and what it needs to exist.

Accuracy is only half of a comparison -- the closed-form rule has no
parameters and no training at all, while the supervised baselines need 3D
ground truth that the private recordings do not have. This builds that
table: parameter counts from the actual model objects, single-window
inference time measured on the current device, and the requirements each
method's recipe imposes (3D labels, camera calibration, pretrained weights,
per-fold training).

    python -m fusion external-published cost [--device cuda] [--repeats 20]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from common.paths import PROJECT_ROOT

# What each method's *published* recipe requires, independent of our protocol.
REQUIREMENTS: dict[str, dict[str, Any]] = {
    "ours (closed-form rule)": {"3d_labels": False, "calibration": False, "pretrained": False, "training": False},
    "ours (v1.1 model)": {"3d_labels": False, "calibration": False, "pretrained": False, "training": "per fold, 50 epochs"},
    "CanonPose": {"3d_labels": False, "calibration": False, "pretrained": False, "training": "per fold, 100 epochs"},
    "MetaPose (S1)": {"3d_labels": False, "calibration": False, "pretrained": "monocular lifter for the init", "training": False},
    "MetaPose (S2)": {"3d_labels": False, "calibration": False, "pretrained": "monocular lifter for the init", "training": "per fold, staged"},
    "MHFormer": {"3d_labels": True, "calibration": "reference in camera frame", "pretrained": False, "training": "per fold, 19 epochs"},
    "MDVPose": {"3d_labels": True, "calibration": "reference in camera frame", "pretrained": "MotionBERT H36M", "training": "per fold, 30 epochs"},
    "VideoPose3D (trained)": {"3d_labels": True, "calibration": "reference in camera frame", "pretrained": False, "training": "per fold, 80 epochs"},
    "VideoPose3D (zero-shot)": {"3d_labels": "H36M, at pretraining", "calibration": False, "pretrained": "H36M checkpoint", "training": False},
}


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def time_forward(callable_fn, inputs, *, repeats: int = 20, warmup: int = 3) -> float:
    """Median wall-clock seconds of one call."""
    import torch

    for _ in range(warmup):
        callable_fn(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_fn(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def measure(device: str = "cuda", *, repeats: int = 20, frames: int = 128) -> list[dict[str, Any]]:
    """Parameters and per-frame inference time of every method's network."""
    import torch

    rows: list[dict[str, Any]] = []
    target = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    def add(name: str, parameters: int, seconds_per_window: float | None, window_frames: int) -> None:
        rows.append({
            "method": name, "parameters_m": parameters / 1e6,
            "ms_per_frame": 1000 * seconds_per_window / window_frames if seconds_per_window is not None else None,
            "window_frames": window_frames, **REQUIREMENTS.get(name, {}),
        })

    # Ours: the fusion model, and its closed form with the learned parts removed.
    from fusion.train import build_module, compose_config

    cfg = compose_config(["data=gymnastics", "data.fold_json=null"])
    module = build_module(cfg).to(target).eval()
    joints = module.skeleton.num_joints if hasattr(module.skeleton, "num_joints") else len(module.skeleton.joint_names)
    batch = {
        "pose_a": torch.randn(1, frames, joints, 3, device=target), "pose_b": torch.randn(1, frames, joints, 3, device=target),
        "valid_a": torch.ones(1, frames, joints, dtype=torch.bool, device=target), "valid_b": torch.ones(1, frames, joints, dtype=torch.bool, device=target),
        "delta_t": torch.full((1, frames), 1 / 30, device=target), "phase": torch.rand(1, frames, device=target),
        "phase_valid": torch.ones(1, frames, dtype=torch.bool, device=target), "frame_mask": torch.ones(1, frames, dtype=torch.bool, device=target),
    }
    with torch.no_grad():
        add("ours (v1.1 model)", count_parameters(module.model), time_forward(lambda b: module(b), (batch,), repeats=repeats), frames)
        half = torch.full_like(batch["pose_a"][..., :1], 0.5)
        rule_args = (batch["pose_a"], batch["pose_b"], half, half, batch["valid_a"], batch["valid_b"], None, None)
        add("ours (closed-form rule)", 0, time_forward(lambda *a: module.model.fuse_base(*a), rule_args, repeats=repeats), frames)

    # External networks (same input sizes their own inference uses).
    try:
        from .canonpose import _load_lifter_class

        lifter = _load_lifter_class()().to(target).eval()
        with torch.no_grad():
            add("CanonPose", count_parameters(lifter), time_forward(lambda p, c: lifter(p, c), (torch.randn(frames, 32, device=target), torch.ones(frames, 16, device=target)), repeats=repeats), frames)
    except Exception as error:
        rows.append({"method": "CanonPose", "error": str(error), **REQUIREMENTS["CanonPose"]})
    try:
        from .mhformer import CONFIG as MH, build_model as build_mhformer

        model = build_mhformer().to(target).eval()
        with torch.no_grad():
            add("MHFormer", count_parameters(model), time_forward(lambda x: model(x), (torch.randn(1, MH["frames"], 17, 2, device=target),), repeats=repeats), 1)
    except Exception as error:
        rows.append({"method": "MHFormer", "error": str(error), **REQUIREMENTS["MHFormer"]})
    try:
        from .mdvpose import CONFIG as MD, build_model as build_mdvpose

        model = build_mdvpose().to(target).eval()
        with torch.no_grad():
            add("MDVPose", count_parameters(model), time_forward(lambda x: model(x), (torch.randn(1, MD["clip_len"], 17, 3, device=target),), repeats=repeats), MD["clip_len"])
    except Exception as error:
        rows.append({"method": "MDVPose", "error": str(error), **REQUIREMENTS["MDVPose"]})
    try:
        from .videopose3d_train import build_model as build_vp3d, receptive_field

        model = build_vp3d().to(target).eval()
        with torch.no_grad():
            add("VideoPose3D (trained)", count_parameters(model), time_forward(lambda x: model(x), (torch.randn(1, receptive_field(), 17, 2, device=target),), repeats=repeats), 1)
    except Exception as error:
        rows.append({"method": "VideoPose3D (trained)", "error": str(error), **REQUIREMENTS["VideoPose3D (trained)"]})
    rows.append({"method": "MetaPose (S1)", "parameters_m": 0.0, "ms_per_frame": None, "note": "iterative optimisation, 100 Adam steps per frame batch", **REQUIREMENTS["MetaPose (S1)"]})
    return rows


def output_path() -> Path:
    return PROJECT_ROOT / "local" / "runs" / "external_published" / "cost" / "cost_table.json"


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = measure(args.device, repeats=args.repeats)
    out = output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": args.device, "rows": rows}, indent=2), encoding="utf-8")
    print(f"{'method':26s} {'params (M)':>10s} {'ms/frame':>9s}  3D labels  pretrained            training")
    for row in rows:
        params = f"{row['parameters_m']:.2f}" if row.get("parameters_m") is not None else "--"
        ms = f"{row['ms_per_frame']:.3f}" if row.get("ms_per_frame") else "--"
        print(f"{row['method']:26s} {params:>10s} {ms:>9s}  {str(row.get('3d_labels')):9s}  {str(row.get('pretrained')):20s}  {row.get('training')}")
    print(f"-> {out}")
    return 0
