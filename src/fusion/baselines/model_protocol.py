"""Deterministic fusion rows under the model protocol.

``python -m fusion deterministic`` scores the deterministic matrix in its own
way (per-cycle similarity alignment on all 70 joints against the triangulated
reference of the private data). This module scores the same methods the way
every learned run is scored -- the folds, phase-normalised windows and
per-frame Procrustes PA-MPJPE of ``docs/cycle_aware_fusion.md`` §1.5 -- so a
deterministic rule and a trained model can stand in one table, on any dataset
that has folds and a reference.

Each method is expressed as a trial transform: it fuses the two world-frame
MHR70 views and writes the result into both views of the trial, after which
the standard DataModule canonicalises and windows it and
:func:`fusion.external.published.evaluate.evaluate_folds` reports the error.

    python -m fusion deterministic-protocol --dataset fit3d
    python -m fusion deterministic-protocol --dataset gymnastics --methods avg_body_current avg_body_depthaware
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from common.paths import PROJECT_ROOT
from fusion.keypoints.schema import PosePairTrial

from .methods import (
    BASELINE_METHODS,
    DEPTH_AWARE_ALPHAS,
    NO_EXTRINSIC_METHODS,
    STABLE_SIM3_JOINTS,
    bodypart_weights,
    current_body_average,
    depth_aware_body_average,
    fuse_weighted,
    root_align_to_reference,
    sim3_align_to_reference,
    smooth_sequence,
)

# Every method that needs neither camera extrinsics nor the evaluation
# reference. ``sim3_face_stable_joint_weight`` derives its weights from the
# triangulated reference it is then scored against, so it is a leakage
# diagnostic and stays out of this table.
PROTOCOL_METHODS: tuple[str, ...] = tuple(m for m in NO_EXTRINSIC_METHODS if m != "sim3_face_stable_joint_weight") + BASELINE_METHODS

VIEW_ROWS: tuple[str, ...] = ("view_a", "view_b")
"""Single-view rows: the unfused input of one camera."""

DEFAULT_OUT_DIR = Path("local/runs/fuse_depthaware")


def fuse_views(method: str, face: np.ndarray, side: np.ndarray, *, fps: float) -> np.ndarray:
    """Fuse two synchronised world-frame MHR70 sequences with one deterministic rule.

    Args:
        method: A name of :data:`PROTOCOL_METHODS` or of :data:`VIEW_ROWS`.
        face: ``[T, 70, 3]`` view A in its own camera frame.
        side: ``[T, 70, 3]`` view B in its own camera frame.
        fps: Sampling rate, needed by the temporal baselines.

    Returns:
        ``[T, 70, 3]`` fused sequence in view A's frame.

    Raises:
        ValueError: If the method is unknown here.
    """
    face = np.asarray(face, dtype=np.float32)
    side = np.asarray(side, dtype=np.float32)
    if method == "view_a":
        return face
    if method == "view_b":
        return side
    if method == "avg_body_current":
        return current_body_average(face, side)
    if method in DEPTH_AWARE_ALPHAS:
        alpha_face, alpha_side = DEPTH_AWARE_ALPHAS[method]
        return depth_aware_body_average(face, side, alpha_face, alpha_side)
    if method == "avg_world_face_ref":
        return 0.5 * (face + side)
    if method == "root_face_stable":
        return 0.5 * (face + root_align_to_reference(side, face))
    if method in BASELINE_METHODS:
        from .classical_baselines import fuse_baseline

        fused, _ = fuse_baseline(method, face, side, fps=float(fps))
        return fused
    if method.startswith("sim3_face_"):
        joints = tuple(range(face.shape[1])) if method == "sim3_face_all" else STABLE_SIM3_JOINTS
        aligned, _ = sim3_align_to_reference(side, face, joints)
        if method == "sim3_face_stable_bodypart_weight":
            return fuse_weighted(face, aligned, bodypart_weights(face.shape[1]))
        if method == "sim3_face_stable_smooth_transform":
            return 0.5 * (face + smooth_sequence(aligned, win=5))
        if method == "sim3_face_stable_smooth_kpt":
            return smooth_sequence(0.5 * (face + aligned), win=5)
        return 0.5 * (face + aligned)
    raise ValueError(f"unsupported deterministic method for the model protocol: {method!r}")


class DeterministicTrialTransform:
    """Replace both views of a trial by one deterministic rule's fused pose.

    Writing the result into both views keeps the contract of the evaluator
    (which reads view A) and of the DataModule (which canonicalises each view
    independently); the fused pose lives in view A's frame, so canonicalising
    it twice is the same operation.

    Attributes:
        method: The rule applied to every trial.
    """

    def __init__(self, method: str) -> None:
        self.method = str(method)

    def __call__(self, trial: PosePairTrial) -> PosePairTrial:
        fused = np.asarray(fuse_views(self.method, trial.face, trial.side, fps=float(trial.fps)), dtype=np.float32)
        finite = np.isfinite(fused).all(axis=-1)
        if self.method == "view_a":
            valid = np.asarray(trial.valid_face, dtype=bool) & finite
        elif self.method == "view_b":
            valid = np.asarray(trial.valid_side, dtype=bool) & finite
        else:
            valid = (np.asarray(trial.valid_face, dtype=bool) | np.asarray(trial.valid_side, dtype=bool)) & finite
        fused = np.where(valid[..., None], fused, 0.0).astype(np.float32)
        return replace(
            trial,
            face=fused,
            side=fused.copy(),
            valid_face=valid,
            valid_side=valid.copy(),
            source_metadata={**dict(trial.source_metadata), "deterministic_method": self.method},
        )


def evaluate_methods(
    dataset: str,
    methods: Sequence[str],
    *,
    folds_dir: Path | None = None,
    joints: Sequence[str] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, dict]:
    """Score every method on every fold of ``dataset`` and write one summary each."""
    from fusion.external.published.evaluate import evaluate_folds, write_summary

    suffix = "" if joints is None else f"_{len(list(joints))}joints"
    results: dict[str, dict] = {}
    for method in methods:
        payload = evaluate_folds(dataset, lambda _fold, m=method: DeterministicTrialTransform(m), folds_dir=folds_dir, joints=joints)
        payload["method"] = method
        payload["protocol"] = "model"
        path = write_summary(Path(out_dir) / dataset / f"summary_{method}{suffix}.json", payload)
        summary = payload["summary"]
        print(f"{method:36s} {summary['pa_mpjpe_mean'] * 1000:7.2f} +- {summary['pa_mpjpe_sd'] * 1000:4.2f} mm  ({summary['folds']} folds, {len(summary['joint_names'])} joints) -> {path}")
        results[method] = payload
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fusion deterministic-protocol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    parser.add_argument("--methods", nargs="*", default=list(VIEW_ROWS + PROTOCOL_METHODS), help="default: the single views and every leakage-free deterministic method")
    parser.add_argument("--folds-dir", type=Path, default=None, help="fold directory (default: the dataset's own)")
    parser.add_argument("--joints", default="all", help="'all' or 'comparison12' (the joint set of the external-baseline table)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    from fusion.external.published.evaluate import COMPARISON_JOINTS

    joints = None if args.joints == "all" else COMPARISON_JOINTS
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    evaluate_methods(args.dataset, args.methods, folds_dir=args.folds_dir, joints=joints, out_dir=out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
