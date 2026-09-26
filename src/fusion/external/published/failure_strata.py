"""Does the learned model beat its closed-form rule where SAM3D really fails?

The corruption sweep shows that the residual recovers *synthetic* damage.
Real SAM3D output is never missing, so its failures are wrong estimates:
a view that swaps left and right, a limb flipped in depth, a view that is
simply much worse than the other one. This analysis finds such frames and
compares the model with the rule on them, per frame and per subject.

Strata (one record per test frame, 12 comparison joints by default):

* ``view_disagreement`` -- label-free: mean distance between the two
  canonicalised input views, binned by the dataset's own percentiles
  (0-50, 50-90, 90-99, 99-100). The only stratum usable at deployment.
* ``worse_view_error`` -- the per-frame PA error of the worse input view,
  binned by percentile (needs the reference).
* ``lr_swap`` -- a view whose error drops by more than 20 % when its left
  and right labels are exchanged (whole body, arms or legs).
* ``gross_limb`` -- after aligning a view to the reference, one of its limb
  segments points more than 60 degrees away from the reference segment
  (depth flips and gross limb errors): none / face / side / both.
* ``view_ratio`` -- the worse view's error is more than twice the better one's.

Each stratum level reports frame means of the model, the rule and both
views, the frames the model wins, and a Wilcoxon test over the subjects with
at least ``MIN_FRAMES`` frames in that level (per-subject mean difference).

    python -m fusion external-published analysis --what failures --dataset freeman \\
        --run local/runs/cycle_aware/freeman_rep_v1_1_equal_reliability_5fold_seed0

Writes ``local/runs/external_published/analysis/<run>/failures.json``.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from fusion.metrics import procrustes_align

from .analysis_rows import frame_errors

SWAP_GAIN = 0.8          # swapped-label error below 80 % of the original -> swap
GROSS_LIMB_DEGREES = 60.0
VIEW_RATIO = 2.0
PERCENTILE_BINS = (0.5, 0.9, 0.99)
PERCENTILE_LABELS = ("0-50 %", "50-90 %", "90-99 %", "99-100 %")
MIN_FRAMES = 10


def _swap(pose: torch.Tensor, pairs: Sequence[tuple[int, int]]) -> torch.Tensor:
    swapped = pose.clone()
    for left, right in pairs:
        swapped[..., left, :], swapped[..., right, :] = pose[..., right, :], pose[..., left, :]
    return swapped


def _max_limb_angle(pose: torch.Tensor, reference: torch.Tensor, usable: torch.Tensor, bones: Sequence[tuple[int, int]]) -> torch.Tensor:
    """``[B, T]`` largest angle (degrees) between aligned and reference limb segments."""
    batch, frames, joints, _ = pose.shape
    aligned = procrustes_align(pose.reshape(-1, joints, 3).float(), reference.reshape(-1, joints, 3).float(), usable.reshape(-1, joints)).reshape(batch, frames, joints, 3)
    worst = torch.zeros(batch, frames)
    for start, end in bones:
        ok = usable[..., start] & usable[..., end]
        a = aligned[..., end, :] - aligned[..., start, :]
        r = reference[..., end, :].float() - reference[..., start, :].float()
        cosine = (a * r).sum(-1) / (torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(r, dim=-1)).clamp_min(1e-8)
        angle = torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))
        worst = torch.where(ok, torch.maximum(worst, angle), worst)
    return worst


def collect(dataset: str, run_dir: Path, *, joints: Sequence[str] | None, device: str = "cuda", folds_dir: Path | None = None, extra: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One record per usable test frame of every fold."""
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.train import compose_config

    from .evaluate import data_overrides, fold_files
    from .model_rows import fold_checkpoint, load_module, model_predictor

    records: list[dict[str, Any]] = []
    for fold in fold_files(dataset, folds_dir):
        cfg = compose_config(data_overrides(dataset, fold, extra))
        datamodule = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
        datamodule.setup("test")
        module = load_module(fold_checkpoint(Path(run_dir), fold.stem), device)
        predict = {name: model_predictor(module, device, name) for name in ("model", "rule", "face", "side")}
        skeleton = datamodule.skeleton
        names = list(skeleton.joint_names)
        subset = torch.ones(len(names), dtype=torch.bool)
        if joints is not None:
            subset = torch.zeros(len(names), dtype=torch.bool)
            subset[[names.index(n) for n in joints]] = True
        pairs = [(l, r) for l, r in skeleton.left_right_pairs if subset[l] and subset[r]]
        arms = [(l, r) for l, r in pairs if any(k in names[l] for k in ("shoulder", "elbow", "wrist"))]
        legs = [(l, r) for l, r in pairs if any(k in names[l] for k in ("hip", "knee", "ankle"))]
        bones = [(a, b) for a, b in skeleton.bones if subset[a] and subset[b] and names[a].split("-")[0] in ("left", "right")]
        for batch in datamodule.test_dataloader():
            frame_mask = batch["frame_mask"]
            reference, reference_valid = batch["reference"], batch["reference_valid"]
            poses = {name: fn(batch) for name, fn in predict.items()}
            usable = reference_valid & frame_mask[..., None] & subset
            for pose, valid in poses.values():
                usable = usable & valid
            errors = {name: frame_errors(pose, reference, usable) for name, (pose, _) in poses.items()}
            ok = frame_mask.clone()
            for _, good in errors.values():
                ok = ok & good
            swap_flags, limb_angles = {}, {}
            for view in ("face", "side"):
                pose = poses[view][0]
                base = errors[view][0]
                best = torch.full_like(base, float("inf"))
                for group in (pairs, arms, legs):
                    if group:
                        best = torch.minimum(best, frame_errors(_swap(pose, group), reference, usable)[0])
                swap_flags[view] = best < SWAP_GAIN * base
                limb_angles[view] = _max_limb_angle(pose, reference, usable, bones)
            # Label-free disagreement of the canonicalised inputs (same frame, same units).
            gap = torch.linalg.vector_norm(batch["pose_a"].float() - batch["pose_b"].float(), dim=-1)
            gap = torch.where(usable, gap, torch.zeros_like(gap)).sum(-1) / usable.sum(-1).clamp_min(1)
            subjects = batch["subject_id"]
            for b in range(frame_mask.shape[0]):
                for t in range(frame_mask.shape[1]):
                    if not bool(ok[b, t]):
                        continue
                    records.append({
                        "fold": fold.stem, "subject": str(subjects[b].item() if torch.is_tensor(subjects[b]) else subjects[b]),
                        **{f"{name}_mm": 1000 * float(errors[name][0][b, t]) for name in errors},
                        "gap": float(gap[b, t]),
                        "swap_face": bool(swap_flags["face"][b, t]), "swap_side": bool(swap_flags["side"][b, t]),
                        "limb_face": float(limb_angles["face"][b, t]), "limb_side": float(limb_angles["side"][b, t]),
                    })
    return records


def _percentile_labels(values: list[float]) -> list[str]:
    ordered = sorted(values)
    cuts = [ordered[min(len(ordered) - 1, int(q * len(ordered)))] for q in PERCENTILE_BINS]
    out = []
    for v in values:
        index = sum(1 for c in cuts if v >= c)
        out.append(PERCENTILE_LABELS[index])
    return out


def label(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    worse = [max(r["face_mm"], r["side_mm"]) for r in records]
    labels = {
        "view_disagreement": _percentile_labels([r["gap"] for r in records]),
        "worse_view_error": _percentile_labels(worse),
        "lr_swap": ["face" if r["swap_face"] and not r["swap_side"] else "side" if r["swap_side"] and not r["swap_face"] else "both" if r["swap_face"] else "none" for r in records],
        "gross_limb": [
            {(False, False): "none", (True, False): "face", (False, True): "side", (True, True): "both"}[(r["limb_face"] > GROSS_LIMB_DEGREES, r["limb_side"] > GROSS_LIMB_DEGREES)]
            for r in records
        ],
        "view_ratio": [f"> {VIEW_RATIO:g}x" if max(r["face_mm"], r["side_mm"]) > VIEW_RATIO * min(r["face_mm"], r["side_mm"]) else f"<= {VIEW_RATIO:g}x" for r in records],
    }
    return labels


def summarise(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scipy.stats import wilcoxon

    rows = []
    for stratum, levels in label(records).items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record, level in zip(records, levels):
            groups[level].append(record)
        for level in sorted(groups):
            members = groups[level]
            diffs = [r["rule_mm"] - r["model_mm"] for r in members]
            by_subject: dict[str, list[float]] = defaultdict(list)
            for r, d in zip(members, diffs):
                by_subject[r["subject"]].append(d)
            subject_means = [statistics.fmean(v) for v in by_subject.values() if len(v) >= MIN_FRAMES]
            p = math.nan
            if len(subject_means) >= 5 and any(abs(v) > 0 for v in subject_means):
                p = float(wilcoxon(subject_means).pvalue)
            rows.append({
                "stratum": stratum, "level": level, "frames": len(members), "frame_share": len(members) / len(records),
                **{f"{name}_mm": statistics.fmean(r[f"{name}_mm"] for r in members) for name in ("model", "rule", "face", "side")},
                "diff_mm": statistics.fmean(diffs), "model_better_frames": sum(1 for d in diffs if d > 0),
                "subjects": len(subject_means), "subjects_model_better": sum(1 for v in subject_means if v > 0),
                "subject_mean_diff_mm": statistics.fmean(subject_means) if subject_means else math.nan, "wilcoxon_p": p,
            })
    return rows


def run(dataset: str, run_dir: Path, *, joints: Sequence[str] | None, device: str = "cuda", folds_dir: Path | None = None, extra: Sequence[str] = ()) -> dict[str, Any]:
    records = collect(dataset, run_dir, joints=joints, device=device, folds_dir=folds_dir, extra=extra)
    return {
        "dataset": dataset, "run": str(run_dir), "frames": len(records),
        "thresholds": {"swap_gain": SWAP_GAIN, "gross_limb_degrees": GROSS_LIMB_DEGREES, "view_ratio": VIEW_RATIO, "min_frames_per_subject": MIN_FRAMES},
        "strata": summarise(records),
    }
