"""Judgment experiment: accuracy under single-view *silent* corruption.

Injects the five validity-preserving corruption families into ONE view of the
canonical streams (the other view stays clean), runs deterministic (A2, A3)
and learned (A4, A5, A6) fusion on identical inputs, and scores every output
against the triangulated pseudo-reference with the paper's protocol (one
similarity transform per cycle via `external_metrics_from_reference`).

Pre-registered criterion: learned fusion beats deterministic fusion on
corrupted-input accuracy in the silent single-view regime.
Cohort: the 27 validation people of the frozen 96/27/14 split (validation-
scoped diagnostic, consistent with the manuscript's corruption reporting).
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time, zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from common.paths import PROJECT_ROOT as REPO  # noqa: E402

os.environ.setdefault("GYMNASTICS_DATA_ROOT", "/work/1/HP260146/chenkaixu/gymnastics")

from fusion.keypoints.config import load_skeleton_spec
from fusion.archive.rotation_aware.corruptions import CorruptionConfig, apply_corruptions
from fusion.keypoints.data import load_cached_trial, resolve_cache_manifest
from fusion.archive.rotation_aware.evaluation import (
    MethodSequence, external_metrics_from_reference, load_triangulated_references,
)
from fusion.archive.rotation_aware.inference import canonicalize_trial, overlap_taper, _starts, _forward
from fusion.archive.rotation_aware.model import RotationAwareFusionModel
from fusion.archive.rotation_aware.cli import model_kwargs_for_training
from fusion.archive.rotation_aware.training import load_checkpoint

RUNS = REPO / "local/runs/fuse_rotation_aware"
TRI_ROOT = Path(os.environ["GYMNASTICS_DATA_ROOT"]) / "sam3d_triangulated/person"
SILENT = ("spike_noise", "random_walk_drift", "thorax_rotation_bias", "freeze_segment", "integer_time_shift")
WINDOW, STRIDE = 128, 64

def severe(cfg: CorruptionConfig, family: str) -> CorruptionConfig:
    table = {
        "spike_noise": dict(spike_probability=0.06, spike_scale=0.30),
        "random_walk_drift": dict(drift_probability=0.30, drift_scale=0.03),
        "thorax_rotation_bias": dict(rotation_probability=0.30, rotation_degrees=36.0),
        "freeze_segment": dict(freeze_probability=0.30, freeze_length=36),
        "integer_time_shift": dict(time_shift_probability=0.30, max_time_shift=8),
    }
    return replace(cfg, **table[family])

def load_model(ablation: str, skeleton):
    ckpt = RUNS / f"runs/all137_{ablation.lower()}_e100_seed0/checkpoints/best.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = RotationAwareFusionModel(skeleton, **model_kwargs_for_training(payload["training_config"]))
    load_checkpoint(ckpt, model)
    model.eval()
    return model

def windowed_outputs(models, face, side, vf, vs, timestamps, fps, skeleton):
    """OLA fused per model + deterministic base, mirroring inference semantics."""
    frames, joints = face.shape[:2]
    taper = overlap_taper(WINDOW)
    sums = {name: np.zeros((frames, joints, 3)) for name in models}
    sums["A3"] = np.zeros((frames, joints, 3))
    weights = {name: np.zeros((frames, joints)) for name in list(models) + ["A3"]}
    for start in _starts(frames, WINDOW, STRIDE):
        end = min(start + WINDOW, frames); count = end - start
        f = torch.zeros((1, WINDOW, joints, 3)); s = torch.zeros_like(f)
        wf = torch.zeros((1, WINDOW, joints), dtype=torch.bool); ws = torch.zeros_like(wf)
        f[:, :count] = torch.from_numpy(face[start:end]); s[:, :count] = torch.from_numpy(side[start:end])
        wf[:, :count] = torch.from_numpy(vf[start:end]); ws[:, :count] = torch.from_numpy(vs[start:end])
        dt = torch.zeros((1, WINDOW)); dt[0, :count] = 1.0 / fps
        if count > 1:
            dt[0, 1:count] = torch.from_numpy(np.diff(timestamps[start:end]).astype(np.float32))
        base_done = False
        for name, model in models.items():
            with torch.no_grad():
                out, _, _ = _forward(model, f, s, wf, ws, skeleton, dt)
            valid = out.valid[0, :count].cpu().numpy()
            w = taper[:count, None] * valid
            sums[name][start:end] += out.fused_kpts[0, :count].cpu().numpy() * w[..., None]
            weights[name][start:end] += w
            if not base_done:
                sums["A3"][start:end] += out.base_kpts[0, :count].cpu().numpy() * w[..., None]
                weights["A3"][start:end] += w
                base_done = True
    result = {}
    for name in sums:
        w = weights[name]
        pts = (sums[name] / np.maximum(w[..., None], 1e-12)).astype(np.float32)
        pts[w == 0] = 0
        result[name] = (pts, w > 0)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=0, help="limit people for timing (0=all)")
    ap.add_argument("--tiers", default="default,severe")
    ap.add_argument("--out", default=str(REPO / "local/runs/analysis/single_view_silent_corruption"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    skeleton = load_skeleton_spec(REPO / "src/configs/shared/skeleton_mhr70.yaml")
    val_people = json.load(open(REPO / "src/configs/shared/folds/paper_137_a6_split.json"))["val"]
    if args.people: val_people = val_people[: args.people]
    models = {a: load_model(a, skeleton) for a in ("A4", "A5", "A6")}
    tiers = args.tiers.split(",")
    rows = []
    t0 = time.time()
    for pi, pid in enumerate(val_people):
        cache = RUNS / "cache" / f"person_{pid}"
        payload_dir, _ = resolve_cache_manifest(cache)
        trial_ids = sorted(p.stem for p in payload_dir.glob("*.npz"))
        for tid in trial_ids:
            try:
                trial, _ = load_cached_trial(cache, tid)
            except Exception as e:
                print(f"skip {pid}/{tid}: {e}"); continue
            canon = canonicalize_trial(trial, skeleton)
            source = canon.trial
            T = source.face.shape[0]
            probe = MethodSequence("probe", np.zeros((T, len(skeleton.joint_names), 3), np.float32),
                                  trial.timestamps, trial_id=tid,
                                  face_map=trial.face_map, side_map=trial.side_map)
            ref = load_triangulated_references(TRI_ROOT, pid, [probe]).get(tid)
            if ref is None:
                continue
            cf = np.ascontiguousarray(source.face, dtype=np.float32)
            cs = np.ascontiguousarray(source.side, dtype=np.float32)
            vf = np.ascontiguousarray(source.valid_face); vs = np.ascontiguousarray(source.valid_side)
            conditions = [("clean", "-", "-")]
            for tier in tiers:
                for fam in SILENT:
                    for view in ("face", "side"):
                        conditions.append((tier, fam, view))
            for tier, fam, view in conditions:
                if tier == "clean":
                    face_in, side_in = cf, cs
                else:
                    cfg = CorruptionConfig(enabled_families=(fam,))
                    if tier == "severe": cfg = severe(cfg, fam)
                    seed = zlib.crc32(f"{pid}/{tid}/{fam}/{view}/{tier}".encode()) & 0x7FFFFFFF
                    batch = apply_corruptions(
                        torch.from_numpy(cf.copy()), torch.from_numpy(cs.copy()),
                        torch.from_numpy(vf.copy()), torch.from_numpy(vs.copy()),
                        seed=seed, config=cfg, skeleton=skeleton)
                    assert bool((batch.corrupted_valid_face == torch.from_numpy(vf)).all())
                    assert bool((batch.corrupted_valid_side == torch.from_numpy(vs)).all())
                    if view == "face":
                        face_in, side_in = batch.corrupted_face.numpy(), cs
                    else:
                        face_in, side_in = cf, batch.corrupted_side.numpy()
                outputs = windowed_outputs(models, face_in, side_in, vf, vs, trial.timestamps, trial.fps, skeleton)
                # A2 arithmetic on the same inputs
                a2_valid = vf | vs
                wsum = vf.astype(np.float32) + vs.astype(np.float32)
                a2 = (face_in * vf[..., None] + side_in * vs[..., None]) / np.maximum(wsum[..., None], 1.0)
                outputs["A2"] = (a2.astype(np.float32), a2_valid)
                for method, (pts, jvalid) in outputs.items():
                    world = canon.restore_face(pts); world[~jvalid] = 0
                    summary, _ = external_metrics_from_reference(world, ref, skeleton, jvalid, alignment="similarity")
                    rows.append(dict(person=pid, trial=tid, tier=tier, family=fam, view=view,
                                     method=method, mpjpe=summary["mpjpe"],
                                     matched_points=summary["matched_points"]))
        print(f"[{pi+1}/{len(val_people)}] person {pid} done  elapsed={time.time()-t0:.0f}s", flush=True)
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("saved", out_dir / "results.csv", len(rows), "rows")

if __name__ == "__main__":
    main()
