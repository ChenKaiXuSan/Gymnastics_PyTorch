"""Out-of-fold publication of the final fusion model (v1.2) for the cohort analysis.

The cohort and repeated-cycle analysis (``features`` -> ``analyze`` -> ``assets``
stages of ``python -m fusion cohort-cycle``) reads one archive per cycle in
native time. The archived pipeline produced those archives from the
rotation-aware A6 model; this module produces them from the final cycle-aware
model instead, using its five subject-disjoint folds: every person's poses
come from the fold whose *test* set contains that person, so the analysis
pose of a person never comes from a model trained or selected on them.

The model works on phase-normalised windows (64 samples per cycle). Its only
learned output is the bounded residual ``Delta_P`` added to the closed-form
depth-aware rule, so the native-time pose is rebuilt as

    P_native(t) = rule(views at native frame t) + interp_t(Delta_P)

with the residual linearly interpolated in physical time within each cycle.
Angular speed and jerk are therefore computed on the native frame grid, as
in the archived publication.

Outputs (``--output``):
    publication/oof_provenance.csv, publication/person_<id>/<cycle>/prediction.npz
        kpts_body (fused), kpts_face_canonical, kpts_side_canonical in the
        70-joint MHR layout (the 20 major joints filled, others invalid),
        theta_fused_rad, omega_fused_rad_s, timestamps, fps, frame_valid,
        joint_valid, face_map (native frame index within the person).
    deterministic/person_<id>/fused_sequence.npz
        the closed-form rule alone (pose source "deterministic").

    python -m fusion.analysis.cohort_cycle.model_export \\
        --run local/runs/cycle_aware/gymnastics_v1_1_equal_reliability_5fold_seed0 \\
        --output local/runs/cohort_cycle/v1_2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from common.skeletons import MHR70_INDEX

from .features import axial_rotation_from_pose

CORE = ("left-hip", "right-hip", "left-shoulder", "right-shoulder")


def _to_mhr70(points: np.ndarray, valid: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Scatter ``[T, J, 3]`` named joints into the 70-joint layout."""
    frames = points.shape[0]
    out = np.zeros((frames, 70, 3), dtype=np.float32)
    ok = np.zeros((frames, 70), dtype=bool)
    for j, name in enumerate(names):
        index = MHR70_INDEX[name]
        out[:, index] = points[:, j]
        ok[:, index] = valid[:, j]
    return out, ok


def _axial(kpts70: np.ndarray, ok70: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis-to-shoulder axial rotation and its rate; NaN where the trunk is invalid."""
    trunk = np.all(ok70[:, [MHR70_INDEX[n] for n in CORE]], axis=1)
    theta = np.full(len(kpts70), np.nan)
    omega = np.full(len(kpts70), np.nan)
    if trunk.sum() >= 2:
        theta[trunk] = axial_rotation_from_pose(kpts70[trunk].astype(np.float64))
        omega[trunk] = np.gradient(theta[trunk], timestamps[trunk])
    return theta.astype(np.float32), omega.astype(np.float32)


def _cohorts(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["person_id"]: row["cohort"] for row in csv.DictReader(handle)}


def export(run_dir: Path, output: Path, *, cohort_source: Path, device: str = "cuda", folds_dir: Path | None = None) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from fusion.data import build_datamodule
    from fusion.external.published.evaluate import data_overrides, fold_files
    from fusion.external.published.model_rows import fold_checkpoint, load_module
    from fusion.train import compose_config

    cohorts = _cohorts(cohort_source)
    publication = output / "publication"
    deterministic = output / "deterministic"
    if publication.exists() or deterministic.exists():
        raise FileExistsError(f"export output already exists under {output}")
    rows: list[dict[str, Any]] = []
    target = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    for fold_number, fold in enumerate(fold_files("gymnastics", folds_dir), start=1):
        cfg = compose_config(data_overrides("gymnastics", fold, ["data.attach_reference=false"]))
        dm = build_datamodule(OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
        dm.setup("test")
        dataset = dm.dataset("test")
        names = list(dm.skeleton.joint_names)
        module = load_module(fold_checkpoint(run_dir, fold.stem), device)
        model = module.model
        # 1) Residual on the phase-normalised test windows.
        residual_sum: dict[str, np.ndarray] = {}
        residual_count: dict[str, np.ndarray] = {}
        prepared: dict[str, Any] = {}
        for index in range(len(dataset)):
            descriptor = dataset._windows[index]
            sample = descriptor.sample
            key = sample.key
            if key not in prepared:
                prepared[key] = sample
                residual_sum[key] = np.zeros((sample.num_frames, len(names), 3), dtype=np.float64)
                residual_count[key] = np.zeros((sample.num_frames, len(names)), dtype=np.float64)
            item = dataset[index]
            batch = {k: (v.unsqueeze(0).to(target) if torch.is_tensor(v) else [v]) for k, v in item.items()}
            with torch.no_grad():
                out = module(batch)
            delta = out.delta_pose[0].float().cpu().numpy()
            valid = (out.valid[0].cpu().numpy()) & item["frame_mask"].numpy()[:, None]
            start = descriptor.start
            stop = min(start + dataset.length, sample.num_frames)
            span = stop - start
            residual_sum[key][start:stop] += np.where(valid[:span, :, None], delta[:span], 0.0)
            residual_count[key][start:stop] += valid[:span]
        # 2) Native-time reconstruction per test person and cycle.
        natives = {s.key: s for s in dm._samples_for("test")}
        for key, phased in prepared.items():
            native = natives[key]
            person = str(native.subject_id)
            residual = residual_sum[key] / np.maximum(residual_count[key], 1)[..., None]
            pose_a = torch.from_numpy(np.asarray(native.view_a, dtype=np.float32))[None].to(target)
            pose_b = torch.from_numpy(np.asarray(native.view_b, dtype=np.float32))[None].to(target)
            valid_a = torch.from_numpy(np.asarray(native.valid_a, dtype=bool))[None].to(target)
            valid_b = torch.from_numpy(np.asarray(native.valid_b, dtype=bool))[None].to(target)
            half = torch.full_like(pose_a[..., :1], 0.5)
            depth_a = torch.from_numpy(np.asarray(native.depth_a, dtype=np.float32))[None].to(target)
            depth_b = torch.from_numpy(np.asarray(native.depth_b, dtype=np.float32))[None].to(target)
            with torch.no_grad():
                rule, rule_valid = model.fuse_base(pose_a, pose_b, half, half, valid_a, valid_b, depth_a, depth_b)
            rule = rule[0].float().cpu().numpy()
            rule_valid = rule_valid[0].cpu().numpy()
            fused = rule.copy()
            # Cycle ids in the adapter's concatenation order (cached trials sorted by face frame).
            trials = sorted(dm._trial_loader(person), key=lambda t: int(t.face_map[0]))
            cycle_ids = [t.trial_id for t in trials]
            face_frames = np.concatenate([np.asarray(t.face_map, dtype=np.int32) for t in trials])
            if len(cycle_ids) != len(native.cycle_bounds) or len(phased.cycle_bounds) != len(native.cycle_bounds):
                raise ValueError(f"{key}: cycle bookkeeping differs between native and phase-normalised samples")
            for (n0, n1), (p0, p1) in zip(native.cycle_bounds, phased.cycle_bounds):
                t_native = native.timestamps[n0:n1]
                t_phase = phased.timestamps[p0:p1]
                for j in range(len(names)):
                    for axis in range(3):
                        fused[n0:n1, j, axis] += np.interp(t_native, t_phase, residual[p0:p1, j, axis])
            fps = float(native.metadata.get("fps", 0.0)) or float(1.0 / np.median(np.diff(native.timestamps)))
            frame_index = face_frames if len(face_frames) == native.num_frames else np.arange(native.num_frames, dtype=np.int32)
            rule70, rule_ok = _to_mhr70(rule, rule_valid, names)
            fused70, fused_ok = _to_mhr70(fused, rule_valid, names)
            face70, face_ok = _to_mhr70(np.asarray(native.view_a), np.asarray(native.valid_a), names)
            side70, side_ok = _to_mhr70(np.asarray(native.view_b), np.asarray(native.valid_b), names)
            person_det = deterministic / f"person_{person}"
            person_det.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(person_det / "fused_sequence.npz", kpts_body=rule70, joint_valid=rule_ok, face_map=frame_index)
            for cycle_id, (n0, n1) in zip(cycle_ids, native.cycle_bounds):
                sl = slice(n0, n1)
                timestamps = native.timestamps[sl] - native.timestamps[n0]
                theta, omega = _axial(fused70[sl], fused_ok[sl], timestamps)
                core = [MHR70_INDEX[n] for n in CORE]
                frame_valid = np.all(fused_ok[sl][:, core], axis=1) & np.isfinite(theta)
                cycle_dir = publication / f"person_{person}" / str(cycle_id)
                cycle_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cycle_dir / "prediction.npz",
                    kpts_body=fused70[sl], kpts_face_canonical=face70[sl], kpts_side_canonical=side70[sl],
                    theta_fused_rad=theta, omega_fused_rad_s=omega, timestamps=timestamps, fps=np.float64(fps),
                    frame_valid=frame_valid, joint_valid=fused_ok[sl] & face_ok[sl] & side_ok[sl], face_map=frame_index[sl],
                )
                rows.append({
                    "person_id": person, "cohort": cohorts[person], "outer_fold": fold_number, "run_id": Path(run_dir).name,
                    "seed": 0, "cycle_id": str(cycle_id), "prediction_path": f"person_{person}/{cycle_id}/prediction.npz",
                })
        print(f"[export] {fold.stem}: {len(prepared)} test people", flush=True)
    rows.sort(key=lambda r: (int(r["person_id"]), r["cycle_id"]))
    with (publication / "oof_provenance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "run": str(run_dir), "people": len({r["person_id"] for r in rows}), "cycles": len(rows),
        "per_cohort": {c: len({r["person_id"] for r in rows if r["cohort"] == c}) for c in ("elderly", "student")},
        "per_fold": {str(k): len({r["person_id"] for r in rows if r["outer_fold"] == k}) for k in sorted({r["outer_fold"] for r in rows})},
    }
    (publication / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, default=Path("local/runs/cohort_cycle/oof_seed0/oof_provenance.csv"),
                        help="CSV with person_id and cohort columns (the archived publication's provenance)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(export(args.run, args.output, cohort_source=args.cohorts, device=args.device, folds_dir=args.folds_dir), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
