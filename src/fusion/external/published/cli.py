"""``python -m fusion external-published`` -- strict external baselines.

    videopose3d   --dataset gymnastics|freeman|sportspose [--mode procrustes_average|per_view]
                  official VideoPose3D lifter on the SAM3D 2D keypoints of both views,
                  evaluated through the model protocol (5 folds, phase windows,
                  per-frame PA-MPJPE on the major joints the method predicts)
    metapose      --dataset ... --stage prepare|s1|s2|evaluate|all [--eval-stage s2|s1|init]
                  official MetaPose (stage-1 solver, released 2-camera stage-2
                  network) on the same two views; see metapose_pipeline.py for the data flow
    canonpose     --dataset ... --stage prepare|train|evaluate|all   CanonPose trained
                  per fold with its released recipe on the training subjects' two-view
                  2D keypoints (self-supervised); see canonpose.py
    keypoints2d   --dataset gymnastics [--persons ...]   build the private 2D cache
                  (decodes every per-frame SAM3D file once; run on the cluster)

Results: ``local/runs/external_published/<method>/<dataset>/<mode>/summary.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from common.paths import PROJECT_ROOT

OUTPUT_ROOT = PROJECT_ROOT / "local" / "runs" / "external_published"


def _run_videopose3d(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_folds, write_summary
    from .transform import LiftedTrialTransform, view_source
    from .videopose3d import VideoPose3DLifter

    lifter = VideoPose3DLifter(args.checkpoint, args.device, test_time_augmentation=not args.no_tta)
    cache_dir = OUTPUT_ROOT / "lifted"
    source = view_source(args.dataset)

    def factory(fold):
        return LiftedTrialTransform(lifter.lift, source, mode=args.mode, cache_dir=cache_dir, method="videopose3d")

    extra = list(args.override or [])
    payload = evaluate_folds(args.dataset, factory, folds_dir=args.folds_dir, extra_overrides=extra)
    payload["method"] = {"name": "videopose3d", "mode": args.mode, "checkpoint": str(args.checkpoint or "default"), "test_time_augmentation": not args.no_tta, "receptive_field": lifter.receptive_field}
    out = write_summary(OUTPUT_ROOT / "videopose3d" / args.dataset / args.mode / "summary.json", payload)
    s = payload["summary"]
    print(f"[external/videopose3d] {args.dataset} {args.mode}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _cache_gymnastics_view(item: tuple[str, str]) -> dict:
    from .keypoints2d import DEFAULT_CACHE_ROOT, gymnastics_view

    person, view = item
    v = gymnastics_view(person, view, cache_root=DEFAULT_CACHE_ROOT)
    return {"person": person, "view": view, "frames": int(len(v.frame_ids)), "size": [v.width, v.height]}


def _run_metapose(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_folds, write_summary
    from .metapose_pipeline import MetaPoseTrialTransform, prepare_dataset, run_stage1, run_stage2
    from .transform import view_source
    from .videopose3d import VideoPose3DLifter

    directory = OUTPUT_ROOT / "metapose" / args.dataset
    extra = list(args.override or [])
    stages = ("prepare", "s1", "s2", "evaluate") if args.stage == "all" else (args.stage,)
    if "prepare" in stages:
        lifter = VideoPose3DLifter(args.checkpoint, args.device, test_time_augmentation=not args.no_tta)
        prepare_dataset(args.dataset, lifter.lift, view_source(args.dataset), output_dir=directory, lifted_cache=OUTPUT_ROOT / "lifted", extra_overrides=extra)
    if "s1" in stages:
        run_stage1(directory, steps=args.s1_steps, batch=args.s1_batch)
    if "s2" in stages:
        run_stage2(directory)
    if "evaluate" in stages:
        for stage in (args.eval_stage.split(",") if args.eval_stage else ["s2"]):
            payload = evaluate_folds(args.dataset, lambda fold, stage=stage: MetaPoseTrialTransform(directory, stage), folds_dir=args.folds_dir, extra_overrides=extra)
            payload["method"] = {"name": "metapose", "stage": stage, "checkpoint": "ckpt/h36m/cam2", "init": "videopose3d", "heatmaps": "single gaussian at the SAM3D keypoint (sigma 2 % of the box)"}
            out = write_summary(OUTPUT_ROOT / "metapose" / args.dataset / f"summary_{stage}.json", payload)
            s = payload["summary"]
            print(f"[external/metapose] {args.dataset} {stage}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _run_canonpose(args: argparse.Namespace) -> int:
    from .canonpose import CanonPoseTrialTransform, fold_train_persons, output_root, prepare_dataset, train
    from .evaluate import evaluate_folds, fold_files, write_summary
    from .transform import view_source

    root = output_root(args.dataset)
    extra = list(args.override or [])
    source = view_source(args.dataset)
    stages = ("prepare", "train", "evaluate") if args.stage == "all" else (args.stage,)
    folds = fold_files(args.dataset, args.folds_dir)
    if args.folds:
        folds = [f for f in folds if f.stem in set(args.folds)]
    if "prepare" in stages:
        prepare_dataset(args.dataset, source, output_dir=root, extra_overrides=extra)
    if "train" in stages:
        for fold in folds:
            out = root / fold.stem / "lifter.pt"
            if out.is_file() and not args.force:
                print(f"[canonpose] {fold.stem}: {out} exists, skipping (use --force)")
                continue
            print(f"[canonpose] training {args.dataset} {fold.stem} on {len(fold_train_persons(fold))} subjects")
            train(root / "inputs.npz", root / "index.json", fold_train_persons(fold), out, device=args.device, epochs=args.epochs, seed=args.seed)
    if "evaluate" in stages:
        for mode in args.mode.split(","):
            payload = evaluate_folds(args.dataset, lambda fold, mode=mode: CanonPoseTrialTransform(root / fold.stem / "lifter.pt", source, mode=mode, device=args.device), folds_dir=args.folds_dir, extra_overrides=extra)
            payload["method"] = {"name": "canonpose", "mode": mode, "training": "released recipe, self-supervised on the fold's training subjects", "epochs": args.epochs or "released default"}
            out = write_summary(root / f"summary_{mode}.json", payload)
            s = payload["summary"]
            print(f"[external/canonpose] {args.dataset} {mode}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _run_keypoints2d(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from common.paths import SAM3D_PERSON_ROOT

    from .keypoints2d import DEFAULT_CACHE_ROOT, write_manifest

    if args.dataset != "gymnastics":
        raise SystemExit("keypoints2d caches are only needed for the private recordings; FreeMan and SportsPose read their benchmark caches directly")
    persons = args.persons or sorted((p.name for p in SAM3D_PERSON_ROOT.iterdir() if p.is_dir()), key=lambda s: (len(s), s))
    jobs = [(person, view) for person in persons for view in ("face", "side")]
    entries = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for entry in pool.map(_cache_gymnastics_view, jobs):
            entries.append(entry)
            print(f"  person_{entry['person']} {entry['view']}: {entry['frames']} frames {entry['size']}")
    path = write_manifest(DEFAULT_CACHE_ROOT, "gymnastics", entries)
    print(f"[external/keypoints2d] {len(entries)} views -> {path}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fusion external-published", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="method", required=True)
    vp = sub.add_parser("videopose3d", help="official VideoPose3D lifter, per view, Procrustes-averaged")
    vp.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "sportspose"))
    vp.add_argument("--mode", default="procrustes_average", choices=("procrustes_average", "per_view"))
    vp.add_argument("--checkpoint", type=Path, default=None)
    vp.add_argument("--device", default="cuda")
    vp.add_argument("--no-tta", action="store_true", help="disable the authors' flip test-time augmentation")
    vp.add_argument("--folds-dir", type=Path, default=None)
    vp.add_argument("--override", nargs="*", default=None, help="extra Hydra data overrides")
    mp = sub.add_parser("metapose", help="official MetaPose stage 1 + released stage-2 network")
    mp.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "sportspose"))
    mp.add_argument("--stage", default="all", choices=("prepare", "s1", "s2", "evaluate", "all"))
    mp.add_argument("--eval-stage", default="s2", help="comma-separated: s2, s1 (iterative refinement only), init (monocular initialisation)")
    mp.add_argument("--checkpoint", type=Path, default=None, help="VideoPose3D checkpoint used for the monocular initialisation")
    mp.add_argument("--device", default="cuda")
    mp.add_argument("--no-tta", action="store_true")
    mp.add_argument("--s1-steps", type=int, default=100)
    mp.add_argument("--s1-batch", type=int, default=4096)
    mp.add_argument("--folds-dir", type=Path, default=None)
    mp.add_argument("--override", nargs="*", default=None, help="extra Hydra data overrides (applied to prepare and evaluate)")
    cp = sub.add_parser("canonpose", help="CanonPose trained per fold with its released recipe")
    cp.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "sportspose"))
    cp.add_argument("--stage", default="all", choices=("prepare", "train", "evaluate", "all"))
    cp.add_argument("--mode", default="canonical_average", help="comma-separated: canonical_average, procrustes_average, per_view")
    cp.add_argument("--epochs", type=int, default=None, help="override the released 100 epochs")
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--device", default="cuda")
    cp.add_argument("--folds", nargs="*", default=None, help="restrict training to these fold names")
    cp.add_argument("--folds-dir", type=Path, default=None)
    cp.add_argument("--force", action="store_true")
    cp.add_argument("--override", nargs="*", default=None)
    kp = sub.add_parser("keypoints2d", help="build the private per-view 2D keypoint cache")
    kp.add_argument("--dataset", default="gymnastics")
    kp.add_argument("--persons", nargs="*", default=None)
    kp.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    if args.method == "videopose3d":
        return _run_videopose3d(args)
    if args.method == "metapose":
        return _run_metapose(args)
    if args.method == "canonpose":
        return _run_canonpose(args)
    return _run_keypoints2d(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
