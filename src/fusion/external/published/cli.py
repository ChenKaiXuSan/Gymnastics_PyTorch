"""``python -m fusion external-published`` -- strict external baselines.

    videopose3d   --dataset gymnastics|freeman|fit3d [--mode procrustes_average|per_view]
                  the released Human3.6M VideoPose3D checkpoint on the SAM3D 2D
                  keypoints of both views (zero-shot, appendix only)
    videopose3d-trained  --dataset freeman|fit3d --stage prepare|train|evaluate|all
                  VideoPose3D (243-frame release recipe, supervised) trained per fold
                  on the training subjects' reference joints, both views as monocular
                  samples; same records as MHFormer; see videopose3d_train.py
    metapose      --dataset ... --stage prepare|s1|train|evaluate|all [--eval-stage s2|s1|init]
                  official MetaPose on the same two views: stage-1 solver, then the
                  stage-2 network trained per fold with the authors' script on the
                  training subjects (label-free); --stage s2 / --released instead runs
                  the released Human3.6M checkpoint (zero-shot, appendix only); see
                  metapose_pipeline.py for the data flow
    canonpose     --dataset ... --stage prepare|train|evaluate|all   CanonPose trained
                  per fold with its released recipe on the training subjects' two-view
                  2D keypoints (self-supervised); see canonpose.py
    mhformer      --dataset freeman|fit3d --stage prepare|train|evaluate|all
                  MHFormer (supervised, 81-frame release configuration) trained per
                  fold on the training subjects' reference joints, both views as
                  monocular samples; no row for the private data (no independent
                  3D reference); see mhformer.py
    mdvpose       --dataset freeman|fit3d --stage prepare|train|evaluate|all
                  MDVPose (MotionBERT fine-tuned with multi-view consistency,
                  supervised) trained per fold from the MotionBERT H36M checkpoint;
                  FreeMan / Fit3D only; see mdvpose.py
    model         --dataset ... --run <sweep dir> [--joints comparison12|all]
                  our own model's checkpoints through the same evaluator, so its
                  number is on the joints the external methods cover
    pseudo-reference build   triangulate FreeMan's two selected views into a
                  private-style pseudo-reference, to measure how much such a
                  reference flatters two-view fusion
    analysis      --dataset ... --run <sweep> [--what strata,measurement]
                  where the learned model beats its rule (strata that ignore the
                  reference) and trunk-rotation measurement error per cycle
    corruption    --dataset ... --run <sweep> [--levels 0,0.5,1,2] [--variants model,rule]
                  the learned model and its closed-form rule under increasing
                  test-time corruption (evaluation only, no retraining)
    compare       --a ours.json --b theirs.json ...   paired statistics over subjects
                  (fold-level tests floor at p = 0.0625 for n = 5)
    report        [--markdown out.md] [--csv out.csv]   table of every summary_*.json
    keypoints2d   --dataset gymnastics [--persons ...]   build the private 2D cache
                  (decodes every per-frame SAM3D file once; run on the cluster)

Results: ``local/runs/external_published/<method>/<dataset>/<mode>/summary.json``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

from common.paths import PROJECT_ROOT

OUTPUT_ROOT = PROJECT_ROOT / "local" / "runs" / "external_published"


def _list(value: str) -> list[str]:
    """Comma- or plus-separated option values (``qsub -v`` splits on commas)."""
    return [v for v in re.split(r"[,+]", value) if v]


def _joints(args: argparse.Namespace) -> list[str] | None:
    """``--joints``: ``all`` (the method's own joints), ``comparison12`` or explicit names."""
    from .evaluate import COMPARISON_JOINTS

    value = getattr(args, "joints", None)
    if not value or value == "all":
        return None
    if value == "comparison12":
        return list(COMPARISON_JOINTS)
    return _list(value)


def _summary_name(stem: str, joints: list[str] | None) -> str:
    """A joint subset writes its own file so the method's own-joint summary survives."""
    return f"summary_{stem}.json" if joints is None else f"summary_{stem}_{len(joints)}joints.json"


def _joints_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--joints", default="all", help="'all' (the method's own joints), 'comparison12', or comma/plus-separated joint names")


def _run_videopose3d(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_folds, write_summary
    from .transform import LiftedTrialTransform, view_source
    from .videopose3d import VideoPose3DLifter

    joints = _joints(args)
    lifter = VideoPose3DLifter(args.checkpoint, args.device, test_time_augmentation=not args.no_tta)
    cache_dir = OUTPUT_ROOT / "lifted"
    source = view_source(args.dataset)

    def factory(fold):
        return LiftedTrialTransform(lifter.lift, source, mode=args.mode, cache_dir=cache_dir, method="videopose3d")

    extra = list(args.override or [])
    payload = evaluate_folds(args.dataset, factory, folds_dir=args.folds_dir, extra_overrides=extra, joints=joints)
    payload["method"] = {"name": "videopose3d", "mode": args.mode, "checkpoint": str(args.checkpoint or "default"), "test_time_augmentation": not args.no_tta, "receptive_field": lifter.receptive_field}
    out = write_summary(OUTPUT_ROOT / "videopose3d" / args.dataset / args.mode / _summary_name("zero_shot", joints).replace("summary_zero_shot.json", "summary.json"), payload)
    s = payload["summary"]
    print(f"[external/videopose3d] {args.dataset} {args.mode}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _cache_gymnastics_view(item: tuple[str, str]) -> dict:
    from .keypoints2d import DEFAULT_CACHE_ROOT, gymnastics_view

    person, view = item
    v = gymnastics_view(person, view, cache_root=DEFAULT_CACHE_ROOT)
    return {"person": person, "view": view, "frames": int(len(v.frame_ids)), "size": [v.width, v.height]}


def _run_metapose(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_folds, fold_files, write_summary
    from .metapose_pipeline import MetaPoseTrialTransform, prepare_dataset, run_shards, run_stage1, run_stage2_predict, run_stage2_released, run_stage2_trained, stage2_name
    from .transform import view_source
    from .videopose3d import VideoPose3DLifter

    directory = OUTPUT_ROOT / "metapose" / args.dataset
    extra = list(args.override or [])
    stages = ("prepare", "s1", "shards", "train", "evaluate") if args.stage == "all" else (args.stage,)
    folds = fold_files(args.dataset, args.folds_dir)
    if args.folds:
        folds = [f for f in folds if f.stem in set(args.folds)]
    if "predict" in stages:
        for fold in folds:
            run_stage2_predict(directory, fold, loss=args.loss, stages=args.checkpoint_stages, seed=args.seed)
    if "prepare" in stages:
        lifter = VideoPose3DLifter(args.checkpoint, args.device, test_time_augmentation=not args.no_tta)
        prepare_dataset(args.dataset, lifter.lift, view_source(args.dataset), output_dir=directory, lifted_cache=OUTPUT_ROOT / "lifted", extra_overrides=extra)
    if "s1" in stages:
        run_stage1(directory, steps=args.s1_steps, batch=args.s1_batch)
    if "shards" in stages:
        run_shards(directory, workers=args.workers)
    if "train" in stages:
        for fold in folds:
            out = directory / f"{stage2_name(fold.stem, args.loss)}.npz"
            if out.is_file() and not args.force:
                print(f"[metapose] {fold.stem}: {out} exists, skipping (use --force)")
                continue
            run_stage2_trained(directory, fold, epochs_per_stage=args.epochs_per_stage, patience=args.patience, max_stages=args.max_stages, seed=args.seed, loss=args.loss)
    if "s2" in stages:
        run_stage2_released(directory)
    if "evaluate" in stages:
        joints = _joints(args)
        released = bool(args.released)
        schedule = {"epochs_per_stage": args.epochs_per_stage, "early_stopping_patience": args.patience, "max_n_stages": args.max_stages, "released_schedule": "300 epochs x 10 stages, patience 50"}
        for stage in (_list(args.eval_stage) if args.eval_stage else ["s2"]):
            if stage == "s2" and not released:
                factory = lambda fold: MetaPoseTrialTransform(directory, "s2", fold=fold.stem, loss=args.loss)  # noqa: E731
                objective = {"fwd": "reprojection MSE to the SAM3D 2D (README default; the release uses 2D ground truth)", "ss": "heatmap log-likelihood (README S1+S2/SS)", "ts": "student of the stage-1 solution (README S1+S2/TS)"}[args.loss]
                method = {"name": "metapose", "stage": "s2", "loss": args.loss, "training": f"authors' train_metapose per fold on the training subjects; objective: {objective}; label-free selection on the stage-1 optimum", **schedule}
                name = stage2_name("", args.loss).rstrip("_")
            else:
                factory = lambda fold, stage=stage: MetaPoseTrialTransform(directory, stage)  # noqa: E731
                method = {"name": "metapose", "stage": stage, "checkpoint": "ckpt/h36m/cam2 (released, zero-shot)" if stage == "s2" else "none (optimisation only)"}
                name = "s2_released" if stage == "s2" else stage
            method.update({"init": "videopose3d", "heatmaps": "single gaussian at the SAM3D keypoint (sigma 2 % of the box)"})
            payload = evaluate_folds(args.dataset, factory, folds_dir=args.folds_dir, extra_overrides=extra, joints=joints)
            payload["method"] = method
            out = write_summary(directory / _summary_name(name, joints), payload)
            s = payload["summary"]
            print(f"[external/metapose] {args.dataset} {name}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
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
        joints = _joints(args)
        for mode in _list(args.mode):
            payload = evaluate_folds(args.dataset, lambda fold, mode=mode: CanonPoseTrialTransform(root / fold.stem / "lifter.pt", source, mode=mode, device=args.device), folds_dir=args.folds_dir, extra_overrides=extra, joints=joints)
            payload["method"] = {"name": "canonpose", "mode": mode, "training": "released recipe, self-supervised on the fold's training subjects", "epochs": args.epochs or "released default"}
            out = write_summary(root / _summary_name(mode, joints), payload)
            s = payload["summary"]
            print(f"[external/canonpose] {args.dataset} {mode}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _run_mhformer(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_folds, fold_files, write_summary
    from .mhformer import CONFIG, MHFormerLifter, fold_persons, output_root, prepare_dataset, train
    from .transform import LiftedTrialTransform, view_source

    root = output_root(args.dataset)
    extra = list(args.override or [])
    source = view_source(args.dataset)
    stages = ("prepare", "train", "evaluate") if args.stage == "all" else (args.stage,)
    folds = fold_files(args.dataset, args.folds_dir)
    if args.folds:
        folds = [f for f in folds if f.stem in set(args.folds)]
    config = {"frames": args.frames} if args.frames else {}
    if "prepare" in stages:
        prepare_dataset(args.dataset, source, output_dir=root, extra_overrides=extra)
    if "train" in stages:
        for fold in folds:
            out = root / fold.stem / "model.pt"
            if out.is_file() and not args.force:
                print(f"[mhformer] {fold.stem}: {out} exists, skipping (use --force)")
                continue
            print(f"[mhformer] training {args.dataset} {fold.stem} on {len(fold_persons(fold, 'train'))} subjects, selecting on {len(fold_persons(fold, 'val'))}")
            train(root / "inputs.npz", root / "index.json", fold_persons(fold, "train"), fold_persons(fold, "val"), out, device=args.device, config=config, epochs=args.epochs)
    if "evaluate" in stages:
        joints = _joints(args)
        for mode in _list(args.mode):

            def factory(fold, mode=mode):
                lifter = MHFormerLifter(root / fold.stem / "model.pt", args.device)
                return LiftedTrialTransform(lifter.lift, source, mode=mode, cache_dir=OUTPUT_ROOT / "lifted", method=f"mhformer_{args.dataset}_{fold.stem}")

            payload = evaluate_folds(args.dataset, factory, folds_dir=args.folds_dir, extra_overrides=extra, joints=joints)
            payload["method"] = {"name": "mhformer", "mode": mode, "training": "released recipe, supervised on the fold's training subjects' reference joints (both views)", "frames": args.frames or CONFIG["frames"], "epochs": (args.epochs or CONFIG["nepoch"]) - 1, "selection": "best validation-subject MPJPE"}
            out = write_summary(root / _summary_name(mode, joints), payload)
            s = payload["summary"]
            print(f"[external/mhformer] {args.dataset} {mode}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _run_mdvpose(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_folds, fold_files, write_summary
    from .mdvpose import CONFIG, MDVPoseLifter, output_root, train
    from .mhformer import fold_persons, prepare_dataset
    from .transform import LiftedTrialTransform, view_source

    root = output_root(args.dataset)
    extra = list(args.override or [])
    source = view_source(args.dataset)
    stages = ("prepare", "train", "evaluate") if args.stage == "all" else (args.stage,)
    folds = fold_files(args.dataset, args.folds_dir)
    if args.folds:
        folds = [f for f in folds if f.stem in set(args.folds)]
    if "prepare" in stages:
        prepare_dataset(args.dataset, source, output_dir=root, extra_overrides=extra)  # same records as MHFormer
    if "train" in stages:
        for fold in folds:
            out = root / fold.stem / "model.pt"
            if out.is_file() and not args.force:
                print(f"[mdvpose] {fold.stem}: {out} exists, skipping (use --force)")
                continue
            print(f"[mdvpose] training {args.dataset} {fold.stem} on {len(fold_persons(fold, 'train'))} subjects, selecting on {len(fold_persons(fold, 'val'))}")
            train(root / "inputs.npz", root / "index.json", fold_persons(fold, "train"), fold_persons(fold, "val"), out, device=args.device, epochs=args.epochs, config={"pairs_per_batch": args.pairs_per_batch})
    if "evaluate" in stages:
        joints = _joints(args)
        for mode in _list(args.mode):

            def factory(fold, mode=mode):
                lifter = MDVPoseLifter(root / fold.stem / "model.pt", args.device)
                return LiftedTrialTransform(lifter.lift, source, mode=mode, cache_dir=OUTPUT_ROOT / "lifted", method=f"mdvpose_{args.dataset}_{fold.stem}")

            payload = evaluate_folds(args.dataset, factory, folds_dir=args.folds_dir, extra_overrides=extra, joints=joints)
            payload["method"] = {"name": "mdvpose", "mode": mode, "training": "released multi-view fine-tuning recipe from the MotionBERT H36M checkpoint, supervised on the fold's training subjects' reference joints (both views)", "epochs": args.epochs or CONFIG["epochs"], "selection": "best validation-subject MPJPE"}
            out = write_summary(root / _summary_name(mode, joints), payload)
            s = payload["summary"]
            print(f"[external/mdvpose] {args.dataset} {mode}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _run_videopose3d_trained(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_folds, fold_files, write_summary
    from .mhformer import fold_persons, prepare_dataset
    from .transform import LiftedTrialTransform, view_source
    from .videopose3d_train import CONFIG, VideoPose3DTrainedLifter, output_root, train

    root = output_root(args.dataset)
    extra = list(args.override or [])
    source = view_source(args.dataset)
    stages = ("prepare", "train", "evaluate") if args.stage == "all" else (args.stage,)
    folds = fold_files(args.dataset, args.folds_dir)
    if args.folds:
        folds = [f for f in folds if f.stem in set(args.folds)]
    if "prepare" in stages:
        prepare_dataset(args.dataset, source, output_dir=root, extra_overrides=extra)  # same records as MHFormer
    if "train" in stages:
        for fold in folds:
            out = root / fold.stem / "model.pt"
            if out.is_file() and not args.force:
                print(f"[videopose3d] {fold.stem}: {out} exists, skipping (use --force)")
                continue
            print(f"[videopose3d] training {args.dataset} {fold.stem} on {len(fold_persons(fold, 'train'))} subjects (validation subjects logged only)")
            train(root / "inputs.npz", root / "index.json", fold_persons(fold, "train"), fold_persons(fold, "val"), out, device=args.device, epochs=args.epochs)
    if "evaluate" in stages:
        joints = _joints(args)
        for mode in _list(args.mode):

            def factory(fold, mode=mode):
                lifter = VideoPose3DTrainedLifter(root / fold.stem / "model.pt", args.device)
                return LiftedTrialTransform(lifter.lift, source, mode=mode, cache_dir=OUTPUT_ROOT / "lifted", method=f"videopose3d_trained_{args.dataset}_{fold.stem}")

            payload = evaluate_folds(args.dataset, factory, folds_dir=args.folds_dir, extra_overrides=extra, joints=joints)
            payload["method"] = {"name": "videopose3d_trained", "mode": mode, "training": "released 243-frame recipe (run.py -e 80 -arc 3,3,3,3,3), supervised on the fold's training subjects' reference joints (both views), final epoch", "epochs": args.epochs or CONFIG["epochs"]}
            out = write_summary(root / _summary_name(mode, joints), payload)
            s = payload["summary"]
            print(f"[external/videopose3d_trained] {args.dataset} {mode}: PA-MPJPE {s['pa_mpjpe_mean'] * 1000:.1f} ± {s['pa_mpjpe_sd'] * 1000:.1f} mm over {s['folds']} folds, joints {len(s['joint_names'])} -> {out}")
    return 0


def _run_keypoints2d(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from common.paths import SAM3D_PERSON_ROOT

    from .keypoints2d import DEFAULT_CACHE_ROOT, write_manifest

    if args.dataset != "gymnastics":
        raise SystemExit("keypoints2d caches are only needed for the private recordings; FreeMan and Fit3D read their benchmark caches directly")
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
    vp.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    vp.add_argument("--mode", default="procrustes_average", choices=("procrustes_average", "per_view"))
    vp.add_argument("--checkpoint", type=Path, default=None)
    vp.add_argument("--device", default="cuda")
    vp.add_argument("--no-tta", action="store_true", help="disable the authors' flip test-time augmentation")
    vp.add_argument("--folds-dir", type=Path, default=None)
    _joints_argument(vp)
    vp.add_argument("--override", nargs="*", default=None, help="extra Hydra data overrides")
    vt = sub.add_parser("videopose3d-trained", help="VideoPose3D trained per fold on the reference joints (supervised)")
    vt.add_argument("--dataset", required=True, choices=("freeman", "fit3d"))
    vt.add_argument("--stage", default="all", choices=("prepare", "train", "evaluate", "all"))
    vt.add_argument("--mode", default="procrustes_average", help="comma- or plus-separated: procrustes_average, per_view")
    vt.add_argument("--epochs", type=int, default=None, help="override the released 80 epochs")
    vt.add_argument("--device", default="cuda")
    vt.add_argument("--folds", nargs="*", default=None)
    vt.add_argument("--folds-dir", type=Path, default=None)
    vt.add_argument("--force", action="store_true")
    _joints_argument(vt)
    vt.add_argument("--override", nargs="*", default=None)
    mp = sub.add_parser("metapose", help="official MetaPose: stage-1 solver + stage-2 network trained per fold")
    mp.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    mp.add_argument("--stage", default="all", choices=("prepare", "s1", "shards", "train", "predict", "s2", "evaluate", "all"), help="all = prepare, s1, shards, train, evaluate; predict = export from a saved best checkpoint; s2 = released checkpoint (zero-shot)")
    mp.add_argument("--checkpoint-stages", type=int, default=1, help="predict: number of stage models in the saved checkpoint")
    mp.add_argument("--workers", type=int, default=8, help="record shard writer processes")
    mp.add_argument("--eval-stage", default="s2", help="comma-separated: s2, s1 (iterative refinement only), init (monocular initialisation)")
    mp.add_argument("--released", action="store_true", help="evaluate the released checkpoint's s2.npz instead of the per-fold trained networks")
    mp.add_argument("--epochs-per-stage", type=int, default=30, help="cap of the authors' 300-epoch stages")
    mp.add_argument("--patience", type=int, default=5, help="early-stopping patience (released: 50)")
    mp.add_argument("--max-stages", type=int, default=3, help="refinement stages (released: up to 10)")
    mp.add_argument("--seed", type=int, default=0)
    mp.add_argument("--loss", default="fwd", choices=("fwd", "ss", "ts"), help="stage-2 objective: fwd (README default), ss (heatmap likelihood), ts (student of stage 1)")
    mp.add_argument("--folds", nargs="*", default=None, help="restrict training to these fold names")
    mp.add_argument("--force", action="store_true")
    mp.add_argument("--checkpoint", type=Path, default=None, help="VideoPose3D checkpoint used for the monocular initialisation")
    mp.add_argument("--device", default="cuda")
    mp.add_argument("--no-tta", action="store_true")
    mp.add_argument("--s1-steps", type=int, default=100)
    mp.add_argument("--s1-batch", type=int, default=4096)
    mp.add_argument("--folds-dir", type=Path, default=None)
    _joints_argument(mp)
    mp.add_argument("--override", nargs="*", default=None, help="extra Hydra data overrides (applied to prepare and evaluate)")
    cp = sub.add_parser("canonpose", help="CanonPose trained per fold with its released recipe")
    cp.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    cp.add_argument("--stage", default="all", choices=("prepare", "train", "evaluate", "all"))
    cp.add_argument("--mode", default="canonical_average", help="comma-separated: canonical_average, procrustes_average, per_view")
    cp.add_argument("--epochs", type=int, default=None, help="override the released 100 epochs")
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--device", default="cuda")
    cp.add_argument("--folds", nargs="*", default=None, help="restrict training to these fold names")
    cp.add_argument("--folds-dir", type=Path, default=None)
    cp.add_argument("--force", action="store_true")
    _joints_argument(cp)
    cp.add_argument("--override", nargs="*", default=None)
    mh = sub.add_parser("mhformer", help="MHFormer trained per fold on the reference joints (supervised)")
    mh.add_argument("--dataset", required=True, choices=("freeman", "fit3d"))
    mh.add_argument("--stage", default="all", choices=("prepare", "train", "evaluate", "all"))
    mh.add_argument("--mode", default="procrustes_average", help="comma-separated: procrustes_average, per_view")
    mh.add_argument("--frames", type=int, default=None, help="receptive field (released: 81 here; 351 = 4x the cost)")
    mh.add_argument("--epochs", type=int, default=None, help="override the released nepoch (20 -> 19 epochs)")
    mh.add_argument("--device", default="cuda")
    mh.add_argument("--folds", nargs="*", default=None)
    mh.add_argument("--folds-dir", type=Path, default=None)
    mh.add_argument("--force", action="store_true")
    _joints_argument(mh)
    mh.add_argument("--override", nargs="*", default=None)
    md = sub.add_parser("mdvpose", help="MDVPose (MotionBERT multi-view fine-tuning) trained per fold (supervised)")
    md.add_argument("--dataset", required=True, choices=("freeman", "fit3d"))
    md.add_argument("--stage", default="all", choices=("prepare", "train", "evaluate", "all"))
    md.add_argument("--mode", default="procrustes_average", help="comma-separated: procrustes_average, per_view")
    md.add_argument("--epochs", type=int, default=None, help="override the released 60 epochs")
    md.add_argument("--pairs-per-batch", type=int, default=3, help="clip pairs per batch (3 = the released batch of six clips)")
    md.add_argument("--device", default="cuda")
    md.add_argument("--folds", nargs="*", default=None)
    md.add_argument("--folds-dir", type=Path, default=None)
    md.add_argument("--force", action="store_true")
    _joints_argument(md)
    md.add_argument("--override", nargs="*", default=None)
    md_ = sub.add_parser("model", help="evaluate our model's fold checkpoints through this evaluator")
    md_.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    md_.add_argument("--run", type=Path, required=True)
    md_.add_argument("--joints", default="comparison12")
    md_.add_argument("--checkpoint", default="auto", choices=("auto", "final", "last", "best"), help="auto = final.ckpt (the reported last-epoch weights) when present, else last.ckpt")
    md_.add_argument("--variant", default="model", help="model | base | rule | face | side")
    md_.add_argument("--device", default="cuda")
    md_.add_argument("--folds-dir", type=Path, default=None)
    md_.add_argument("--override", nargs="*", default=None)
    pr = sub.add_parser("pseudo-reference", help="build FreeMan's two-view triangulated reference")
    pr.add_argument("action", choices=("build",))
    pr.add_argument("--benchmark-root", type=Path, default=None)
    pr.add_argument("--subjects", nargs="*", type=int, default=None)
    pr.add_argument("--scale-to-m", type=float, default=0.01)
    an = sub.add_parser("analysis", help="stratified comparison and measurement-level errors")
    an.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    an.add_argument("--run", type=Path, required=True)
    an.add_argument("--what", default="strata,measurement")
    an.add_argument("--joints", default="comparison12")
    an.add_argument("--device", default="cuda")
    an.add_argument("--folds-dir", type=Path, default=None)
    an.add_argument("--override", nargs="*", default=None)
    cr = sub.add_parser("corruption", help="corruption-level sweep of a finished run")
    cr.add_argument("--dataset", required=True, choices=("gymnastics", "freeman", "fit3d"))
    cr.add_argument("--run", type=Path, required=True)
    cr.add_argument("--levels", default="0,0.5,1,2")
    cr.add_argument("--variants", default="model,rule")
    cr.add_argument("--joints", default="comparison12")
    cr.add_argument("--device", default="cuda")
    cr.add_argument("--folds-dir", type=Path, default=None)
    cr.add_argument("--override", nargs="*", default=None)
    cm = sub.add_parser("compare", help="paired statistics over subjects between two result files")
    cm.add_argument("--a", type=Path, required=True)
    cm.add_argument("--b", type=Path, nargs="+", required=True)
    cm.add_argument("--labels", nargs="*", default=None)
    cm.add_argument("--json", type=Path, default=None)
    cm.add_argument("--bootstrap", type=int, default=10000)
    cm.add_argument("--seed", type=int, default=0)
    rp = sub.add_parser("report", help="collect every summary_*.json into one table")
    rp.add_argument("--markdown", type=Path, default=None)
    rp.add_argument("--csv", type=Path, default=None)
    rp.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    kp = sub.add_parser("keypoints2d", help="build the private per-view 2D keypoint cache")
    kp.add_argument("--dataset", default="gymnastics")
    kp.add_argument("--persons", nargs="*", default=None)
    kp.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    if args.method == "videopose3d":
        return _run_videopose3d(args)
    if args.method == "videopose3d-trained":
        return _run_videopose3d_trained(args)
    if args.method == "metapose":
        return _run_metapose(args)
    if args.method == "canonpose":
        return _run_canonpose(args)
    if args.method == "mhformer":
        return _run_mhformer(args)
    if args.method == "mdvpose":
        return _run_mdvpose(args)
    if args.method == "model":
        from .model_rows import main as model_main

        argv2 = ["--dataset", args.dataset, "--run", str(args.run), "--joints", args.joints, "--checkpoint", args.checkpoint, "--device", args.device, "--variant", args.variant]
        if args.folds_dir:
            argv2 += ["--folds-dir", str(args.folds_dir)]
        if args.override:
            argv2 += ["--override", *args.override]
        return model_main(argv2)
    if args.method == "pseudo-reference":
        from .pseudo_reference import main as pseudo_main

        argv2 = [args.action, "--scale-to-m", str(args.scale_to_m)]
        if args.benchmark_root:
            argv2 += ["--benchmark-root", str(args.benchmark_root)]
        if args.subjects:
            argv2 += ["--subjects", *[str(s) for s in args.subjects]]
        return pseudo_main(argv2)
    if args.method == "analysis":
        from .analysis_rows import main as analysis_main

        argv2 = ["--dataset", args.dataset, "--run", str(args.run), "--what", args.what, "--joints", args.joints, "--device", args.device]
        if args.folds_dir:
            argv2 += ["--folds-dir", str(args.folds_dir)]
        if args.override:
            argv2 += ["--override", *args.override]
        return analysis_main(argv2)
    if args.method == "corruption":
        from .corruption_sweep import main as corruption_main

        argv2 = ["--dataset", args.dataset, "--run", str(args.run), "--levels", args.levels, "--variants", args.variants, "--joints", args.joints, "--device", args.device]
        if args.folds_dir:
            argv2 += ["--folds-dir", str(args.folds_dir)]
        if args.override:
            argv2 += ["--override", *args.override]
        return corruption_main(argv2)
    if args.method == "compare":
        from .stats import main as stats_main

        argv2 = ["--a", str(args.a), "--b", *[str(p) for p in args.b], "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)]
        if args.labels:
            argv2 += ["--labels", *args.labels]
        if args.json:
            argv2 += ["--json", str(args.json)]
        return stats_main(argv2)
    if args.method == "report":
        from .report import main as report_main

        return report_main([a for pair in (("--root", str(args.root)), ("--markdown", str(args.markdown)) if args.markdown else (), ("--csv", str(args.csv)) if args.csv else ()) for a in pair])
    return _run_keypoints2d(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
