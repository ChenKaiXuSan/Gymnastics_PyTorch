"""``python -m fusion external-published`` -- strict external baselines.

    videopose3d   --dataset gymnastics|freeman|sportspose [--mode procrustes_average|per_view]
                  official VideoPose3D lifter on the SAM3D 2D keypoints of both views,
                  evaluated through the model protocol (5 folds, phase windows,
                  per-frame PA-MPJPE on the major joints the method predicts)
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

    def factory():
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
    kp = sub.add_parser("keypoints2d", help="build the private per-view 2D keypoint cache")
    kp.add_argument("--dataset", default="gymnastics")
    kp.add_argument("--persons", nargs="*", default=None)
    kp.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    if args.method == "videopose3d":
        return _run_videopose3d(args)
    return _run_keypoints2d(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
