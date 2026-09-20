"""``python -m fusion benchmark-sportspose`` -- SportsPose preparation stages.

    inspect       count clips / subjects / activities of the release
    select-views  choose the face-like and side-like camera per subject and
                  sequence (day + activity) and write paths.views_path
    infer         run SAM3D-Body on the two selected views of every clip
                  (resumable; --subjects limits one job to a few subjects)

Cycle records and training use the cache these stages produce:
``python -m cycle_alignment cycles sportspose`` and ``python -m fusion train data=sportspose``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from common.paths import PROJECT_ROOT

from .dataset import discover_clips, group_clips, load_calibration, select_views
from .schema import SelectedViews, SportsPoseClip

DEFAULT_CONFIG = Path("src/configs/benchmarks/sportspose.yaml")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_config(path: Path) -> Mapping[str, object]:
    """Load the benchmark YAML, resolving ``${oc.env:...}`` interpolations."""
    from omegaconf import OmegaConf

    import common.paths  # noqa: F401  (exports the dataset roots when unset)

    payload = OmegaConf.to_container(OmegaConf.load(_resolve(path)), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("SportsPose benchmark config must be a mapping")
    return payload


def _clips(config: Mapping[str, object], args: argparse.Namespace) -> list[SportsPoseClip]:
    dataset = dict(config.get("dataset") or {})
    subjects = args.subjects or dataset.get("subjects")
    return discover_clips(
        _resolve(str(config["paths"]["dataset_root"])),
        days=args.days or dataset.get("days"),
        subjects=subjects,
        activities=args.activities or dataset.get("activities"),
    )


def read_selected_views(path: Path) -> dict[tuple[str, str], SelectedViews]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for entry in payload["groups"]:
        views = SelectedViews.from_dict(entry)
        result[(views.subject_key, views.sequence_key)] = views
    return result


def _inspect(config: Mapping[str, object], args: argparse.Namespace) -> int:
    clips = _clips(config, args)
    subjects = Counter(clip.subject_key for clip in clips)
    activities = Counter(clip.activity for clip in clips)
    frames = sum(clip.frames for clip in clips)
    print(f"[sportspose] {len(clips)} clips, {len(subjects)} subjects, {frames} reference frames ({frames / 90.0 / 60.0:.1f} min at 90 fps)")
    for activity, count in sorted(activities.items()):
        print(f"  {activity:16s} {count} clips")
    for subject, count in sorted(subjects.items()):
        print(f"  {subject:16s} {count} clips")
    return 0


def _select_views(config: Mapping[str, object], args: argparse.Namespace) -> int:
    clips = _clips(config, args)
    options = dict(config.get("views") or {})
    groups = group_clips(clips)
    selected: list[SelectedViews] = []
    failures: list[str] = []
    for (subject_key, sequence_key), group in groups.items():
        cameras = load_calibration(group[0].joints_path.parents[1])
        try:
            views = select_views(subject_key, sequence_key, group, cameras, target_separation_deg=float(options.get("target_separation_deg", 90.0)), max_frontal_deg=float(options.get("max_frontal_deg", 60.0)))
        except ValueError as error:
            failures.append(str(error))
            continue
        selected.append(views)
        print(f"  {subject_key:5s} {sequence_key:24s} A={views.view_a} ({views.azimuth_a_deg:6.1f} deg)  B={views.view_b} ({views.azimuth_b_deg:6.1f} deg)  sep {views.separation_deg:5.1f}")
    out = _resolve(str(config["paths"]["views_path"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target_separation_deg": options.get("target_separation_deg", 90.0), "groups": [v.to_dict() for v in selected], "failures": failures}, indent=2), encoding="utf-8")
    print(f"[sportspose] {len(selected)} groups selected, {len(failures)} failed -> {out}")
    return 1 if failures else 0


def _infer(config: Mapping[str, object], args: argparse.Namespace) -> int:
    from .sam3d import infer_clips

    clips = _clips(config, args)
    views = read_selected_views(_resolve(str(config["paths"]["views_path"])))
    groups = group_clips(clips)
    missing = [key for key in groups if key not in views]
    if missing:
        raise SystemExit(f"[sportspose] {len(missing)} groups have no selected views (run select-views first): {missing[:5]}")
    cameras_by_day_subject = {}
    for (_, _), group in groups.items():
        cameras_by_day_subject.setdefault((group[0].day, group[0].subject), load_calibration(group[0].joints_path.parents[1]))
    sam3d = dict(config.get("sam3d") or {})
    dataset = dict(config.get("dataset") or {})
    summaries = infer_clips(
        clips,
        {key: (v.view_a, v.view_b) for key, v in views.items()},
        cameras_by_day_subject,
        cache_root=_resolve(str(config["paths"]["sam3d_cache_root"])),
        config_path=_resolve(str(config["paths"]["sam3d_config"])),
        device=int(args.device if args.device is not None else sam3d.get("device", 0)),
        frame_stride=int(dataset.get("frame_stride", 3)),
        force=bool(args.force),
        accepted_config_hashes=tuple(str(h) for h in sam3d.get("identity_aliases") or ()),
    )
    reused = sum(s.reused for s in summaries)
    failed = sum(s.failed for s in summaries)
    frames = sum(s.frames for s in summaries)
    print(f"[sportspose] {len(summaries)} clip-views ({reused} reused), {frames} frames, {failed} failed detections")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fusion benchmark-sportspose", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="stage", required=True)

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--days", nargs="*", default=None)
        p.add_argument("--subjects", nargs="*", default=None, help="S00 S01 ... (applies to every day)")
        p.add_argument("--activities", nargs="*", default=None)

    selection(sub.add_parser("inspect", help="count clips, subjects and activities"))
    selection(sub.add_parser("select-views", help="choose the face-like / side-like camera per subject and sequence"))
    infer = sub.add_parser("infer", help="SAM3D-Body on the two selected views of every clip")
    infer.add_argument("--device", type=int, default=None)
    infer.add_argument("--force", action="store_true")
    selection(infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    if args.stage == "inspect":
        return _inspect(config, args)
    if args.stage == "select-views":
        return _select_views(config, args)
    return _infer(config, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
