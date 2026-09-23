"""``python -m fusion benchmark-fit3d``: inspect the release and select the views.

    inspect        subjects, exercises, repetitions and SAM3D cache coverage
    select-views   write ``selected_views.json`` (face/side pair per sequence)

Both stages only read the release and the cache; SAM3D inference itself runs
outside this repository (the per-video ``derived/normal_camera`` jobs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from common.paths import PROJECT_ROOT

from .dataset import discover_sequences, load_reference, load_repetitions, repetition_bounds, select_views
from .sam3d import cached_frames
from .schema import SelectedViews

DEFAULT_CONFIG = Path("src/configs/benchmarks/fit3d.yaml")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read the benchmark configuration, resolving ``${oc.env:...}`` interpolations."""
    from omegaconf import OmegaConf

    import common.paths  # noqa: F401  (exports the dataset roots when unset)

    payload = OmegaConf.to_container(OmegaConf.load(_resolve(path)), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Fit3D benchmark config must be a mapping")
    return payload


def read_selected_views(path: str | Path) -> dict[tuple[str, str], SelectedViews]:
    """Load ``selected_views.json`` keyed by ``(subject, sequence)``."""
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    entries = payload["sequences"] if isinstance(payload, Mapping) else payload
    views = [SelectedViews.from_dict(entry) for entry in entries]
    return {(view.subject_key, view.sequence_key): view for view in views}


def write_selected_views(path: Path, views: Sequence[SelectedViews], *, target_separation_deg: float) -> Path:
    """Write ``selected_views.json`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_separation_deg": float(target_separation_deg),
        "sequences": [view.to_dict() for view in sorted(views, key=lambda v: (v.subject_key, v.sequence_key))],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _sequences(config: Mapping, subjects: Sequence[str] | None, actions: Sequence[str] | None):
    dataset = dict(config.get("dataset") or {})
    return discover_sequences(
        _resolve(str(config["paths"]["dataset_root"])),
        split=str(dataset.get("split", "train")),
        subjects=subjects or dataset.get("subjects"),
        actions=actions or dataset.get("actions"),
    )


def run_inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    derived_root = _resolve(str(config["paths"]["sam3d_derived_root"]))
    split = str(dict(config.get("dataset") or {}).get("split", "train"))
    sequences = _sequences(config, args.subjects, args.actions)
    by_subject: dict[str, dict[str, int]] = {}
    for sequence in sequences:
        stats = by_subject.setdefault(sequence.subject, {"sequences": 0, "with_reps": 0, "cycles": 0, "cached": 0, "frames": 0})
        marks = load_repetitions(sequence.root).get(sequence.action, ())
        bounds = repetition_bounds(marks, sequence.frames)
        cached = all(cached_frames(derived_root, sequence, camera, split=split) > 0 for camera in sequence.cameras)
        stats["sequences"] += 1
        stats["with_reps"] += int(bool(bounds))
        stats["cycles"] += len(bounds)
        stats["cached"] += int(cached)
        stats["frames"] += int(sequence.frames)
    total = {key: sum(stats[key] for stats in by_subject.values()) for key in ("sequences", "with_reps", "cycles", "cached", "frames")}
    print(f"{'subject':>8} {'sequences':>10} {'with reps':>10} {'cycles':>7} {'cached':>7} {'frames':>9}")
    for subject in sorted(by_subject):
        stats = by_subject[subject]
        print(f"{subject:>8} {stats['sequences']:10d} {stats['with_reps']:10d} {stats['cycles']:7d} {stats['cached']:7d} {stats['frames']:9d}")
    print(f"{'total':>8} {total['sequences']:10d} {total['with_reps']:10d} {total['cycles']:7d} {total['cached']:7d} {total['frames']:9d}")
    return 0


def run_select_views(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pairing = dict(config.get("pairing") or {})
    target = float(args.target_separation_deg if args.target_separation_deg is not None else pairing.get("target_separation_deg", 90.0))
    up_axis = int(pairing.get("world_up_axis", 2))
    sequences = _sequences(config, args.subjects, args.actions)
    views: list[SelectedViews] = []
    for sequence in sequences:
        reference = load_reference(sequence.reference_path)
        bounds = repetition_bounds(load_repetitions(sequence.root).get(sequence.action, ()), sequence.frames)
        views.append(select_views(sequence, reference, target_separation_deg=target, up_axis=up_axis, reps=len(bounds)))
    path = write_selected_views(_resolve(str(config["paths"]["views_path"])), views, target_separation_deg=target)
    separations = sorted(view.separation_deg for view in views)
    median = separations[len(separations) // 2] if separations else float("nan")
    pairs = sorted({(view.view_a, view.view_b) for view in views})
    print(f"[fit3d/select-views] {len(views)} sequences, median separation {median:.0f} deg, {len(pairs)} distinct pairs -> {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fusion benchmark-fit3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, handler in (("inspect", run_inspect), ("select-views", run_select_views)):
        stage = sub.add_parser(name, help=name)
        stage.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        stage.add_argument("--subjects", nargs="*", default=None, help="s03 s04 ... (default: every subject of the config)")
        stage.add_argument("--actions", nargs="*", default=None, help="exercise names (default: all)")
        if name == "select-views":
            stage.add_argument("--target-separation-deg", type=float, default=None, help="desired azimuth separation of the two views (default: from the config)")
        stage.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
