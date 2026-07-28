"""Unified command-line entry point for the gymnastics pipeline."""

from __future__ import annotations

import argparse
from importlib import import_module
import os
from pathlib import Path
import sys
from typing import Sequence

from gymnastics.common.paths import LOCAL_ROOT


_COMMANDS = {
    "sam3d": ("gymnastics.sam3d.main", "main", False),
    "align": ("gymnastics.alignment.main", "cli_main", True),
    "triangulate:run": (
        "gymnastics.triangulation.sam3d_from_split_cycle",
        "main",
        False,
    ),
    "triangulate:estimate-extrinsics": (
        "gymnastics.triangulation.estimate_extrinsics",
        "main",
        False,
    ),
    "fuse:deterministic": (
        "gymnastics.fusion.deterministic.experiment_matrix",
        "main",
        False,
    ),
    "fuse:rotation-aware": (
        "gymnastics.fusion.rotation_aware.cli",
        "main",
        True,
    ),
    "classify": ("gymnastics.classification.train", "init_params", False),
    "analyze": ("gymnastics.analysis.main", "main", False),
    "cohort-cycle": (
        "gymnastics.analysis.cohort_cycle.cli",
        "main",
        True,
    ),
    "calibrate": ("gymnastics.calibration.main", "main", False),
    "benchmark:freeman": (
        "gymnastics.benchmarks.freeman.cli",
        "main",
        True,
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gymnastics",
        description="Multi-view gymnastics motion analysis pipeline",
    )
    commands = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("sam3d", "run SAM3D-Body extraction"),
        ("align", "align views and segment motion cycles"),
        ("classify", "train or evaluate motion classifiers"),
        ("analyze", "generate metrics and analysis outputs"),
        ("calibrate", "calibrate cameras"),
    ):
        commands.add_parser(
            name,
            help=help_text,
            add_help=name in {"classify", "analyze", "calibrate"},
        )
    commands.add_parser(
        "cohort-cycle",
        help="run out-of-fold cohort and repeated-cycle analysis",
        add_help=False,
    )

    triangulate = commands.add_parser(
        "triangulate",
        help="estimate geometry or build the pseudo-reference",
    )
    triangulation_commands = triangulate.add_subparsers(dest="triangulation_command")
    triangulation_commands.add_parser(
        "run",
        help="build the triangulated pseudo-reference",
        add_help=False,
    )
    triangulation_commands.add_parser(
        "estimate-extrinsics",
        help="estimate per-person camera extrinsics",
        add_help=False,
    )

    fuse = commands.add_parser("fuse", help="run multi-view fusion")
    fusion_commands = fuse.add_subparsers(dest="fusion_command")
    fusion_commands.add_parser(
        "deterministic",
        help="run the deterministic comparison matrix",
        add_help=False,
    )
    fusion_commands.add_parser(
        "rotation-aware",
        help="run the self-supervised paper method",
        add_help=False,
    )
    benchmark = commands.add_parser(
        "benchmark",
        help="run external zero-shot benchmarks",
    )
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command")
    benchmark_commands.add_parser(
        "freeman",
        help="run the full-release FreeMan benchmark",
        add_help=False,
    )
    return parser


def _target_key(args: argparse.Namespace) -> str | None:
    if args.command == "benchmark":
        if args.benchmark_command is None:
            return None
        return f"benchmark:{args.benchmark_command}"
    if args.command == "triangulate":
        mode = args.triangulation_command or "run"
        return f"triangulate:{mode}"
    if args.command != "fuse":
        return args.command
    if args.fusion_command is None:
        return None
    return f"fuse:{args.fusion_command}"


def _invoke(target: tuple[str, str, bool], argv: list[str]) -> int:
    module_name, function_name, accepts_argv = target
    function = getattr(import_module(module_name), function_name)
    if accepts_argv:
        result = function(argv)
    else:
        previous = sys.argv
        sys.argv = [f"gymnastics {module_name.rsplit('.', 1)[-1]}", *argv]
        try:
            result = function()
        finally:
            sys.argv = previous
    return int(result) if isinstance(result, int) else 0


def _configure_runtime_cache() -> None:
    credential_cache = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    os.environ.setdefault(
        "HF_HOME",
        str(credential_cache / "huggingface"),
    )
    cache_root = LOCAL_ROOT / "cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def main(argv: Sequence[str] | None = None) -> int:
    _configure_runtime_cache()
    parser = _parser()
    args, remainder = parser.parse_known_args(list(argv) if argv is not None else None)
    key = _target_key(args)
    if key is None:
        parser.print_help()
        return 0
    return _invoke(_COMMANDS[key], remainder)
