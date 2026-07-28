"""Command-line interface for cohort and repeated-cycle analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


STAGES = ("folds", "audit", "features", "analyze", "assets")


def make_parser() -> argparse.ArgumentParser:
    """Build the public cohort-cycle command parser."""
    parser = argparse.ArgumentParser(prog="gymnastics cohort-cycle")
    commands = parser.add_subparsers(dest="command", required=True)
    for stage in STAGES:
        child = commands.add_parser(stage)
        child.add_argument(
            "--config",
            default="configs/analysis/cohort_cycle.yaml",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the requested stage; stage handlers are added incrementally."""
    make_parser().parse_args(list(argv) if argv is not None else None)
    return 0
