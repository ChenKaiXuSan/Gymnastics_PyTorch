"""Command-line interface for cohort and repeated-cycle analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from .cohorts import load_cohort_records, sha256_file
from .config import load_config
from .folds import (
    build_crossfit_folds,
    load_fold_split,
    write_crossfit_artifacts,
)


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
    """Run one cohort-cycle pipeline stage."""
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    if args.command == "folds":
        return _cmd_folds(config)
    raise NotImplementedError(f"stage is not implemented yet: {args.command}")


def _cmd_folds(config: Mapping[str, Any]) -> int:
    paths = _mapping(config, "paths")
    crossfit = _mapping(config, "crossfit")
    student_mapping = Path(_string(paths, "student_mapping"))
    organization_mapping = Path(_string(paths, "organization_mapping"))
    fold0_split = Path(_string(paths, "fold0_split"))
    fold_output = Path(_string(paths, "fold_output"))
    split_seed = int(crossfit.get("split_seed", 20260728))

    records = load_cohort_records(student_mapping, organization_mapping)
    folds = build_crossfit_folds(
        records,
        load_fold_split(fold0_split),
        seed=split_seed,
    )
    write_crossfit_artifacts(
        folds,
        records,
        fold_output,
        seed=split_seed,
        source_hashes={
            "organization_mapping": sha256_file(organization_mapping),
            "student_mapping": sha256_file(student_mapping),
        },
    )
    return 0


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"cohort-cycle config requires mapping: {key}")
    return value


def _string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"cohort-cycle config requires path: {key}")
    return value
