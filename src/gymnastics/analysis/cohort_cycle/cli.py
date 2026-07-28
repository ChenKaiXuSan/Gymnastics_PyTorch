"""Command-line interface for cohort and repeated-cycle analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .cohorts import load_cohort_records, sha256_file
from .config import load_config
from .folds import (
    build_crossfit_folds,
    load_fold_split,
    write_crossfit_artifacts,
)
from .features import extract_publication_features
from .oof import OOFRun, collect_oof_cycles, publish_oof_cycles
from .report import render_report
from .statistics import analyze_feature_artifacts


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
        if stage == "audit":
            child.add_argument("--fold", action="append", type=int)
            child.add_argument("--check-only", action="store_true")
            child.add_argument("--pilot", action="store_true")
            child.add_argument("--publication-root")
            child.add_argument("--strict", action="store_true")
            child.add_argument("--write-final-audit", action="store_true")
        if stage == "features":
            child.add_argument("--person", action="append")
            child.add_argument("--pilot", action="store_true")
            child.add_argument("--publication-root")
            child.add_argument("--output-root")
        if stage == "analyze":
            child.add_argument("--pilot", action="store_true")
            child.add_argument("--feature-root")
            child.add_argument("--output-root")
            child.add_argument("--permutations", type=int)
            child.add_argument("--no-random-slope", action="store_true")
        if stage == "assets":
            child.add_argument("--pilot", action="store_true")
            child.add_argument("--feature-root")
            child.add_argument("--statistics-root")
            child.add_argument("--output-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one cohort-cycle pipeline stage."""
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    if args.command == "folds":
        return _cmd_folds(config)
    if args.command == "audit":
        return _cmd_audit(args, config)
    if args.command == "features":
        return _cmd_features(args, config)
    if args.command == "analyze":
        return _cmd_analyze(args, config)
    if args.command == "assets":
        return _cmd_assets(args, config)
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


def _cmd_audit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    paths = _mapping(config, "paths")
    crossfit = _mapping(config, "crossfit")
    fold_output = Path(_string(paths, "fold_output"))
    rotation_root = Path(_string(paths, "rotation_aware_root"))
    cohort_output = Path(_string(paths, "cohort_output"))

    registry = _load_json(fold_output / "run_registry.json")
    crossfit_manifest = _load_json(
        fold_output / "crossfit_manifest.json"
    )
    raw_runs = registry.get("runs")
    cohorts = crossfit_manifest.get("cohorts")
    if not isinstance(raw_runs, Mapping) or not isinstance(cohorts, Mapping):
        raise ValueError("crossfit registry or cohort manifest is incomplete")

    selected_folds = (
        sorted(set(args.fold))
        if args.fold
        else sorted(int(key) for key in raw_runs)
    )
    runs: list[OOFRun] = []
    for fold in selected_folds:
        entry = raw_runs.get(f"{fold:02d}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"run registry has no fold {fold:02d}")
        run_id = str(entry["run_id"])
        runs.append(
            OOFRun(
                outer_fold=fold,
                run_id=run_id,
                seed=int(entry.get("seed", 0)),
                checkpoint=(
                    rotation_root
                    / "runs"
                    / run_id
                    / "checkpoints"
                    / "best.pt"
                ),
                split_manifest=fold_output / str(entry["split_file"]),
                inference_root=rotation_root / "inference" / run_id,
            )
        )

    selected_people = {
        person_id
        for run in runs
        for person_id in load_fold_split(run.split_manifest).test
    }
    full_run = len(selected_folds) == len(raw_runs)
    expected_people = (
        set(str(person_id) for person_id in cohorts)
        if full_run
        else selected_people
    )
    expected_people_count = int(
        crossfit.get("expected_people", len(expected_people))
    )
    if full_run and len(expected_people) != expected_people_count:
        raise ValueError(
            "configured expected person count does not match cohort manifest"
        )
    expected_cycles = (
        int(crossfit["expected_cycles"])
        if full_run
        else _expected_cycle_count(runs)
    )
    cycles, audit = collect_oof_cycles(
        runs,
        {str(key): str(value) for key, value in cohorts.items()},
        expected_people=expected_people,
        expected_cycles=expected_cycles,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.check_only:
        return 0

    if args.publication_root:
        publication_root = Path(args.publication_root)
    elif args.pilot or not full_run:
        suffix = "_".join(f"{fold:02d}" for fold in selected_folds)
        publication_root = cohort_output / "pilot" / f"oof_seed0_f{suffix}"
    else:
        publication_root = cohort_output / "oof_seed0"
    publish_oof_cycles(cycles, audit, publication_root)
    return 0


def _expected_cycle_count(runs: list[OOFRun]) -> int:
    count = 0
    for run in runs:
        payload = torch.load(
            run.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        provenance = payload.get("provenance", {})
        cache_manifests = provenance.get("cache_manifests", {})
        split = load_fold_split(run.split_manifest)
        for person_id in split.test:
            identity = cache_manifests.get(person_id, {})
            trials = identity.get("trials", [])
            if not isinstance(trials, list):
                raise ValueError(
                    f"invalid cache trials for person {person_id}"
                )
            count += len(trials)
    return count


def _cmd_features(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    paths = _mapping(config, "paths")
    quality = config.get("quality_control", {})
    if not isinstance(quality, Mapping):
        raise ValueError("quality_control must be a mapping")
    cohort_output = Path(_string(paths, "cohort_output"))
    publication = (
        Path(args.publication_root)
        if args.publication_root
        else (
            cohort_output / "pilot" / "oof_seed0_f00"
            if args.pilot
            else cohort_output / "oof_seed0"
        )
    )
    output = (
        Path(args.output_root)
        if args.output_root
        else (
            cohort_output / "pilot" / "features_seed0_f00"
            if args.pilot
            else cohort_output / "analysis" / "features"
        )
    )
    summary = extract_publication_features(
        publication,
        output,
        people=set(args.person) if args.person else None,
        phase_points=int(quality.get("phase_points", 101)),
        minimum_person_cycles=int(
            quality.get("minimum_person_cycles", 4)
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_analyze(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    paths = _mapping(config, "paths")
    statistics = _mapping(config, "statistics")
    cohort_output = Path(_string(paths, "cohort_output"))
    feature_root = (
        Path(args.feature_root)
        if args.feature_root
        else (
            cohort_output / "pilot" / "features_seed0_f00"
            if args.pilot
            else cohort_output / "analysis" / "features"
        )
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else (
            cohort_output / "pilot" / "statistics_seed0_f00"
            if args.pilot
            else cohort_output / "analysis" / "statistics"
        )
    )
    configured_permutations = int(statistics.get("permutations", 10000))
    permutations = (
        args.permutations
        if args.permutations is not None
        else min(configured_permutations, 499)
        if args.pilot
        else configured_permutations
    )
    raw_log = statistics.get("log_transform", [])
    if not isinstance(raw_log, list):
        raise ValueError("statistics.log_transform must be a list")
    summary = analyze_feature_artifacts(
        feature_root,
        output_root,
        permutations=permutations,
        seed=int(statistics.get("permutation_seed", 20260728)),
        try_random_slope=not args.no_random_slope,
        log_outcomes={str(value) for value in raw_log},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_assets(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    paths = _mapping(config, "paths")
    cohort_output = Path(_string(paths, "cohort_output"))
    feature_root = (
        Path(args.feature_root)
        if args.feature_root
        else (
            cohort_output / "pilot" / "features_seed0_f00"
            if args.pilot
            else cohort_output / "analysis" / "features"
        )
    )
    statistics_root = (
        Path(args.statistics_root)
        if args.statistics_root
        else (
            cohort_output / "pilot" / "statistics_seed0_f00_v2"
            if args.pilot
            else cohort_output / "analysis" / "statistics"
        )
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else (
            cohort_output / "pilot" / "report_seed0_f00"
            if args.pilot
            else cohort_output / "analysis" / "report"
        )
    )
    summary = render_report(feature_root, statistics_root, output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return value


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
