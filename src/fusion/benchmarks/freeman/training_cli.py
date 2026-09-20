"""CLI for subject-disjoint FreeMan training of the rotation-aware model.

Stages::

    python -m fusion benchmark-freeman-train prepare-cache --config src/configs/archive/rotation_aware_freeman.yaml
    python -m fusion benchmark-freeman-train write-folds   --config ...
    python -m fusion rotation-aware train --config src/configs/archive/rotation_aware_freeman.yaml \
        --fold src/configs/shared/folds/freeman/fold_01.json --run-id freeman_fold01_a6_e12_s0 --ablation A6
    python -m fusion benchmark-freeman-train evaluate --config ... --seed 0
    python -m fusion benchmark-freeman-train compare  --config ... --seed 0

``prepare-cache`` and ``write-folds`` read only SAM3D view predictions and the
benchmark manifests. ``evaluate`` runs each fold's checkpoint on its held-out
test subjects and scores the sessions with the zero-shot benchmark evaluator.
``compare`` merges those rows with the existing zero-shot session metrics and
writes subject-balanced tables plus paired subject-level statistics.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from common.paths import PROJECT_ROOT
from fusion.archive.rotation_aware.cli import load_config as load_rotation_config

from .evaluation import SessionMetrics, _holm_adjust, aggregate_metrics
from .training import (
    FoldRun,
    METHOD_PREFIX,
    build_training_cache,
    evaluate_deterministic_methods,
    evaluate_trained_family,
    evaluate_zero_shot_checkpoint,
    fold_runs_from_checkpoints,
    load_session_metric_rows,
    make_subject_disjoint_folds,
    write_subject_disjoint_folds,
)

DEFAULT_CONFIG = "src/configs/archive/rotation_aware_freeman.yaml"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _section(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("freeman_training")
    if not isinstance(section, Mapping) or not section:
        raise ValueError("config requires a freeman_training mapping")
    return dict(section)


def _benchmark_settings(section: Mapping[str, Any]) -> tuple[tuple[float, ...], float]:
    import yaml

    benchmark_config = _project_path(section["benchmark_config"])
    raw = yaml.safe_load(benchmark_config.read_text(encoding="utf-8"))
    thresholds = tuple(float(v) for v in raw["evaluation"]["pck_thresholds_mm"])
    scale = float(raw["dataset"]["reference_scale_to_m"])
    return thresholds, scale


def _folds(section: Mapping[str, Any]):
    return make_subject_disjoint_folds(
        evaluation_subjects=section["evaluation_subjects"],
        test_groups=section["test_groups"],
        val_subjects=section["validation_subjects"],
    )


def _run_id(section: Mapping[str, Any], fold_name: str, seed: int) -> str:
    return str(section["run_id_template"]).format(fold=fold_name, seed=int(seed))


def _family(section: Mapping[str, Any], seed: int) -> str:
    return str(section["family_template"]).format(seed=int(seed))


def _cmd_prepare_cache(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    section = _section(config)
    data = config.get("data", {})
    cache_root = _project_path(
        data.get("cache_dir", Path(config["paths"]["output_root"]) / "cache")
    )
    subjects = sorted(
        {int(s) for s in section["evaluation_subjects"]}
        | {int(s) for s in section["validation_subjects"]}
    )
    if args.subject:
        subjects = sorted({int(s) for s in args.subject})
    written = build_training_cache(
        _project_path(section["benchmark_root"]),
        subjects,
        cache_root,
        config_metadata={
            "config": str(args.config),
            "window": dict(config.get("window", {})),
            "data": dict(data),
            "training_source": "freeman_subject_disjoint",
        },
    )
    for person, manifest in written.items():
        print(f"[freeman-train] cached person_{person}: {manifest}")
    return 0


def _cmd_write_folds(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    section = _section(config)
    fold_root = _project_path(config["paths"]["fold_root"])
    paths = write_subject_disjoint_folds(_folds(section), fold_root)
    for path in paths:
        print(f"[freeman-train] wrote {path}")
    return 0


def _cmd_plan(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    """Print the train commands for every fold so a scheduler can submit them."""
    section = _section(config)
    fold_root = Path(config["paths"]["fold_root"])
    for fold in _folds(section):
        run_id = _run_id(section, fold.name, args.seed)
        print(
            "python -m fusion rotation-aware train "
            f"--config {args.config} --fold {fold_root / (fold.name + '.json')} "
            f"--run-id {run_id} --ablation {section.get('ablation', 'A6')}"
        )
    return 0


def _fold_runs(section: Mapping[str, Any], config: Mapping[str, Any], seed: int) -> tuple[FoldRun, ...]:
    run_root = _project_path(config["paths"]["output_root"]) / "runs"
    run_ids = [_run_id(section, fold.name, seed) for fold in _folds(section)]
    return fold_runs_from_checkpoints(run_root, run_ids)


def _cmd_evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    section = _section(config)
    thresholds, scale = _benchmark_settings(section)
    runs = _fold_runs(section, config, args.seed)
    subjects = [int(s) for s in (args.subject or section["evaluation_subjects"])]
    rows = evaluate_trained_family(
        benchmark_root=_project_path(section["benchmark_root"]),
        output_root=_project_path(section["evaluation_output_root"]),
        rotation_config=_project_path(args.config),
        family=_family(section, args.seed),
        runs=runs,
        subjects=subjects,
        thresholds_mm=thresholds,
        reference_scale_to_m=scale,
        progress=lambda text: print(f"[freeman-train] {text}", flush=True),
    )
    print(f"[freeman-train] evaluated {len(rows)} session rows")
    return 0


def _cmd_baselines(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    from fusion.baselines.experiment_matrix import BASELINE_METHODS

    section = _section(config)
    thresholds, scale = _benchmark_settings(section)
    subjects = [int(s) for s in (args.subject or section["evaluation_subjects"])]
    methods = tuple(args.method or BASELINE_METHODS)
    rows = evaluate_deterministic_methods(
        benchmark_root=_project_path(section["benchmark_root"]),
        output_root=_project_path(section["baselines_output_root"]),
        methods=methods,
        subjects=subjects,
        thresholds_mm=thresholds,
        reference_scale_to_m=scale,
        progress=lambda text: print(f"[freeman-baselines] {text}", flush=True),
    )
    print(f"[freeman-baselines] evaluated {len(rows)} session rows for {methods}")
    return 0


def _cmd_zero_shot(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    """Score an extra private-data checkpoint (e.g. B1) zero-shot on FreeMan."""
    import yaml

    section = _section(config)
    thresholds, scale = _benchmark_settings(section)
    subjects = [int(s) for s in (args.subject or section["evaluation_subjects"])]
    rotation_config = _project_path(args.rotation_config)
    raw = yaml.safe_load(rotation_config.read_text(encoding="utf-8"))
    checkpoint = (
        _project_path(args.checkpoint)
        if args.checkpoint
        else _project_path(raw["paths"]["output_root"]) / "runs" / args.run_id / "checkpoints" / "best.pt"
    )
    rows = evaluate_zero_shot_checkpoint(
        benchmark_root=_project_path(section["benchmark_root"]),
        output_root=_project_path(section["baselines_output_root"]),
        rotation_config=rotation_config,
        run_id=args.run_id,
        checkpoint=checkpoint,
        subjects=subjects,
        thresholds_mm=thresholds,
        reference_scale_to_m=scale,
        progress=lambda text: print(f"[freeman-zero-shot] {text}", flush=True),
    )
    print(f"[freeman-zero-shot] evaluated {len(rows)} session rows for rotation_aware:{args.run_id}")
    return 0


def _subject_means(rows: Sequence[SessionMetrics], metric: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject_id": [row.subject_id for row in rows],
            "method": [row.method for row in rows],
            metric: [getattr(row, metric) for row in rows],
        }
    )
    return frame.groupby(["subject_id", "method"], as_index=False)[metric].mean()


def paired_subject_comparison(
    rows: Sequence[SessionMetrics],
    *,
    candidate: str,
    references: Sequence[str],
    metrics: Sequence[str] = ("sim3_mpjpe_mm", "pa_mpjpe_mm"),
    seed: int = 20260726,
    bootstrap_samples: int = 10000,
) -> pd.DataFrame:
    """Subject-level paired differences candidate minus each reference method."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for metric in metrics:
        table = _subject_means(rows, metric).pivot(
            index="subject_id", columns="method", values=metric
        )
        if candidate not in table.columns:
            raise ValueError(f"candidate method {candidate} has no rows")
        p_values: list[float] = []
        for reference in references:
            if reference not in table.columns:
                raise ValueError(f"reference method {reference} has no rows")
            paired = table[[candidate, reference]].dropna()
            diff = (paired[candidate] - paired[reference]).to_numpy(dtype=np.float64)
            n = len(diff)
            if n < 2:
                raise ValueError("paired comparison requires at least two subjects")
            resampled = rng.choice(diff, size=(bootstrap_samples, n), replace=True).mean(axis=1)
            low, high = np.quantile(resampled, [0.025, 0.975])
            if np.allclose(diff, 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
            p_values.append(p_value)
            records.append(
                {
                    "metric": metric,
                    "candidate": candidate,
                    "reference": reference,
                    "subjects": int(n),
                    "candidate_mean": float(paired[candidate].mean()),
                    "reference_mean": float(paired[reference].mean()),
                    "difference": float(diff.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "improved_subjects": int((diff < 0).sum()),
                    "p_wilcoxon": p_value,
                }
            )
        adjusted = _holm_adjust(p_values)
        for record, value in zip(records[-len(references):], adjusted):
            record["p_holm"] = float(value)
    return pd.DataFrame(records)


def _cmd_compare(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    section = _section(config)
    subjects = [int(s) for s in section["evaluation_subjects"]]
    family = _family(section, args.seed)
    candidate = f"{METHOD_PREFIX}:{family}"
    trained_root = _project_path(section["evaluation_output_root"])
    zero_shot_root = _project_path(section["benchmark_root"])
    trained_rows = load_session_metric_rows(
        trained_root / "evaluation" / "session_metrics", subjects, methods=[candidate]
    )
    references = list(section.get("comparison_methods", ()))
    zero_shot_rows = load_session_metric_rows(
        zero_shot_root / "evaluation" / "session_metrics", subjects
    )
    rows = (*zero_shot_rows, *trained_rows)
    trained_sessions = {(r.subject_id, r.session_id) for r in trained_rows}
    for reference in references:
        reference_sessions = {
            (r.subject_id, r.session_id) for r in zero_shot_rows if r.method == reference
        }
        if reference_sessions != trained_sessions:
            raise ValueError(
                f"session sets differ between {candidate} and {reference}: "
                f"{len(trained_sessions)} vs {len(reference_sessions)}"
            )
    tables = aggregate_metrics(rows)
    comparison = paired_subject_comparison(
        rows,
        candidate=candidate,
        references=references,
        seed=int(section.get("random_seed", 20260726)),
        bootstrap_samples=int(section.get("bootstrap_samples", 10000)),
    )
    report_root = trained_root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    tables.by_method.to_csv(report_root / f"metrics_by_method_{family}.csv", index=False)
    tables.by_subject.to_csv(report_root / f"metrics_by_subject_{family}.csv", index=False)
    comparison.to_csv(report_root / f"paired_comparison_{family}.csv", index=False)
    columns = ["method", "sim3_mpjpe_mm", "pa_mpjpe_mm", "root_mpjpe_mm", "coverage"]
    by_method = tables.by_method[
        tables.by_method["classification"] == "VALID"
    ][columns].sort_values("pa_mpjpe_mm")
    summary = {
        "family": family,
        "candidate": candidate,
        "subjects": subjects,
        "sessions": len(trained_sessions),
        "by_method": by_method.to_dict(orient="records"),
        "paired": comparison.to_dict(orient="records"),
    }
    (report_root / f"summary_{family}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(by_method.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fusion benchmark-freeman-train")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-cache", "write-folds", "plan", "evaluate", "baselines", "zero-shot", "compare"):
        child = commands.add_parser(name)
        child.add_argument("--config", default=argparse.SUPPRESS)
        if name in {"prepare-cache", "evaluate", "baselines", "zero-shot"}:
            child.add_argument("--subject", type=int, nargs="+")
        if name == "baselines":
            child.add_argument("--method", action="append")
        if name == "zero-shot":
            child.add_argument("--run-id", required=True)
            child.add_argument("--checkpoint")
            child.add_argument("--rotation-config", default="src/configs/archive/rotation_aware.yaml")
        if name in {"plan", "evaluate", "compare"}:
            child.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    config = load_rotation_config(_project_path(args.config))
    handlers = {
        "prepare-cache": _cmd_prepare_cache,
        "write-folds": _cmd_write_folds,
        "plan": _cmd_plan,
        "evaluate": _cmd_evaluate,
        "baselines": _cmd_baselines,
        "zero-shot": _cmd_zero_shot,
        "compare": _cmd_compare,
    }
    return int(handlers[args.command](args, config))


if __name__ == "__main__":
    sys.exit(main())
