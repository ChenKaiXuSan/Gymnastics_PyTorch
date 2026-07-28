"""Reproducible, cohort-aware project result aggregation.

The functions in this module keep learned-model test evidence, descriptive
all-person results, and validation-only diagnostics separate.  The command-line
generator is added below the pure aggregation layer so the statistical rules can
be tested without access to the project data directories.
"""

from __future__ import annotations

import ast
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

DEFAULT_LEARNED_METRICS = Path(
    "local/runs/fuse_rotation_aware/evaluation/"
    "all137_a4_e100_seed0+all137_a5_e100_seed0+all137_a6_e100_seed0+"
    "all137_a7_e100_seed0+all137_a8_e100_seed0+all137_a9_e100_seed0/"
    "metrics_by_person.csv"
)
DEFAULT_SPLIT_MANIFEST = Path(
    "local/runs/fuse_rotation_aware/runs/all137_a6_e100_seed0/split_manifest.json"
)
DEFAULT_CLASSIFICATION_ROOT = Path("local/runs/train")
DEFAULT_OUTPUT_DIR = Path("local/runs/analysis/project_results")


def load_split_manifest(path: Path) -> dict[str, set[str]]:
    """Load and validate mutually exclusive train/validation/test person sets."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("train", "val", "test")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"split manifest is missing: {', '.join(missing)}")

    splits = {name: {str(person) for person in payload[name]} for name in required}
    seen: set[str] = set()
    for name in required:
        duplicates = seen & splits[name]
        if duplicates:
            joined = ", ".join(sorted(duplicates, key=_person_sort_key))
            raise ValueError(f"people occur in more than one split: {joined}")
        seen.update(splits[name])
    return splits


def _person_sort_key(person_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(person_id))
    except ValueError:
        return (1, person_id)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def summarize_learned_by_split(
    rows: Iterable[Mapping[str, object]],
    splits: Mapping[str, set[str]],
    *,
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    """Aggregate learned metrics without mixing fitted and held-out cohorts."""
    materialized = list(rows)
    expected_people = set().union(*(set(people) for people in splits.values()))
    methods = sorted({str(row["method"]) for row in materialized})
    results: list[dict[str, object]] = []

    rows_by_method: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in materialized:
        method = str(row["method"])
        person_id = str(row["person_id"])
        if person_id in rows_by_method[method]:
            raise ValueError(f"duplicate learned row for {method}, person {person_id}")
        rows_by_method[method][person_id] = row

    cohorts = [("train", set(splits["train"])), ("val", set(splits["val"]))]
    cohorts.extend([("test", set(splits["test"])), ("all", expected_people)])
    for method in methods:
        actual_people = set(rows_by_method[method])
        if actual_people != expected_people:
            missing = expected_people - actual_people
            extra = actual_people - expected_people
            raise ValueError(
                f"{method} person coverage differs from split manifest; "
                f"missing={sorted(missing, key=_person_sort_key)}, "
                f"extra={sorted(extra, key=_person_sort_key)}"
            )
        for split_name, people in cohorts:
            for metric in metrics:
                values = [
                    parsed
                    for person_id in sorted(people, key=_person_sort_key)
                    if (
                        parsed := _finite_float(
                            rows_by_method[method][person_id].get(metric)
                        )
                    )
                    is not None
                ]
                mean = float(np.mean(values)) if values else float("nan")
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                results.append(
                    {
                        "method": method,
                        "split": split_name,
                        "metric": metric,
                        "n_people": len(people),
                        "n_measured": len(values),
                        "mean": mean,
                        "std": std,
                    }
                )
    return results


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite values in [0, 1]")
    if len(values) == 0:
        return []

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * np.arange(len(values), 0, -1))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def paired_comparisons(
    rows: Iterable[Mapping[str, object]],
    *,
    reference_method: str,
    metric: str,
    person_ids: set[str] | None = None,
    seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> list[dict[str, object]]:
    """Compare every method with a reference on identical people."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    by_method: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        person_id = str(row["person_id"])
        if person_ids is not None and person_id not in person_ids:
            continue
        value = _finite_float(row.get(metric))
        if value is not None:
            by_method[str(row["method"])][person_id] = value
    if reference_method not in by_method:
        raise ValueError(f"missing reference method: {reference_method}")

    rng = np.random.default_rng(seed)
    comparisons: list[dict[str, object]] = []
    for method in sorted(set(by_method) - {reference_method}):
        paired_people = sorted(
            set(by_method[method]) & set(by_method[reference_method]),
            key=_person_sort_key,
        )
        if not paired_people:
            continue
        differences = np.asarray(
            [
                by_method[method][person] - by_method[reference_method][person]
                for person in paired_people
            ],
            dtype=float,
        )
        sample_indices = rng.integers(
            0, len(differences), size=(bootstrap_samples, len(differences))
        )
        bootstrap_means = differences[sample_indices].mean(axis=1)
        try:
            p_value = float(wilcoxon(differences).pvalue)
        except ValueError:
            p_value = 1.0
        comparisons.append(
            {
                "method": method,
                "reference_method": reference_method,
                "metric": metric,
                "n_pairs": len(paired_people),
                "mean_difference": float(differences.mean()),
                "ci_low": float(np.percentile(bootstrap_means, 2.5)),
                "ci_high": float(np.percentile(bootstrap_means, 97.5)),
                "wilcoxon_p": p_value,
            }
        )

    adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in comparisons])
    for row, adjusted_p in zip(comparisons, adjusted, strict=True):
        row["holm_p"] = adjusted_p
    return comparisons


def _classification_run(path: Path) -> tuple[str, str]:
    run_dir = next(
        (
            parent
            for parent in path.parents
            if "_[" in parent.name and parent.name.endswith("]")
        ),
        None,
    )
    if run_dir is None:
        raise ValueError(f"cannot identify classification run from {path}")
    separator = run_dir.name.rfind("_[")
    if separator < 1:
        raise ValueError(f"cannot split model and targets in {run_dir.name}")
    model = run_dir.name[:separator]
    target_literal = run_dir.name[separator + 1 :]
    targets = ast.literal_eval(target_literal)
    if not isinstance(targets, list) or not all(
        isinstance(target, str) for target in targets
    ):
        raise ValueError(f"invalid target list in {run_dir.name}")
    return model, ",".join(targets)


def summarize_classification(
    metric_paths: Iterable[Path],
) -> list[dict[str, object]]:
    """Aggregate classification accuracy/F1 over person-level folds."""
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in sorted(Path(item) for item in metric_paths):
        model, targets = _classification_run(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError(f"expected one metric object in {path}")
        metrics = payload[0]
        for metric, raw_value in metrics.items():
            if not (metric.startswith("test/acc_") or metric.startswith("test/f1_")):
                continue
            value = _finite_float(raw_value)
            if value is not None:
                values[(model, targets, metric)].append(value)

    results: list[dict[str, object]] = []
    for (model, targets, metric), fold_values in sorted(values.items()):
        results.append(
            {
                "model": model,
                "targets": targets,
                "metric": metric,
                "n_folds": len(fold_values),
                "mean": float(np.mean(fold_values)),
                "std": (
                    float(np.std(fold_values, ddof=1))
                    if len(fold_values) > 1
                    else 0.0
                ),
            }
        )
    return results


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _fmt_mean_std(mean: float, std: float, *, scale: float = 1.0) -> str:
    return f"{mean * scale:.2f} ± {std * scale:.2f}"


def _build_markdown_summary(
    *,
    splits: Mapping[str, set[str]],
    learned_summary: Sequence[Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
    classification_summary: Sequence[Mapping[str, object]],
    reference_method: str,
) -> str:
    total_people = len(set().union(*splits.values()))
    test_n = len(splits["test"])
    val_n = len(splits["val"])
    lookup = {
        (str(row["method"]), str(row["split"]), str(row["metric"])): row
        for row in learned_summary
    }
    methods = sorted(
        {
            str(row["method"])
            for row in learned_summary
            if str(row["method"]).startswith("A")
            and str(row["method"])[1:].isdigit()
        },
        key=lambda name: (int(name[1:]) if name[1:].isdigit() else 10_000, name),
    )

    lines = [
        "# Project Results Summary",
        "",
        "This report is generated from per-person/fold artefacts. Learned-model "
        "generalization is the held-out test result; the all-person result includes "
        "fitted people and is descriptive only.",
        "",
        "## Evidence cohorts",
        "",
        f"- Primary learned result: held-out test (`N={test_n}`).",
        f"- Secondary learned result: descriptive all-person (`N={total_people}`).",
        f"- Fixed-corruption diagnostic: validation-only (`N={val_n}`).",
        "- Classification variation is the standard deviation across three "
        "person-level folds, not repeated-seed uncertainty.",
        "",
        "## Learned fusion MPJPE",
        "",
        "| Method | held-out test MPJPE (mm) | descriptive all-person MPJPE (mm) |",
        "|---|---:|---:|",
    ]
    for method in methods:
        test_row = lookup[(method, "test", "mpjpe")]
        all_row = lookup[(method, "all", "mpjpe")]
        lines.append(
            f"| {method} | "
            f"{_fmt_mean_std(float(test_row['mean']), float(test_row['std']), scale=1000)} "
            f"(N={test_row['n_measured']}) | "
            f"{_fmt_mean_std(float(all_row['mean']), float(all_row['std']), scale=1000)} "
            f"(N={all_row['n_measured']}) |"
        )

    lines.extend(
        [
            "",
            f"Paired test-set comparisons use `{reference_method}` as the reference. "
            "Positive differences mean the comparison method has higher MPJPE.",
            "",
            "| Method | Δ MPJPE (mm) | paired 95% bootstrap CI (mm) | Holm p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['method']} | {float(row['mean_difference']) * 1000:+.2f} | "
            f"[{float(row['ci_low']) * 1000:+.2f}, "
            f"{float(row['ci_high']) * 1000:+.2f}] | "
            f"{float(row['holm_p']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Validation-only fixed-corruption diagnostic",
            "",
            "| Method | recovery mean ± SD | measured people |",
            "|---|---:|---:|",
        ]
    )
    for method in methods:
        row = lookup[(method, "val", "fixed_corruption_recovery")]
        if int(row["n_measured"]) == 0:
            value = "not measured"
        else:
            value = _fmt_mean_std(float(row["mean"]), float(row["std"]))
        lines.append(f"| {method} | {value} | {row['n_measured']} |")

    full_targets = "posture,relax,twist,total"
    selected_classification = [
        row
        for row in classification_summary
        if row["targets"] == full_targets
        and (
            str(row["metric"]).startswith("test/acc_")
            or str(row["metric"]).startswith("test/f1_")
        )
    ]
    lines.extend(
        [
            "",
            "## Classification (full multitask configuration)",
            "",
            "| Model | Metric | fold mean ± SD | folds |",
            "|---|---|---:|---:|",
        ]
    )
    for row in selected_classification:
        lines.append(
            f"| {row['model']} | {row['metric']} | "
            f"{_fmt_mean_std(float(row['mean']), float(row['std']))} | "
            f"{row['n_folds']} |"
        )
    if not selected_classification:
        lines.append("| — | no complete full-multitask fold metrics found | — | — |")

    lines.extend(
        [
            "",
            "## Evidence still pending",
            "",
            "- Learned fusion inference/evaluation for additional random seeds.",
            "- Offset and perturbation robustness experiments.",
            "- Independent ground truth or public benchmark validation.",
            "",
            "A9 did not complete the nominal 100-epoch schedule; training stopped at "
            "epoch 85 and its best checkpoint was near epoch 83.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_project_results(
    *,
    learned_metrics_path: Path = DEFAULT_LEARNED_METRICS,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    classification_metric_paths: Iterable[Path] | None = None,
    classification_root: Path = DEFAULT_CLASSIFICATION_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reference_method: str = "A6",
    bootstrap_samples: int = 10_000,
) -> dict[str, Path]:
    """Generate cohort-aware CSV and Markdown result artefacts."""
    with learned_metrics_path.open(encoding="utf-8", newline="") as handle:
        learned_rows = list(csv.DictReader(handle))
    if not learned_rows:
        raise ValueError(f"no learned result rows in {learned_metrics_path}")
    splits = load_split_manifest(split_manifest_path)

    available_metrics = [
        metric
        for metric in (
            "mpjpe",
            "median",
            "p95",
            "bone_cv",
            "rigidity",
            "joint_jerk",
            "trunk_angular_jerk",
            "rom_retention",
            "peak_angular_velocity_retention",
            "swap_error",
            "fixed_corruption_recovery",
            "theta_rom",
            "peak_omega",
        )
        if metric in learned_rows[0]
    ]
    required_metrics = {"mpjpe", "fixed_corruption_recovery"}
    if not required_metrics.issubset(available_metrics):
        raise ValueError(
            "learned result CSV must contain mpjpe and fixed_corruption_recovery"
        )
    learned_summary = summarize_learned_by_split(
        learned_rows, splits, metrics=available_metrics
    )
    learned_ablation_rows = [
        row
        for row in learned_rows
        if str(row["method"]).startswith("A")
        and str(row["method"])[1:].isdigit()
    ]
    comparisons = paired_comparisons(
        learned_ablation_rows,
        reference_method=reference_method,
        metric="mpjpe",
        person_ids=set(splits["test"]),
        seed=0,
        bootstrap_samples=bootstrap_samples,
    )

    if classification_metric_paths is None:
        classification_metric_paths = classification_root.glob(
            "**/metrics/fold_*_test_metrics.txt"
        )
    classification_summary = summarize_classification(classification_metric_paths)
    if not classification_summary:
        raise ValueError("no classification accuracy/F1 metrics found")

    outputs = {
        "learned_by_split": output_dir / "learned_results_by_split.csv",
        "learned_test_comparisons": output_dir / "learned_test_comparisons.csv",
        "classification_summary": output_dir / "classification_summary.csv",
        "markdown_summary": output_dir / "RESULTS_SUMMARY.md",
    }
    _write_csv(outputs["learned_by_split"], learned_summary)
    _write_csv(outputs["learned_test_comparisons"], comparisons)
    _write_csv(outputs["classification_summary"], classification_summary)
    _write_text(
        outputs["markdown_summary"],
        _build_markdown_summary(
            splits=splits,
            learned_summary=learned_summary,
            comparisons=comparisons,
            classification_summary=classification_summary,
            reference_method=reference_method,
        ),
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the cohort-aware project results summary."
    )
    parser.add_argument("--learned-metrics", type=Path, default=DEFAULT_LEARNED_METRICS)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--classification-root", type=Path, default=DEFAULT_CLASSIFICATION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-method", default="A6")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args(argv)

    outputs = generate_project_results(
        learned_metrics_path=args.learned_metrics,
        split_manifest_path=args.split_manifest,
        classification_root=args.classification_root,
        output_dir=args.output_dir,
        reference_method=args.reference_method,
        bootstrap_samples=args.bootstrap_samples,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
