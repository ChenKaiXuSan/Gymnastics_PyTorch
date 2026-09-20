"""Write machine-readable and human-readable Unity benchmark reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .evaluation import EvaluationBundle


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_plain(rows))


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        return ["_No results._"]
    columns = (
        "method",
        "mpjpe_mm",
        "median_mm",
        "p95_mm",
        "angle_mae_deg",
        "valid_points",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _best_row(
    rows: Sequence[Mapping[str, object]],
    predicate,
) -> Mapping[str, object] | None:
    candidates = [row for row in rows if predicate(str(row["method"]))]
    return (
        min(candidates, key=lambda row: float(row["mpjpe_mm"]))
        if candidates
        else None
    )


def _selected_rows(
    bundle: EvaluationBundle,
) -> tuple[Mapping[str, object], ...]:
    rows = bundle.valid_ranking
    choices = (
        rows[0] if rows else None,
        _best_row(rows, lambda method: method in {"cam0", "cam1"}),
        _best_row(
            rows,
            lambda method: method not in {"cam0", "cam1"}
            and not method.startswith("triangulation_"),
        ),
        _best_row(rows, lambda method: method.startswith("A")),
    )
    selected: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for row in choices:
        if row is not None and str(row["method"]) not in seen:
            selected.append(row)
            seen.add(str(row["method"]))
    return tuple(selected)


def _executive_conclusions(bundle: EvaluationBundle) -> list[str]:
    ranking = bundle.valid_ranking
    if not ranking:
        return ["- No valid methods were evaluated."]
    best = ranking[0]
    lines = [
        f"- Best valid method: `{best['method']}` at "
        f"{float(best['mpjpe_mm']):.3f} mm MPJPE and "
        f"{float(best['angle_mae_deg']):.3f}° angle MAE."
    ]
    best_single = _best_row(
        ranking, lambda method: method in {"cam0", "cam1"}
    )
    best_fusion = _best_row(
        ranking,
        lambda method: method not in {"cam0", "cam1"}
        and not method.startswith("triangulation_"),
    )
    best_triangulation = _best_row(
        ranking, lambda method: method.startswith("triangulation_")
    )
    best_learned = _best_row(ranking, lambda method: method.startswith("A"))

    def reduction(candidate, baseline) -> float:
        return (
            float(baseline["mpjpe_mm"]) - float(candidate["mpjpe_mm"])
        ) / float(baseline["mpjpe_mm"]) * 100.0

    if best_triangulation is not None and best_fusion is not None:
        lines.append(
            f"- `{best_triangulation['method']}` reduces MPJPE by "
            f"{reduction(best_triangulation, best_fusion):.2f}% relative to "
            f"the best direct-3D fusion (`{best_fusion['method']}`)."
        )
    if best_single is not None and best_fusion is not None:
        change = reduction(best_fusion, best_single)
        direction = "reduces" if change >= 0 else "increases"
        lines.append(
            f"- Best direct-3D fusion: `{best_fusion['method']}` at "
            f"{float(best_fusion['mpjpe_mm']):.3f} mm; it {direction} MPJPE "
            f"by {abs(change):.2f}% versus the best single view "
            f"(`{best_single['method']}`)."
        )
    if best_single is not None and best_learned is not None:
        change = reduction(best_learned, best_single)
        direction = "improves on" if change >= 0 else "underperforms"
        lines.append(
            f"- Best zero-shot learned model: `{best_learned['method']}` at "
            f"{float(best_learned['mpjpe_mm']):.3f} mm; it {direction} the "
            f"best single view by {abs(change):.2f}%."
        )
    return lines


def _per_sequence_table(bundle: EvaluationBundle) -> list[str]:
    methods = {str(row["method"]) for row in _selected_rows(bundle)}
    rows = [
        row
        for row in bundle.tables["by_sequence"]
        if str(row["method"]) in methods
    ]
    if not rows:
        return ["_No selected per-sequence results._"]
    columns = (
        "method",
        "sequence_id",
        "eval_frames",
        "mpjpe_mm",
        "angle_mae_deg",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = [
            f"{row[column]:.3f}"
            if isinstance(row.get(column), float)
            else str(row.get(column, ""))
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _visibility_table(bundle: EvaluationBundle) -> list[str]:
    selected = _selected_rows(bundle)
    methods = {str(row["method"]) for row in selected[:3]}
    partitions = {"both_visible", "one_visible", "neither_visible"}
    accumulators: dict[tuple[str, str], list[float]] = {}
    for row in bundle.tables["by_visibility"]:
        method = str(row["method"])
        visibility = str(row["visibility"])
        if method not in methods or visibility not in partitions:
            continue
        key = (method, visibility)
        count = int(row["valid_points"])
        total, points = accumulators.get(key, [0.0, 0.0])
        accumulators[key] = [
            total + float(row["mpjpe_mm"]) * count,
            points + count,
        ]
    if not accumulators:
        return ["_No selected visibility results._"]
    lines = [
        "| method | visibility | valid_points | mpjpe_mm |",
        "|---|---|---|---|",
    ]
    for (method, visibility), (total, points) in accumulators.items():
        mean = total / points if points else float("nan")
        lines.append(
            f"| {method} | {visibility} | {int(points)} | {mean:.3f} |"
        )
    return lines


def write_report(
    bundle: EvaluationBundle,
    output_root: Path,
    *,
    provenance: Mapping[str, object],
) -> Path:
    output_root = Path(output_root)
    evaluation_root = output_root / "evaluation"
    report_root = output_root / "report"
    figures_root = report_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    _write_csv(evaluation_root / "metrics_summary.csv", bundle.tables["summary"])
    _write_csv(
        evaluation_root / "metrics_by_sequence.csv", bundle.tables["by_sequence"]
    )
    _write_csv(evaluation_root / "metrics_by_joint.csv", bundle.tables["by_joint"])
    _write_csv(
        evaluation_root / "metrics_by_visibility.csv",
        bundle.tables["by_visibility"],
    )
    error_payload = {
        f"{result.method}__{result.sequence_id}": result.errors_m
        for result in bundle.results
    }
    np.savez_compressed(evaluation_root / "per_frame_errors.npz", **error_payload)
    machine = {
        "valid_ranking": _plain(bundle.valid_ranking),
        "diagnostics": _plain(bundle.diagnostics),
        "failures": _plain(bundle.failures),
        "tables": _plain(bundle.tables),
        "provenance": {**_plain(bundle.provenance), **_plain(provenance)},
    }
    (report_root / "results.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt

        rows = bundle.valid_ranking
        if rows:
            figure, axis = plt.subplots(figsize=(10, 5))
            axis.bar(
                [str(row["method"]) for row in rows],
                [float(row["mpjpe_mm"]) for row in rows],
            )
            axis.set_ylabel("MPJPE (mm)")
            axis.tick_params(axis="x", rotation=55)
            figure.tight_layout()
            figure.savefig(figures_root / "valid_method_mpjpe.png", dpi=160)
            plt.close(figure)
    except ImportError:
        pass

    lines = [
        "# Unity Benchmark Report",
        "",
        "Unity native 3D keypoints are the evaluation ground truth. "
        "Every sequence uses one shared Sim3 alignment.",
        "",
        "## Executive Conclusions",
        "",
        *_executive_conclusions(bundle),
        "",
        "## Valid Method Ranking",
        "",
        *_markdown_table(bundle.valid_ranking),
        "",
        "## Selected Per-Sequence Results",
        "",
        *_per_sequence_table(bundle),
        "",
        "## Visibility Breakdown",
        "",
        "The selected methods below are aggregated over all sequences. "
        "Visibility labels come from Unity and are used only for evaluation.",
        "",
        *_visibility_table(bundle),
        "",
        "## Diagnostic Methods",
        "",
        "These rows are excluded from the valid ranking because they are oracle "
        "or use Unity ground truth during method construction.",
        "",
        *_markdown_table(bundle.diagnostics),
        "",
        "## Completion and Failures",
        "",
        f"- Explicit failures: {len(bundle.failures)}",
        f"- Evaluated method-sequences: {len(bundle.results)}",
        "",
        "## Interpretation Boundaries",
        "",
        "This dataset contains one avatar, one environment, five static samples, "
        "and two continuous sequences. Results describe this external synthetic "
        "benchmark and do not establish population-level generalization or "
        "statistical significance.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps({**_plain(bundle.provenance), **_plain(provenance)}, indent=2),
        "```",
        "",
    ]
    report_path = report_root / "unity_benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
