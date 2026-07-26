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
        "## Valid Method Ranking",
        "",
        *_markdown_table(bundle.valid_ranking),
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
