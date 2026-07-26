"""Deterministic machine-readable and Markdown FreeMan benchmark reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import EvaluationTables
from .mapping import FREEMAN_COCO17_NAMES


@dataclass(frozen=True)
class ReportContext:
    """Provenance required to make a benchmark report auditable."""

    resolved_config: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    download_manifest: Mapping[str, Any]
    camera_pairs: pd.DataFrame
    checkpoint_metadata: Mapping[str, Any]
    code_commit: str


@dataclass(frozen=True)
class ReportOutputs:
    markdown: Path
    results_json: Path
    csv_paths: Mapping[str, Path]


_TABLE_FILES = {
    "metrics_by_session": "metrics_by_session.csv",
    "metrics_by_subject": "metrics_by_subject.csv",
    "metrics_by_method": "metrics_by_method.csv",
    "metrics_by_joint": "metrics_by_joint.csv",
    "metrics_by_split": "metrics_by_split.csv",
    "metrics_by_scenario": "metrics_by_scenario.csv",
    "paired_statistics": "paired_statistics.csv",
    "failures": "failures.csv",
}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "subject_id",
            "fps",
            "split",
            "scenario",
            "action",
            "session_id",
            "method",
            "classification",
            "joint",
            "baseline",
        )
        if column in frame.columns
    ]
    if not columns or frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(columns, na_position="last").reset_index(drop=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_ready(row) for row in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is pd.NA:
        return None
    return value


def _classification(by_method: pd.DataFrame) -> dict[str, str]:
    if by_method.empty:
        return {}
    result: dict[str, str] = {}
    for row in by_method[["method", "classification"]].drop_duplicates().itertuples(
        index=False
    ):
        method = str(row.method)
        classification = str(row.classification)
        previous = result.get(method)
        if previous is not None and previous != classification:
            raise ValueError(f"method has conflicting classifications: {method}")
        result[method] = classification
    return dict(sorted(result.items()))


def _coverage(
    context: ReportContext,
    by_subject: pd.DataFrame,
) -> tuple[int, int, float, float, bool]:
    dataset_config = context.resolved_config.get("dataset", {})
    evaluation_config = context.resolved_config.get("evaluation", {})
    expected = {
        int(subject) for subject in dataset_config.get("subjects", range(1, 41))
    }
    processed = {
        int(subject)
        for subject in context.dataset_manifest.get("processed_subjects", ())
    }
    evaluated = {
        int(subject)
        for subject in by_subject.loc[
            by_subject["classification"] == "VALID", "subject_id"
        ]
    }
    covered = len(expected & processed & evaluated)
    total = len(expected)
    fraction = covered / total if total else 0.0
    minimum = float(evaluation_config.get("minimum_subject_coverage", 1.0))
    complete = total > 0 and fraction >= minimum
    return covered, total, fraction, minimum, complete


def _subject_balanced_group(
    by_session: pd.DataFrame,
    groups: Sequence[str],
) -> pd.DataFrame:
    if by_session.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in by_session.columns
        if column.endswith("_mm")
        or column.endswith("_mm_s")
        or column.endswith("_mm_s2")
        or column.startswith("pck_")
        or column in {"auc", "coverage"}
    ]
    subject_groups = [*groups, "subject_id", "method", "classification"]
    subject = (
        by_session.groupby(subject_groups, dropna=False, as_index=False)[
            metric_columns
        ]
        .mean(numeric_only=True)
    )
    return (
        subject.groupby(
            [*groups, "method", "classification"],
            dropna=False,
            as_index=False,
        )[metric_columns]
        .mean(numeric_only=True)
        .sort_values([*groups, "method", "classification"])
        .reset_index(drop=True)
    )


def _compact_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected:
        return "_No rows available._"
    display = frame[selected].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else ""
        )
    header = "| " + " | ".join(selected) + " |"
    divider = "| " + " | ".join("---" for _ in selected) + " |"
    rows = [
        "| "
        + " | ".join(str(value).replace("|", "\\|") for value in row)
        + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _markdown(
    tables: EvaluationTables,
    context: ReportContext,
    *,
    complete: bool,
    covered_subjects: int,
    expected_subjects: int,
    coverage_fraction: float,
    minimum_coverage: float,
) -> str:
    classifications = _classification(tables.by_method)
    valid = tables.by_method[
        tables.by_method["classification"] == "VALID"
    ].sort_values("sim3_mpjpe_mm")
    diagnostic = tables.by_method[
        tables.by_method["classification"] != "VALID"
    ].sort_values(["classification", "sim3_mpjpe_mm"])
    by_fps = _subject_balanced_group(tables.by_session, ["fps"])
    action_rows = tables.by_session[tables.by_session["action"].notna()]
    by_action = (
        _subject_balanced_group(action_rows, ["action"])
        if not action_rows.empty
        else pd.DataFrame()
    )
    pairs = context.camera_pairs
    pair_error = (
        float(pairs["target_error_deg"].mean())
        if "target_error_deg" in pairs and not pairs.empty
        else float("nan")
    )
    coverage_phrase = (
        "The benchmark processed all 40 subjects requested by the protocol."
        if complete and expected_subjects == 40 and covered_subjects == 40
        else (
            f"The benchmark processed {covered_subjects} of "
            f"{expected_subjects} requested subjects."
        )
    )
    failure_count = len(tables.failures)
    repo = context.resolved_config.get("repository", {})
    fps_counts = context.dataset_manifest.get("fps_session_counts", {})
    reference_scale = context.resolved_config.get("dataset", {}).get(
        "reference_scale_to_m"
    )
    lines = [
        "# FreeMan Zero-Shot Multi-View Benchmark",
        "",
        f"Complete: {'yes' if complete else 'no'}",
        "",
        "## Protocol",
        "",
        (
            "Existing gymnastics-trained SAM3D/fusion checkpoints are evaluated "
            "zero-shot. No FreeMan reference pose is used for training, checkpoint "
            "selection, fusion weights, or valid-method inference."
        ),
        "",
        (
            "FreeMan is used as a public markerless multi-view reference produced "
            "from synchronized cameras and multi-view reconstruction. It is not "
            "independent marker-based motion capture, so the numbers measure "
            "agreement with that markerless reference rather than absolute mocap "
            "accuracy."
        ),
        "",
        f"Repository: `{repo.get('repo_id', 'unknown')}` at `{repo.get('revision', 'unknown')}`.",
        f"Code commit: `{context.code_commit}`.",
        f"Configured reference scale to metres: `{reference_scale}`.",
        "",
        "## Coverage",
        "",
        coverage_phrase,
        (
            f" Subject coverage is {coverage_fraction:.1%}; the configured "
            f"completion threshold is {minimum_coverage:.1%}."
        ),
        f"Processed sessions: {context.dataset_manifest.get('processed_sessions', 0)}.",
        f"FPS session counts: `{json.dumps(_json_ready(fps_counts), sort_keys=True)}`.",
        f"Recorded failures/exclusions: {failure_count}.",
        "",
        "## Camera pairs",
        "",
        (
            f"Selected pairs: {len(pairs)}; mean absolute target-angle error: "
            f"{pair_error:.3f} degrees."
            if math.isfinite(pair_error)
            else f"Selected pairs: {len(pairs)}; target-angle error unavailable."
        ),
        "",
        "## Valid subject-balanced ranking",
        "",
        _compact_table(
            valid,
            (
                "method",
                "sim3_mpjpe_mm",
                "median_mpjpe_mm",
                "p95_mpjpe_mm",
                "root_mpjpe_mm",
                "pa_mpjpe_mm",
                "velocity_error_mm_s",
                "acceleration_error_mm_s2",
                "coverage",
            ),
        ),
        "",
        "## Oracle/leaky diagnostics",
        "",
        (
            "The following methods are excluded from valid ranking because their "
            "metadata classifies them as oracle, GT-leaky, or otherwise diagnostic."
        ),
        "",
        _compact_table(
            diagnostic,
            ("method", "classification", "sim3_mpjpe_mm", "coverage"),
        ),
        "",
        "## Results by FPS",
        "",
        _compact_table(
            by_fps,
            ("fps", "method", "classification", "sim3_mpjpe_mm", "coverage"),
        ),
        "",
        "## Results by official split",
        "",
        _compact_table(
            tables.by_split,
            ("split", "method", "classification", "sim3_mpjpe_mm", "coverage"),
        ),
        "",
        "## Results by scenario",
        "",
        _compact_table(
            tables.by_scenario,
            (
                "scenario",
                "method",
                "classification",
                "sim3_mpjpe_mm",
                "coverage",
            ),
        ),
        "",
        "## Results by action",
        "",
        _compact_table(
            by_action,
            ("action", "method", "classification", "sim3_mpjpe_mm", "coverage"),
        ),
        "",
        "## Paired subject statistics",
        "",
        _compact_table(
            tables.paired_statistics,
            (
                "method",
                "baseline",
                "matched_subjects",
                "mean_difference_mm",
                "median_difference_mm",
                "ci95_low_mm",
                "ci95_high_mm",
                "p_value",
                "holm_p_value",
                "status",
            ),
        ),
        "",
        "## Failures and exclusions",
        "",
        _compact_table(
            tables.failures,
            ("subject_id", "session_id", "stage", "reason"),
        ),
        "",
        "## Reproduction",
        "",
        "```bash",
        "conda run -n gymnastic gymnastics benchmark freeman inspect",
        "conda run -n gymnastic gymnastics benchmark freeman run",
        "conda run -n gymnastic gymnastics benchmark freeman report",
        "```",
        "",
        "Machine-readable artifacts are under `evaluation/`; the selected camera "
        "pairs and this report are under `report/`.",
        "",
    ]
    if not diagnostic.empty:
        method_names = ", ".join(f"`{name}`" for name in diagnostic["method"])
        lines.insert(
            lines.index("## Results by FPS"),
            f"Diagnostic methods written separately: {method_names}; each is excluded from valid ranking.",
        )
        lines.insert(lines.index("## Results by FPS"), "")
    if classifications.get("sim3_face_stable_joint_weight") is not None:
        # Keep the protocol-sensitive method visible even if a compact table is
        # later filtered by a downstream document renderer.
        lines.append(
            "`sim3_face_stable_joint_weight` remains an explicitly separated diagnostic."
        )
        lines.append("")
    return "\n".join(lines)


def write_report(
    tables: EvaluationTables,
    context: ReportContext,
    output_root: Path,
) -> ReportOutputs:
    """Write stable CSV, JSON, and Markdown artifacts for one benchmark run."""
    root = Path(output_root)
    evaluation_root = root / "evaluation"
    report_root = root / "report"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    table_values = {
        "metrics_by_session": tables.by_session,
        "metrics_by_subject": tables.by_subject,
        "metrics_by_method": tables.by_method,
        "metrics_by_joint": tables.by_joint,
        "metrics_by_split": tables.by_split,
        "metrics_by_scenario": tables.by_scenario,
        "paired_statistics": tables.paired_statistics,
        "failures": tables.failures,
    }
    csv_paths: dict[str, Path] = {}
    for name, filename in _TABLE_FILES.items():
        path = evaluation_root / filename
        frame = _sorted(table_values[name])
        _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))
        csv_paths[name] = path

    pairs_path = report_root / "camera_pairs.csv"
    _atomic_text(
        pairs_path,
        _sorted(context.camera_pairs).to_csv(index=False, lineterminator="\n"),
    )
    csv_paths["camera_pairs"] = pairs_path

    covered, expected, fraction, minimum, complete = _coverage(
        context,
        tables.by_subject,
    )
    method_classification = _classification(tables.by_method)
    headline = _sorted(
        tables.by_method[tables.by_method["classification"] == "VALID"]
    )
    result_payload = {
        "repository": context.resolved_config.get("repository", {}),
        "archive_inventory_sha256": context.download_manifest.get(
            "inventory_sha256"
        ),
        "resolved_config": context.resolved_config,
        "camera_pairs": _sorted(context.camera_pairs),
        "mapping": {
            "version": "freeman_coco17_to_mhr70_v1",
            "joint_names": FREEMAN_COCO17_NAMES,
        },
        "checkpoint_metadata": context.checkpoint_metadata,
        "method_classification": method_classification,
        "coverage": {
            "processed_subjects": covered,
            "expected_subjects": expected,
            "fraction": fraction,
            "minimum_required": minimum,
            "complete": complete,
            "processed_sessions": context.dataset_manifest.get(
                "processed_sessions", 0
            ),
            "failure_count": len(tables.failures),
        },
        "subject_balanced_headline": headline,
        "code_commit": context.code_commit,
    }
    results_path = evaluation_root / "results.json"
    _atomic_text(
        results_path,
        json.dumps(
            _json_ready(result_payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )

    markdown_path = report_root / "freeman_benchmark_report.md"
    _atomic_text(
        markdown_path,
        _markdown(
            tables,
            context,
            complete=complete,
            covered_subjects=covered,
            expected_subjects=expected,
            coverage_fraction=fraction,
            minimum_coverage=minimum,
        ),
    )
    return ReportOutputs(
        markdown=markdown_path,
        results_json=results_path,
        csv_paths=csv_paths,
    )
