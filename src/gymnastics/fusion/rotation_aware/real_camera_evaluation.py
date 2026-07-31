"""Pseudo-GT-isolated evaluation for the collected-data camera pilot."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_skeleton_spec
from .evaluation import (
    MethodSequence,
    evaluate_person_trials,
    load_triangulated_references,
)
from .real_camera_training import RealCameraRun


@dataclass(frozen=True)
class CameraEvaluationSummary:
    method_rows: list[dict[str, Any]]
    paired_rows: list[dict[str, Any]]
    camera_claim_supported: bool


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(
        float(value)
    )


def _clustered_interval(
    records: Sequence[tuple[str, float]],
    *,
    samples: int,
    seed: int = 20260730,
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = {}
    for person_id, value in records:
        grouped.setdefault(str(person_id), []).append(float(value))
    people = sorted(grouped)
    if not people:
        return float("nan"), float("nan")
    person_means = np.asarray(
        [np.mean(grouped[person]) for person in people], dtype=np.float64
    )
    generator = np.random.default_rng(int(seed))
    draws = generator.choice(
        person_means, size=(int(samples), len(person_means)), replace=True
    ).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def aggregate_camera_metrics(
    person_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
) -> CameraEvaluationSummary:
    """Aggregate already person-pooled metrics and make paired G comparisons."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    indexed: dict[tuple[str, int, str], float] = {}
    for row in person_rows:
        key = (
            str(row["person_id"]),
            int(row["seed"]),
            str(row["method"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate person/seed/method metric: {key}")
        if _finite(row.get("mpjpe")):
            indexed[key] = float(row["mpjpe"])
    if not indexed:
        raise ValueError("No finite person MPJPE values are available")

    methods = sorted({key[2] for key in indexed})
    method_rows: list[dict[str, Any]] = []
    for method in methods:
        values = [
            value for (person, seed, name), value in indexed.items() if name == method
        ]
        seed_means = [
            np.mean(
                [
                    value
                    for (person, row_seed, name), value in indexed.items()
                    if name == method and row_seed == seed
                ]
            )
            for seed in sorted(
                {key[1] for key in indexed if key[2] == method}
            )
        ]
        method_rows.append(
            {
                "method": method,
                "person_seed_pairs": len(values),
                "mean_person_mpjpe": float(np.mean(values)),
                "median_person_mpjpe": float(np.median(values)),
                "seed_mean_std": (
                    float(np.std(seed_means, ddof=1))
                    if len(seed_means) > 1
                    else 0.0
                ),
            }
        )

    paired_rows: list[dict[str, Any]] = []

    def compare(method: str, baseline: str) -> dict[str, Any]:
        shared = sorted(
            (person, seed)
            for person, seed, name in indexed
            if name == method and (person, seed, baseline) in indexed
        )
        differences = [
            indexed[(person, seed, method)] - indexed[(person, seed, baseline)]
            for person, seed in shared
        ]
        clustered = [
            (person, difference)
            for (person, _), difference in zip(shared, differences)
        ]
        low, high = _clustered_interval(
            clustered, samples=bootstrap_samples
        )
        by_person: dict[str, list[float]] = {}
        for person, value in clustered:
            by_person.setdefault(person, []).append(value)
        return {
            "method": method,
            "baseline": baseline,
            "paired_samples": len(shared),
            "paired_people": len(by_person),
            "mean_delta_mpjpe": (
                float(np.mean(differences)) if differences else float("nan")
            ),
            "median_delta_mpjpe": (
                float(np.median(differences)) if differences else float("nan")
            ),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "improved_people": sum(
                float(np.mean(values)) < 0.0 for values in by_person.values()
            ),
        }

    for method in ("G1", "G2", "G3", "G4", "G5"):
        if method in methods and "G0" in methods:
            paired_rows.append(compare(method, "G0"))
    if "G4" in methods and "G5" in methods:
        paired_rows.append(compare("G4", "G5"))

    comparisons = {
        (row["method"], row["baseline"]): row for row in paired_rows
    }
    g4_g0 = comparisons.get(("G4", "G0"))
    g4_g5 = comparisons.get(("G4", "G5"))
    supported = bool(
        g4_g0
        and g4_g5
        and _finite(g4_g0["mean_delta_mpjpe"])
        and _finite(g4_g5["mean_delta_mpjpe"])
        and float(g4_g0["mean_delta_mpjpe"]) < 0.0
        and float(g4_g5["mean_delta_mpjpe"]) < 0.0
    )
    return CameraEvaluationSummary(method_rows, paired_rows, supported)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _external_metrics_in_millimeters(row: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    for name in ("mpjpe", "median", "p95"):
        if _finite(converted.get(name)):
            converted[name] = 1000.0 * float(converted[name])
    return converted


def write_real_camera_report(
    summary: CameraEvaluationSummary,
    path: str | Path,
    *,
    camera_audit: Mapping[str, Any],
) -> Path:
    """Write a compact report whose conclusion is gated by the G5 control."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ranking = sorted(
        summary.method_rows, key=lambda row: float(row["mean_person_mpjpe"])
    )
    conclusion = (
        "The fitted-camera claim is supported in this pilot: G4 improves on "
        "both G0 and the wrong-camera G5 control."
        if summary.camera_claim_supported
        else "The fitted-camera claim is not supported: G4 does not improve "
        "on both G0 and the wrong-camera G5 control."
    )
    lines = [
        "# Real-data fitted-camera pilot",
        "",
        conclusion,
        "",
        "## Ranking",
        "",
        "| Rank | Method | Mean person MPJPE (mm) | Median (mm) | Seed SD (mm) |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | {row['method']} | "
            f"{float(row['mean_person_mpjpe']):.4f} | "
            f"{float(row['median_person_mpjpe']):.4f} | "
            f"{float(row['seed_mean_std']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Method | Baseline | Mean delta | 95% clustered bootstrap CI | Improved people |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary.paired_rows:
        lines.append(
            f"| {row['method']} | {row['baseline']} | "
            f"{float(row['mean_delta_mpjpe']):.4f} | "
            f"[{float(row['bootstrap_ci_low']):.4f}, "
            f"{float(row['bootstrap_ci_high']):.4f}] | "
            f"{int(row['improved_people'])}/{int(row['paired_people'])} |"
        )
    lines.extend(
        [
            "",
            "## Camera audit and interpretation",
            "",
            f"- Camera-audit people: {camera_audit.get('people', 'unknown')}.",
            "- Median holdout reprojection error: "
            f"{camera_audit.get('median_holdout_reprojection_px', 'unknown')} px.",
            "- A single fixed-rig fit failed before training "
            f"(holdout: {camera_audit.get('fixed_rig_holdout_reprojection_px', 'unknown')} px); "
            "the pilot therefore uses declared per-person transductive input fitting.",
            "- Triangulated 3D is loaded only by evaluation and is not a training input.",
            "- Because camera fitting and pseudo-GT originate from the same SAM3D observations, "
            "the result is descriptive rather than a fully independent causal validation.",
            "",
        ]
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(target)
    return target


def _load_method_sequence(path: Path, method: str) -> MethodSequence:
    with np.load(path, allow_pickle=False) as data:
        return MethodSequence(
            method=method,
            kpts_world=np.asarray(data["kpts_world"], dtype=np.float32),
            timestamps=np.asarray(data["timestamps"], dtype=np.float64),
            frame_valid=np.asarray(data["frame_valid"], dtype=bool),
            joint_valid=np.asarray(data["joint_valid"], dtype=bool),
            trial_id=path.parent.name,
            face_map=np.asarray(data["face_map"], dtype=np.int32),
            side_map=np.asarray(data["side_map"], dtype=np.int32),
        )


def _verified_provenance(run: RealCameraRun) -> Mapping[str, Any]:
    with run.provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if (
        provenance.get("triangulated_3d_available_to_training") is not False
        or provenance.get("test_people_available_to_training") is not False
    ):
        raise ValueError(f"Run does not prove evaluation isolation: {run.run_root}")
    return provenance


def evaluate_real_camera_runs(
    runs: Sequence[RealCameraRun],
    *,
    triangulated_root: str | Path,
    skeleton_path: str | Path,
    output_root: str | Path,
    camera_audit: Mapping[str, Any],
    bootstrap_samples: int = 10_000,
) -> CameraEvaluationSummary:
    """Evaluate complete inference artifacts, loading pseudo-GT only here."""

    skeleton = load_skeleton_spec(Path(skeleton_path))
    cycle_rows: list[dict[str, Any]] = []
    person_rows: list[dict[str, Any]] = []
    for run in runs:
        provenance = _verified_provenance(run)
        paths = sorted((run.run_root / "inference").glob("person_*/*/fused_sequence.npz"))
        if not paths:
            raise FileNotFoundError(f"No inference outputs for {run.run_root}")
        grouped: dict[str, list[MethodSequence]] = {}
        for path in paths:
            person_id = path.parent.parent.name.removeprefix("person_")
            if person_id in set(provenance.get("train_people", ())) | set(
                provenance.get("validation_people", ())
            ):
                raise ValueError(f"Evaluation person leaked into training: {person_id}")
            grouped.setdefault(person_id, []).append(
                _load_method_sequence(path, run.ablation)
            )
        for person_id, sequences in grouped.items():
            references = load_triangulated_references(
                triangulated_root, person_id, sequences
            )
            report = evaluate_person_trials(
                person_id,
                sequences,
                skeleton,
                references=references,
                alignment="similarity",
            )
            for row in report.person_metrics:
                person_rows.append(
                    {
                        **_external_metrics_in_millimeters(row),
                        "seed": run.seed,
                        "method": run.ablation,
                    }
                )
            for sequence in sequences:
                cycle_report = evaluate_person_trials(
                    person_id,
                    [sequence],
                    skeleton,
                    references=references,
                    alignment="similarity",
                )
                cycle_rows.append(
                    {
                        **_external_metrics_in_millimeters(
                            cycle_report.person_metrics[0]
                        ),
                        "trial_id": sequence.trial_id,
                        "seed": run.seed,
                        "method": run.ablation,
                    }
                )

    summary = aggregate_camera_metrics(
        person_rows, bootstrap_samples=bootstrap_samples
    )
    output_root = Path(output_root)
    _write_csv(output_root / "metrics_by_cycle.csv", cycle_rows)
    _write_csv(output_root / "metrics_by_person.csv", person_rows)
    _write_csv(output_root / "metrics_by_method.csv", summary.method_rows)
    _write_csv(output_root / "paired_comparisons.csv", summary.paired_rows)
    write_real_camera_report(
        summary,
        output_root / "real_camera_feature_report.md",
        camera_audit=camera_audit,
    )
    return summary
