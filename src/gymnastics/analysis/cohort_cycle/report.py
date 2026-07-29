"""Publication table and figure rendering contracts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cohorts import sha256_file
from .features import CORE_OUTCOMES


OUTCOME_LABELS = {
    "trunk_axial_rotation_rom": "Axial rotation ROM",
    "angular_speed_p95": "Angular speed (P95)",
    "peak_rotation_phase": "Peak rotation phase",
    "trunk_tilt_p95": "Trunk tilt (P95)",
    "wrist_lead_p95": "Wrist wrapping (P95)",
    "cycle_duration": "Cycle duration",
    "log_dimensionless_angular_jerk": "Log dimensionless jerk",
    "whole_body_repeatability": "Whole-body repeatability",
}


def _repetition_panel_title(
    *,
    label: str,
    interaction_p_holm: float,
) -> str:
    return (
        f"C  {label} repetition trend "
        f"(interaction $p_{{Holm}}={interaction_p_holm:.3f}$)"
    )


def render_report(
    feature_root: str | Path,
    statistics_root: str | Path,
    output_root: str | Path,
) -> dict[str, int]:
    """Render source-matched publication data, LaTeX, and four-panel PDF."""
    features = Path(feature_root)
    statistics = Path(statistics_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"report output already exists: {output}")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"report staging already exists: {staging}")

    analysis_manifest = _load_json(statistics / "analysis_manifest.json")
    if tuple(analysis_manifest.get("core_outcomes", ())) != CORE_OUTCOMES:
        raise ValueError("analysis manifest must contain eight prespecified outcomes")
    core = pd.read_csv(statistics / "core_mixed_models.csv")
    variability = pd.read_csv(statistics / "variability_results.csv")
    sensitivity = pd.read_csv(
        statistics / "sensitivity_mixed_models.csv"
    )
    people = pd.read_csv(features / "person_features.csv")
    if tuple(core["outcome"]) != CORE_OUTCOMES:
        raise ValueError("core model table does not match prespecified order")
    variability = variability.set_index("outcome")
    if set(sensitivity["cycle_reference"]) != {0.5}:
        raise ValueError("sensitivity models must use the mid-repetition reference")
    if set(sensitivity["estimand"]) != {
        "mixed_model_mid_repetition_cohort_effect"
    }:
        raise ValueError("sensitivity models must use the primary estimand")

    rows: list[dict[str, object]] = []
    for _, model in core.iterrows():
        outcome = str(model["outcome"])
        column = f"{outcome}_median"
        if column not in people:
            raise ValueError(f"person table lacks typical outcome: {column}")
        elderly = people.loc[people["cohort"] == "elderly", column].dropna()
        student = people.loc[people["cohort"] == "student", column].dropna()
        variability_row = variability.loc[outcome]
        rows.append(
            {
                "outcome": outcome,
                "label": OUTCOME_LABELS[outcome],
                "elderly_n": len(elderly),
                "elderly_median": float(elderly.median()),
                "elderly_q25": float(elderly.quantile(0.25)),
                "elderly_q75": float(elderly.quantile(0.75)),
                "student_n": len(student),
                "student_median": float(student.median()),
                "student_q25": float(student.quantile(0.25)),
                "student_q75": float(student.quantile(0.75)),
                "cohort_effect": float(model["cohort_effect"]),
                "cohort_effect_standardized": float(
                    model["cohort_effect_standardized"]
                ),
                "cohort_ci_low": float(model["cohort_ci_low"]),
                "cohort_ci_high": float(model["cohort_ci_high"]),
                "cohort_p_holm": float(model["cohort_p_holm"]),
                "variability_effect": float(
                    variability_row["median_difference"]
                ),
                "variability_ci_low": float(
                    variability_row["median_difference_ci_low"]
                ),
                "variability_ci_high": float(
                    variability_row["median_difference_ci_high"]
                ),
                "variability_p_holm": float(variability_row["p_holm"]),
                "cycle_effect": float(model["cycle_effect"]),
                "interaction_effect": float(model["interaction_effect"]),
                "interaction_p_holm": float(
                    model["interaction_p_holm"]
                ),
            }
        )
    report_table = pd.DataFrame(rows)

    staging.mkdir(parents=True)
    try:
        report_table.to_csv(
            staging / "cohort_cycle_core.csv",
            index=False,
            float_format="%.10g",
        )
        (staging / "cohort_cycle_results.tex").write_text(
            _latex_table(report_table),
            encoding="utf-8",
        )
        _render_figure(
            report_table,
            features / "phase_curves.npz",
            statistics / "phase_clusters.csv",
            sensitivity,
            staging / "cohort_cycle_analysis.pdf",
        )
        manifest = {
            "schema_version": 1,
            "core_outcomes": list(CORE_OUTCOMES),
            "inputs": {
                "person_features.csv": sha256_file(
                    features / "person_features.csv"
                ),
                "phase_curves.npz": sha256_file(
                    features / "phase_curves.npz"
                ),
                "core_mixed_models.csv": sha256_file(
                    statistics / "core_mixed_models.csv"
                ),
                "variability_results.csv": sha256_file(
                    statistics / "variability_results.csv"
                ),
                "phase_clusters.csv": sha256_file(
                    statistics / "phase_clusters.csv"
                ),
                "sensitivity_mixed_models.csv": sha256_file(
                    statistics / "sensitivity_mixed_models.csv"
                ),
                "analysis_manifest.json": sha256_file(
                    statistics / "analysis_manifest.json"
                ),
            },
            "outputs": {
                path.name: sha256_file(path)
                for path in sorted(staging.iterdir())
                if path.is_file()
            },
        }
        (staging / "report_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "core_outcomes": len(report_table),
        "figure_panels": 4,
        "panel_c": "source_matched_sensitivity",
    }


def _latex_table(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Out-of-fold cohort and repeated-cycle analysis. Cohort effects are elderly-minus-student mixed-model coefficients at the mid-repetition reference (normalized cycle position 0.5); angular speed, duration, and repeatability use log models. MAD effects are person-level median differences. Values are estimated kinematic descriptors, not clinical joint angles.}",
        r"\label{tab:cohort_cycle_results}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Outcome & Elderly median [IQR] & Student median [IQR] & Cohort effect [95\% CI] & $p_{\mathrm{Holm}}$ & MAD effect ($p_{\mathrm{Holm}}$) & Cohort$\times$cycle ($p_{\mathrm{Holm}}$) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['label']} & "
            f"{row['elderly_median']:.3f} "
            f"[{row['elderly_q25']:.3f}, {row['elderly_q75']:.3f}] & "
            f"{row['student_median']:.3f} "
            f"[{row['student_q25']:.3f}, {row['student_q75']:.3f}] & "
            f"{row['cohort_effect']:.4f} "
            f"[{row['cohort_ci_low']:.4f}, {row['cohort_ci_high']:.4f}] & "
            f"{row['cohort_p_holm']:.4f} & "
            f"{row['variability_effect']:.4f} "
            f"({row['variability_p_holm']:.4f}) & "
            f"{row['interaction_effect']:.4f} "
            f"({row['interaction_p_holm']:.4f}) \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_figure(
    table: pd.DataFrame,
    phase_path: Path,
    cluster_path: Path,
    sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    labels = table["label"].tolist()
    positions = np.arange(len(table))

    axis = axes[0, 0]
    effect = table["cohort_effect_standardized"].to_numpy()
    raw_effect = table["cohort_effect"].to_numpy()
    scale = np.divide(
        effect,
        raw_effect,
        out=np.ones_like(effect),
        where=np.abs(raw_effect) > 1e-12,
    )
    low = effect - table["cohort_ci_low"].to_numpy() * scale
    high = table["cohort_ci_high"].to_numpy() * scale - effect
    axis.errorbar(
        effect,
        positions,
        xerr=np.vstack([low, high]),
        fmt="o",
        color="#315a7d",
        capsize=3,
    )
    axis.axvline(0.0, color="0.5", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title("A  Standardized adjusted cohort effects")
    axis.set_xlabel("Standardized model-scale coefficient")

    axis = axes[0, 1]
    variability = table["variability_effect"].to_numpy()
    variability_low = (
        variability - table["variability_ci_low"].to_numpy()
    )
    variability_high = (
        table["variability_ci_high"].to_numpy() - variability
    )
    axis.errorbar(
        variability,
        positions,
        xerr=np.vstack([variability_low, variability_high]),
        fmt="s",
        color="#a14f3d",
        capsize=3,
    )
    axis.axvline(0.0, color="0.5", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title("B  Within-person MAD differences")
    axis.set_xlabel("Elderly − student median MAD")

    axis = axes[1, 0]
    selected_outcomes = (
        ("angular_speed_p95", "Angular speed", "o", "#315a7d"),
        (
            "log_dimensionless_angular_jerk",
            "Log dimensionless jerk",
            "s",
            "#a14f3d",
        ),
    )
    source_order = [
        source
        for source in ("oof_a6", "face", "side", "deterministic")
        if source in set(sensitivity["source"])
    ]
    source_labels = {
        "oof_a6": "OOF A6",
        "face": "Face",
        "side": "Side",
        "deterministic": "Deterministic",
    }
    source_positions = np.arange(len(source_order), dtype=float)
    for outcome, label, marker, color in selected_outcomes:
        subset = (
            sensitivity.loc[sensitivity["outcome"] == outcome]
            .set_index("source")
            .reindex(source_order)
        )
        estimate = subset["cohort_effect"].to_numpy(dtype=float)
        low = estimate - subset["cohort_ci_low"].to_numpy(dtype=float)
        high = subset["cohort_ci_high"].to_numpy(dtype=float) - estimate
        axis.errorbar(
            estimate,
            source_positions,
            xerr=np.vstack([low, high]),
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )
    axis.axvline(0.0, color="0.5", linewidth=1)
    axis.set_yticks(
        source_positions,
        [source_labels[source] for source in source_order],
    )
    axis.invert_yaxis()
    axis.set_title("C  Source-matched cohort sensitivity")
    axis.set_xlabel("Mid-repetition elderly − student coefficient")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    phase = np.linspace(0.0, 1.0, 101)
    with np.load(phase_path, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        people = np.asarray(archive["person_id"]).astype(str)
        cohorts = np.asarray(archive["cohort"]).astype(str)
    unique_people = np.unique(people)
    person_curves = np.stack(
        [np.median(theta[people == person], axis=0) for person in unique_people]
    )
    person_cohorts = np.asarray(
        [np.unique(cohorts[people == person])[0] for person in unique_people]
    )
    for cohort, color, label in (
        ("student", "#315a7d", "Student"),
        ("elderly", "#a14f3d", "Elderly"),
    ):
        selected = person_curves[person_cohorts == cohort]
        median = np.median(selected, axis=0)
        low_curve, high_curve = np.quantile(selected, [0.25, 0.75], axis=0)
        axis.plot(phase, median, color=color, label=label)
        axis.fill_between(
            phase,
            low_curve,
            high_curve,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    clusters = pd.read_csv(cluster_path)
    if not clusters.empty:
        for _, cluster in clusters.loc[
            (clusters["metric"] == "theta")
            & (clusters["p_holm_across_metrics"] < 0.05)
        ].iterrows():
            axis.axvspan(
                cluster["start_phase"],
                cluster["end_phase"],
                color="#6b6b6b",
                alpha=0.15,
            )
    axis.set_title("D  Phase-normalized axial rotation")
    axis.set_xlabel("Movement phase")
    axis.set_ylabel("Aligned angle (rad)")
    axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain a mapping: {path}")
    return value
