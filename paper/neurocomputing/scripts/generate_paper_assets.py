#!/usr/bin/env python3
"""Generate source-checked paper tables and plots from fuse metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
WORKTREE_ROOT = PAPER_ROOT.parent.parent
EXPECTED_METHODS = 9
EXPECTED_PEOPLE = 137
EXPECTED_ROWS = EXPECTED_METHODS * EXPECTED_PEOPLE
LEAKY_METHODS = frozenset({"sim3_face_stable_joint_weight"})
EXPECTED_OVERALL_BEST = "sim3_face_stable_joint_weight"
EXPECTED_OVERALL_BEST_MEAN = 0.063481653
EXPECTED_LEAKAGE_FREE_BEST = "avg_body_current"
EXPECTED_LEAKAGE_FREE_BEST_MEAN = 0.064045285

DISPLAY_NAMES = {
    "avg_body_current": "Body-frame average",
    "avg_world_face_ref": "World-coordinate average",
    "root_face_stable": "Root alignment + average",
    "sim3_face_all": "Sim3 (all joints) + average",
    "sim3_face_stable": "Sim3 (stable joints) + average",
    "sim3_face_stable_joint_weight": "Sim3 + pseudo-reference-fitted joint weights",
    "sim3_face_stable_bodypart_weight": "Sim3 + body-part weights",
    "sim3_face_stable_smooth_transform": "Sim3 + side smoothing + average",
    "sim3_face_stable_smooth_kpt": "Sim3 + average + output smoothing",
}

LEARNED_NAMES = {
    "A0": "Face only",
    "A1": "Side only",
    "A2": "Arithmetic mean",
    "A3": "Quality mean",
    "A4": "Spatial objectives",
    "A5": "+ rotation/temporal",
    "A6": "+ complete-cycle ROM (mainline)",
    "A7": "+ per-view-peak ROM",
    "A8": "+ twist residual",
    "A9": "+ twist-rate anchor",
}


def project_root() -> Path:
    """Resolve the checkout that owns immutable experiment artifacts."""
    explicit = os.environ.get("GYMNASTICS_PROJECT_ROOT")
    if explicit:
        return Path(explicit)
    if WORKTREE_ROOT.parent.name == ".worktrees":
        return WORKTREE_ROOT.parent.parent
    return WORKTREE_ROOT


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"evidence CSV is empty: {path}")
    return rows


def load_learned_evidence(
    result_path: Path | None = None,
    comparison_path: Path | None = None,
) -> dict[str, object]:
    """Load and validate the frozen 14-person learned test evidence."""
    root = project_root()
    evidence_root = root / "local/runs/analysis/project_results"
    result_path = result_path or evidence_root / "learned_results_by_split.csv"
    comparison_path = (
        comparison_path
        or evidence_root / "learned_test_comparisons.csv"
    )
    rows = _read_csv_rows(result_path)
    test_rows = [
        row
        for row in rows
        if row["split"] == "test"
        and row["method"] in LEARNED_NAMES
        and row["metric"]
        in {"mpjpe", "rom_retention", "peak_angular_velocity_retention"}
    ]
    indexed = {
        (row["method"], row["metric"]): row
        for row in test_rows
    }
    expected_keys = {
        (method, metric)
        for method in LEARNED_NAMES
        for metric in (
            "mpjpe",
            "rom_retention",
            "peak_angular_velocity_retention",
        )
    }
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise ValueError(
            f"learned test evidence mismatch; missing={missing}, extra={extra}"
        )
    if {
        int(row["n_people"])
        for row in indexed.values()
    } != {14}:
        raise ValueError("primary learned evidence must use the 14-person test set")
    if {
        int(row["n_measured"])
        for row in indexed.values()
    } != {14}:
        raise ValueError("learned test evidence must measure all 14 people")

    comparisons = _read_csv_rows(comparison_path)
    comparison_index = {
        row["method"]: row
        for row in comparisons
        if row["reference_method"] == "A6" and row["metric"] == "mpjpe"
    }
    if set(comparison_index) != set(LEARNED_NAMES) - {"A6"}:
        raise ValueError("paired test table must compare every non-A6 method to A6")
    if {int(row["n_pairs"]) for row in comparison_index.values()} != {14}:
        raise ValueError("paired learned comparisons must contain 14 pairs")
    return {
        "results": indexed,
        "comparisons": comparison_index,
        "result_path": result_path,
        "comparison_path": comparison_path,
    }


def write_learned_table(path: Path, evidence: dict[str, object]) -> None:
    """Write the primary held-out learned comparison."""
    results = evidence["results"]
    comparisons = evidence["comparisons"]
    assert isinstance(results, dict)
    assert isinstance(comparisons, dict)
    rows = []
    for method, label in LEARNED_NAMES.items():
        mpjpe = results[(method, "mpjpe")]
        rom = results[(method, "rom_retention")]
        peak = results[(method, "peak_angular_velocity_retention")]
        if method == "A6":
            difference = "--"
            p_holm = "--"
        else:
            comparison = comparisons[method]
            difference = (
                f"{1000.0 * float(comparison['mean_difference']):+.2f} "
                f"[{1000.0 * float(comparison['ci_low']):+.2f}, "
                f"{1000.0 * float(comparison['ci_high']):+.2f}]"
            )
            p_holm = f"{float(comparison['holm_p']):.4f}"
        rows.append(
            f"{method} & {label} & "
            f"{1000.0 * float(mpjpe['mean']):.2f} $\\pm$ "
            f"{1000.0 * float(mpjpe['std']):.2f} & "
            f"{difference} & {p_holm} & "
            f"{float(rom['mean']):.3f} & {float(peak['mean']):.3f} \\\\"
        )
    text = """\\begin{table*}[t]
\\centering
\\caption{Primary learned comparison on the held-out test set ($N=14$).
MPJPE is measured in millimeters after one sequence-level Sim3 alignment to the
same-video triangulated pseudo-reference and is reported as person-level
mean $\\pm$ SD. Differences are method minus A6 with person-bootstrap 95\\%
confidence intervals and Holm-adjusted paired Wilcoxon $p$-values. ROM and
peak-$\\omega$ are A3-relative retention ratios; they do not use the
triangulated pseudo-reference as their denominator.}
\\label{tab:learned-results}
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{c l c c c c c}
\\toprule
ID & Method & MPJPE (mm) & $\\Delta$ vs A6 (mm) [95\\% CI] &
$p_{\\mathrm{Holm}}$ & A3-relative ROM & A3-relative peak-$\\omega$ \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
}
\\end{table*}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_unity_evidence(
    zero_shot_path: Path | None = None,
    calibrated_path: Path | None = None,
) -> dict[str, object]:
    """Load Unity native-3D results without mixing input regimes."""
    root = project_root()
    zero_shot_path = (
        zero_shot_path
        or root / "local/runs/unity_benchmark/report/results.json"
    )
    calibrated_path = (
        calibrated_path
        or root
        / "local/runs/unity_benchmark/extrinsic_learning/report/results.json"
    )
    zero = json.loads(zero_shot_path.read_text(encoding="utf-8"))
    calibrated = json.loads(calibrated_path.read_text(encoding="utf-8"))
    provenance = zero.get("provenance", {})
    if provenance.get("expected_samples") != 199:
        raise ValueError("Unity report does not match the frozen 199-sample audit")
    if provenance.get("expected_sequences") != 3:
        raise ValueError("Unity report must contain the frozen three sequences")
    if provenance.get("alignment") != "one_sim3_per_sequence":
        raise ValueError("Unity report must use one Sim3 per sequence")
    zero_index = {
        row["method"]: row
        for row in zero.get("valid_ranking", [])
    }
    required_zero = {"avg_world_face_ref", "A6", "triangulation_sam3d2d"}
    if not required_zero.issubset(zero_index):
        raise ValueError("Unity zero-shot report lacks required comparators")
    calibrated_index = {
        row["method"]: row
        for row in calibrated.get("by_method", [])
    }
    required_calibrated = {"extrinsic_gate", "learnable_triangulation"}
    if not required_calibrated.issubset(calibrated_index):
        raise ValueError("Unity calibrated report lacks required learned baselines")
    for method in required_calibrated:
        row = calibrated_index[method]
        if row.get("folds") != 2 or row.get("seeds") != 3:
            raise ValueError(
                f"Unity calibrated {method} must contain two folds and three seeds"
            )
    return {
        "zero_shot": zero_index,
        "calibrated": calibrated_index,
        "zero_shot_path": zero_shot_path,
        "calibrated_path": calibrated_path,
    }


def write_unity_table(path: Path, evidence: dict[str, object]) -> None:
    """Write a regime-stratified Unity native-3D benchmark table."""
    zero = evidence["zero_shot"]
    calibrated = evidence["calibrated"]
    assert isinstance(zero, dict)
    assert isinstance(calibrated, dict)
    rows = [
        (
            "Uncalibrated direct 3D, zero-shot",
            "World-coordinate average",
            zero["avg_world_face_ref"]["mpjpe_mm"],
            zero["avg_world_face_ref"]["angle_mae_deg"],
            "--",
        ),
        (
            "Uncalibrated direct 3D, zero-shot",
            "A6",
            zero["A6"]["mpjpe_mm"],
            zero["A6"]["angle_mae_deg"],
            "--",
        ),
        (
            "Calibrated 3D, Unity-supervised",
            "Extrinsic gate",
            calibrated["extrinsic_gate"]["macro_mpjpe_mm"],
            calibrated["extrinsic_gate"]["macro_angle_mae_deg"],
            "2 folds $\\times$ 3 seeds",
        ),
        (
            "Calibrated 2D $\\rightarrow$ 3D, zero-shot",
            "Triangulation (SAM3D 2D)",
            zero["triangulation_sam3d2d"]["mpjpe_mm"],
            zero["triangulation_sam3d2d"]["angle_mae_deg"],
            "--",
        ),
        (
            "Calibrated 2D $\\rightarrow$ 3D, Unity-supervised",
            "Learnable triangulation",
            calibrated["learnable_triangulation"]["macro_mpjpe_mm"],
            calibrated["learnable_triangulation"]["macro_angle_mae_deg"],
            "2 folds $\\times$ 3 seeds",
        ),
    ]
    rendered = [
        f"{regime} & {method} & {float(mpjpe):.3f} & "
        f"{float(angle):.3f} & {replication} \\\\"
        for regime, method, mpjpe, angle, replication in rows
    ]
    text = """\\begin{table*}[t]
\\centering
\\caption{Limited external evaluation against Unity native 3D (199 samples,
three sequences, one avatar/environment, Unity16 joints). All rows use one
sequence-level Sim3 before scoring. The calibrated 2D and uncalibrated direct 3D
input regimes use different evidence and supervision and must not be pooled into
a single ranking.}
\\label{tab:unity-benchmark}
\\small
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{l l r r l}
\\toprule
Input regime & Method & MPJPE (mm) & Angle MAE ($^\\circ$) & Replication \\\\
\\midrule
""" + "\n".join(rendered) + """
\\bottomrule
\\end{tabular}
}
\\end{table*}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def publish_cohort_assets(
    paper_root: Path = PAPER_ROOT,
    statistics_root: Path | None = None,
    report_root: Path | None = None,
) -> dict[str, int]:
    """Validate and publish the frozen centered cohort analysis."""
    root = project_root()
    base = root / "local/runs/cohort_cycle/analysis"
    statistics_root = statistics_root or base / "statistics_midcycle_v2"
    report_root = report_root or base / "report_midcycle_v2"
    core = _read_csv_rows(statistics_root / "core_mixed_models.csv")
    no_fold = _read_csv_rows(
        statistics_root / "core_mixed_models_no_fold.csv"
    )
    sensitivity = _read_csv_rows(
        statistics_root / "sensitivity_mixed_models.csv"
    )
    expected_outcomes = {
        "trunk_axial_rotation_rom",
        "angular_speed_p95",
        "peak_rotation_phase",
        "trunk_tilt_p95",
        "wrist_lead_p95",
        "cycle_duration",
        "log_dimensionless_angular_jerk",
        "whole_body_repeatability",
    }
    if {row["outcome"] for row in core} != expected_outcomes or len(core) != 8:
        raise ValueError("cohort primary table must contain eight outcomes")
    if {row["outcome"] for row in no_fold} != expected_outcomes or len(no_fold) != 8:
        raise ValueError("cohort no-fold table must contain eight outcomes")
    for row in core:
        if (
            row["model_status"] != "mixed_effect"
            or row["converged"] != "True"
            or int(row["n_people"]) != 137
            or int(row["n_cycles"]) != 928
            or float(row["cycle_reference"]) != 0.5
            or row["include_outer_fold"] != "True"
        ):
            raise ValueError(f"invalid primary cohort model: {row}")
    for row in no_fold:
        if (
            row["model_status"] != "mixed_effect"
            or row["converged"] != "True"
            or float(row["cycle_reference"]) != 0.5
            or row["include_outer_fold"] != "False"
        ):
            raise ValueError(f"invalid no-fold cohort model: {row}")
    expected_sources = {"oof_a6", "face", "side", "deterministic"}
    if (
        len(sensitivity) != 32
        or {row["source"] for row in sensitivity} != expected_sources
        or {row["outcome"] for row in sensitivity} != expected_outcomes
    ):
        raise ValueError("cohort sensitivity must contain 4 x 8 models")
    for row in sensitivity:
        if (
            row["estimand"]
            != "mixed_model_mid_repetition_cohort_effect"
            or float(row["cycle_reference"]) != 0.5
            or row["converged"] != "True"
        ):
            raise ValueError(f"invalid cohort sensitivity model: {row}")

    copies = {
        statistics_root / "core_mixed_models.csv": (
            paper_root / "artifacts/cohort_core_mixed_models.csv"
        ),
        statistics_root / "core_mixed_models_no_fold.csv": (
            paper_root / "artifacts/cohort_core_mixed_models_no_fold.csv"
        ),
        statistics_root / "variability_results.csv": (
            paper_root / "artifacts/cohort_variability_results.csv"
        ),
        statistics_root / "sensitivity_mixed_models.csv": (
            paper_root / "artifacts/cohort_sensitivity_mixed_models.csv"
        ),
        statistics_root / "sensitivity_person_medians.csv": (
            paper_root / "artifacts/cohort_sensitivity_person_medians.csv"
        ),
        statistics_root / "phase_clusters.csv": (
            paper_root / "artifacts/cohort_phase_clusters.csv"
        ),
        report_root / "cohort_cycle_results.tex": (
            paper_root / "tables/cohort_cycle_results.tex"
        ),
        report_root / "cohort_cycle_analysis.pdf": (
            paper_root / "figures/cohort_cycle_analysis.pdf"
        ),
    }
    for source, destination in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _write_cohort_sensitivity_table(
        paper_root / "tables/cohort_cycle_sensitivity.tex",
        sensitivity,
    )
    return {"core_outcomes": len(core), "sensitivity_models": len(sensitivity)}


def _write_cohort_sensitivity_table(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    indexed = {
        (row["outcome"], row["source"]): row
        for row in rows
    }
    sources = (
        ("oof_a6", "OOF A6"),
        ("face", "Face only"),
        ("side", "Side only"),
        ("deterministic", "Deterministic"),
    )
    outcomes = (
        ("angular_speed_p95", "Angular speed (P95)"),
        (
            "log_dimensionless_angular_jerk",
            "Log dimensionless jerk",
        ),
    )
    body = []
    for outcome, label in outcomes:
        cells = []
        for source, _ in sources:
            row = indexed[(outcome, source)]
            cells.append(
                f"{float(row['cohort_effect']):.4f} "
                f"[{float(row['cohort_ci_low']):.4f}, "
                f"{float(row['cohort_ci_high']):.4f}] "
                f"({float(row['cohort_p_holm_within_source']):.4f})"
            )
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    headers = " & ".join(label for _, label in sources)
    text = """\\begin{table*}[t]
\\centering
\\caption{Pose-source sensitivity using the same centered cycle-level mixed model
as the primary analysis. Cells report the mid-repetition
elderly-minus-student coefficient [95\\% CI] with
$p_{\\mathrm{Holm}}$ corrected across eight outcomes within each source.}
\\label{tab:cohort-cycle-sensitivity}
\\scriptsize
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{l c c c c}
\\toprule
Outcome & """ + headers + r""" \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def default_metrics_path() -> Path:
    explicit = os.environ.get("GYMNASTICS_SOURCE_ROOT")
    candidates = []
    if explicit:
        candidates.append(
            Path(explicit)
            / "local/runs/fuse_experiments/metrics_by_person.csv"
        )
        candidates.append(
            Path(explicit) / "logs/fuse_experiments/metrics_by_person.csv"
        )
    candidates.append(
        project_root()
        / "local/runs/fuse_experiments/metrics_by_person.csv"
    )
    candidates.append(WORKTREE_ROOT / "logs/fuse_experiments/metrics_by_person.csv")
    if WORKTREE_ROOT.parent.name == ".worktrees":
        candidates.append(WORKTREE_ROOT.parent.parent / "logs/fuse_experiments/metrics_by_person.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find metrics_by_person.csv. Checked:\n{rendered}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], seed: int, repetitions: int = 10_000) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    means = [statistics.fmean(values[rng.randrange(count)] for _ in range(count)) for _ in range(repetitions)]
    return quantile(means, 0.025), quantile(means, 0.975)


def load_metrics(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"person_id", "method", "mpjpe"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} does not contain required columns {sorted(required)}")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")

    grouped: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    people: set[str] = set()
    for row in rows:
        key = (row["person_id"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate person-method row: {key}")
        seen.add(key)
        value = float(row["mpjpe"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite MPJPE for {key}")
        grouped[row["method"]].append(value)
        people.add(row["person_id"])

    if len(grouped) != EXPECTED_METHODS or set(grouped) != set(DISPLAY_NAMES):
        raise ValueError(f"Expected methods {sorted(DISPLAY_NAMES)}, found {sorted(grouped)}")
    if len(people) != EXPECTED_PEOPLE:
        raise ValueError(f"Expected {EXPECTED_PEOPLE} people, found {len(people)}")
    if any(len(values) != EXPECTED_PEOPLE for values in grouped.values()):
        raise ValueError("Each method must contain one row for every person")
    return grouped


def summarize(grouped: dict[str, list[float]]) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for index, (method, values) in enumerate(sorted(grouped.items())):
        ci_low, ci_high = bootstrap_mean_ci(values, seed=20260721 + index)
        summaries.append(
            {
                "method": method,
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values),
                "median": statistics.median(values),
                "q1": quantile(values, 0.25),
                "q3": quantile(values, 0.75),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    summaries.sort(key=lambda item: float(item["mean"]))
    overall_best = summaries[0]
    if (
        overall_best["method"] != EXPECTED_OVERALL_BEST
        or abs(float(overall_best["mean"]) - EXPECTED_OVERALL_BEST_MEAN) > 5e-7
    ):
        raise ValueError(f"Verified overall baseline changed: best row is {overall_best}")
    leakage_free_best = next(
        item for item in summaries if str(item["method"]) not in LEAKY_METHODS
    )
    if (
        leakage_free_best["method"] != EXPECTED_LEAKAGE_FREE_BEST
        or abs(float(leakage_free_best["mean"]) - EXPECTED_LEAKAGE_FREE_BEST_MEAN)
        > 5e-7
    ):
        raise ValueError(
            f"Verified leakage-free baseline changed: best eligible row is {leakage_free_best}"
        )
    return summaries


def write_summary(path: Path, summaries: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "n", "mean", "std", "median", "q1", "q3", "ci_low", "ci_high"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_table(path: Path, summaries: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, item in enumerate(summaries, start=1):
        method = str(item["method"])
        label = DISPLAY_NAMES[method]
        if method == EXPECTED_LEAKAGE_FREE_BEST:
            label = f"\\textbf{{{label} (recommended leakage-free)}}"
        elif method in LEAKY_METHODS:
            label = f"{label} (leaky diagnostic)"
        rows.append(
            f"{rank} & {label} & {int(item['n'])} & "
            f"{float(item['mean']):.4f} & {float(item['std']):.4f} & "
            f"{float(item['median']):.4f} & "
            f"[{float(item['q1']):.4f}, {float(item['q3']):.4f}] & "
            f"[{float(item['ci_low']):.4f}, {float(item['ci_high']):.4f}] \\\\"
        )
    text = """\\begin{table*}[t]
\\centering
\\caption{Person-level agreement with the triangulated pseudo-reference for the
verified deterministic experiment matrix. Values are in repository coordinate
units. The confidence interval is a fixed-seed percentile bootstrap over people
and does not establish absolute 3D accuracy. Lower MPJPE is better. The
pseudo-reference-fitted joint-weight row is a legacy diagnostic and is not an
eligible label-free comparator.}
\\label{tab:deterministic-baselines}
\\small
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{r l r S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] c c}
\\toprule
Rank & Method & $N$ & {Mean} & {SD} & {Median} & {IQR} & {95\\% CI of mean} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
}
\\end{table*}
"""
    path.write_text(text, encoding="utf-8")


def write_figure(path: Path, summaries: list[dict[str, float | int | str]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PAPER_ROOT / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PAPER_ROOT / ".cache"))
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [DISPLAY_NAMES[str(item["method"])] for item in summaries]
    means = [float(item["mean"]) for item in summaries]
    lower = [mean - float(item["ci_low"]) for mean, item in zip(means, summaries)]
    upper = [float(item["ci_high"]) - mean for mean, item in zip(means, summaries)]
    colors = [
        "#009E73"
        if str(item["method"]) == EXPECTED_LEAKAGE_FREE_BEST
        else "#E69F00"
        if str(item["method"]) in LEAKY_METHODS
        else "#0072B2"
        for item in summaries
    ]

    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    positions = list(range(len(labels)))
    axis.errorbar(means, positions, xerr=[lower, upper], fmt="none", ecolor="#555555", capsize=3, linewidth=1.1)
    axis.scatter(means, positions, c=colors, s=35, zorder=3)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Person-level MPJPE to triangulated pseudo-reference\n(repository coordinate units; lower is better)")
    axis.grid(axis="x", color="#D0D0D0", linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_csv or default_metrics_path()
    grouped = load_metrics(metrics_path)
    summaries = summarize(grouped)
    write_summary(PAPER_ROOT / "artifacts/deterministic_summary.csv", summaries)
    write_table(PAPER_ROOT / "tables/deterministic_baselines.tex", summaries)
    write_figure(PAPER_ROOT / "figures/deterministic_mpjpe.pdf", summaries)
    learned = load_learned_evidence()
    write_learned_table(PAPER_ROOT / "tables/learned_results.tex", learned)
    unity = load_unity_evidence()
    write_unity_table(PAPER_ROOT / "tables/unity_benchmark.tex", unity)
    publish_cohort_assets()
    print(f"verified methods={len(grouped)} people={len(next(iter(grouped.values())))} rows={sum(map(len, grouped.values()))}")
    print(f"source={metrics_path}")


if __name__ == "__main__":
    main()
