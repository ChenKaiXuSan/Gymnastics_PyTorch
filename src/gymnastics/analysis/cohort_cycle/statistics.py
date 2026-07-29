"""Hierarchical statistical analysis contracts."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .cohorts import sha256_file
from .features import CORE_OUTCOMES
from .multiplicity import holm_adjust
from .phase_statistics import cluster_permutation_test


COHORT_TERM = "C(cohort, Treatment(reference='student'))[T.elderly]"
CYCLE_TERM = "_cycle_position_centered"
INTERACTION_TERM = COHORT_TERM + f":{CYCLE_TERM}"


def fit_mixed_effect(
    table: pd.DataFrame,
    outcome: str,
    *,
    try_random_slope: bool = True,
    log_transform: bool = False,
    cycle_reference: float = 0.5,
    include_outer_fold: bool = True,
) -> dict[str, object]:
    """Fit the prespecified cycle-level mixed-effects model."""
    required = {
        "person_id",
        "cohort",
        "normalized_cycle_position",
        outcome,
    }
    if include_outer_fold:
        required.add("outer_fold")
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"mixed model columns missing: {sorted(missing)}")
    if not 0.0 <= cycle_reference <= 1.0:
        raise ValueError("cycle_reference must lie in [0, 1]")
    data = table[list(required)].dropna().copy()
    if set(data["cohort"]) != {"elderly", "student"}:
        raise ValueError("mixed model requires both cohorts")
    data[CYCLE_TERM] = (
        data["normalized_cycle_position"].astype(float) - cycle_reference
    )
    if log_transform:
        if np.any(data[outcome] <= 0):
            raise ValueError("log-transformed outcome must be positive")
        data["_model_outcome"] = np.log(data[outcome].astype(float))
    else:
        data["_model_outcome"] = data[outcome].astype(float)
    formula = (
        "_model_outcome ~ "
        "C(cohort, Treatment(reference='student'))"
        f" * {CYCLE_TERM}"
    )
    if include_outer_fold:
        formula += " + C(outer_fold)"
    fallback_reason: str | None = None
    result = None
    random_structure = "random_intercept"
    if try_random_slope:
        try:
            candidate = _fit_model(
                data,
                formula,
                re_formula=f"~{CYCLE_TERM}",
            )
            covariance = np.asarray(candidate.cov_re)
            if (
                not bool(candidate.converged)
                or covariance.size == 0
                or np.min(np.linalg.eigvalsh(covariance)) <= 1e-8
            ):
                raise ValueError("random-slope fit is singular or unconverged")
            result = candidate
            random_structure = "random_intercept_slope"
        except (ValueError, np.linalg.LinAlgError) as error:
            fallback_reason = str(error)
    if result is None:
        result = _fit_model(data, formula, re_formula="1")
    confidence = result.conf_int()
    covariance = np.asarray(result.cov_re, dtype=np.float64)
    random_intercept_variance = (
        float(covariance[0, 0]) if covariance.size else 0.0
    )
    random_slope_variance = (
        float(covariance[1, 1]) if covariance.shape == (2, 2) else 0.0
    )
    intercept_slope_covariance = (
        float(covariance[0, 1]) if covariance.shape == (2, 2) else 0.0
    )
    residual_scale = float(result.scale)
    if residual_scale > 0:
        maximum_absolute_standardized_residual = float(
            np.max(np.abs(np.asarray(result.resid))) / np.sqrt(residual_scale)
        )
    else:
        maximum_absolute_standardized_residual = float("nan")
    model_outcome_sd = float(data["_model_outcome"].std(ddof=1))
    cohort_effect = float(result.params[COHORT_TERM])
    return {
        "outcome": outcome,
        "transform": "log" if log_transform else "identity",
        "formula": formula,
        "cycle_reference": float(cycle_reference),
        "include_outer_fold": bool(include_outer_fold),
        "random_structure": random_structure,
        "fallback_reason": fallback_reason,
        "converged": bool(result.converged),
        "n_people": int(data["person_id"].nunique()),
        "n_cycles": int(len(data)),
        "cohort_effect": cohort_effect,
        "cohort_effect_se": float(result.bse[COHORT_TERM]),
        "cohort_effect_standardized": (
            cohort_effect / model_outcome_sd
            if model_outcome_sd > 0
            else float("nan")
        ),
        "cohort_ci_low": float(confidence.loc[COHORT_TERM, 0]),
        "cohort_ci_high": float(confidence.loc[COHORT_TERM, 1]),
        "cohort_p_value": float(result.pvalues[COHORT_TERM]),
        "cycle_effect": float(result.params[CYCLE_TERM]),
        "cycle_effect_se": float(result.bse[CYCLE_TERM]),
        "cycle_ci_low": float(
            confidence.loc[CYCLE_TERM, 0]
        ),
        "cycle_ci_high": float(
            confidence.loc[CYCLE_TERM, 1]
        ),
        "cycle_p_value": float(result.pvalues[CYCLE_TERM]),
        "interaction_effect": float(result.params[INTERACTION_TERM]),
        "interaction_effect_se": float(result.bse[INTERACTION_TERM]),
        "interaction_ci_low": float(confidence.loc[INTERACTION_TERM, 0]),
        "interaction_ci_high": float(confidence.loc[INTERACTION_TERM, 1]),
        "interaction_p_value": float(result.pvalues[INTERACTION_TERM]),
        "random_intercept_variance": max(0.0, random_intercept_variance),
        "random_slope_variance": max(0.0, random_slope_variance),
        "intercept_slope_covariance": intercept_slope_covariance,
        "model_outcome_sd": model_outcome_sd,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "residual_scale": residual_scale,
        "maximum_absolute_standardized_residual": (
            maximum_absolute_standardized_residual
        ),
    }


def adjust_phase_cluster_families(
    rows: list[dict[str, object]],
    *,
    metrics: tuple[str, ...] = ("theta", "omega", "tilt", "wrist"),
) -> list[dict[str, object]]:
    """Apply Holm correction to the minimum cluster p-value per descriptor."""
    if not rows:
        return []
    minimum_by_metric = {
        metric: min(
            (
                float(row["p_value"])
                for row in rows
                if row["metric"] == metric
            ),
            default=1.0,
        )
        for metric in metrics
    }
    adjusted = holm_adjust(
        np.asarray([minimum_by_metric[metric] for metric in metrics])
    )
    adjusted_by_metric = dict(zip(metrics, adjusted, strict=True))
    return [
        {
            **row,
            "p_holm_across_metrics": float(
                max(
                    float(row["p_value"]),
                    adjusted_by_metric[str(row["metric"])],
                )
            ),
        }
        for row in rows
    ]


def _permutation_fallback_model(
    cycles: pd.DataFrame,
    outcome: str,
    *,
    permutations: int,
    seed: int,
    error: Exception,
    cycle_reference: float,
    include_outer_fold: bool,
) -> dict[str, object]:
    """Return an explicitly labelled person-level fallback result."""
    person_values = (
        cycles.groupby(["person_id", "cohort"], as_index=False)[outcome]
        .median()
        .rename(columns={outcome: "_fallback_value"})
    )
    permutation = person_label_permutation(
        person_values,
        "_fallback_value",
        permutations=permutations,
        seed=seed,
    )
    outcome_sd = float(person_values["_fallback_value"].std(ddof=1))
    cohort_effect = float(permutation["median_difference"])
    return {
        "outcome": outcome,
        "transform": "identity",
        "formula": "",
        "cycle_reference": cycle_reference,
        "include_outer_fold": include_outer_fold,
        "random_structure": "none",
        "fallback_reason": str(error),
        "converged": False,
        "n_people": int(person_values["person_id"].nunique()),
        "n_cycles": int(cycles[outcome].notna().sum()),
        "cohort_effect": cohort_effect,
        "cohort_effect_se": np.nan,
        "cohort_effect_standardized": (
            cohort_effect / outcome_sd if outcome_sd > 0 else np.nan
        ),
        "cohort_ci_low": np.nan,
        "cohort_ci_high": np.nan,
        "cohort_p_value": permutation["p_value"],
        "cycle_effect": np.nan,
        "cycle_effect_se": np.nan,
        "cycle_ci_low": np.nan,
        "cycle_ci_high": np.nan,
        "cycle_p_value": 1.0,
        "interaction_effect": np.nan,
        "interaction_effect_se": np.nan,
        "interaction_ci_low": np.nan,
        "interaction_ci_high": np.nan,
        "interaction_p_value": 1.0,
        "random_intercept_variance": np.nan,
        "random_slope_variance": np.nan,
        "intercept_slope_covariance": np.nan,
        "model_outcome_sd": outcome_sd,
        "aic": np.nan,
        "bic": np.nan,
        "residual_scale": np.nan,
        "maximum_absolute_standardized_residual": np.nan,
        "model_status": "person_permutation_fallback",
    }


def person_label_permutation(
    table: pd.DataFrame,
    value_column: str,
    *,
    permutations: int,
    seed: int,
    bootstrap_samples: int = 1000,
) -> dict[str, float | int]:
    """Compare person-level values by permuting cohort labels over people."""
    data = table[["person_id", "cohort", value_column]].dropna().copy()
    if data["person_id"].duplicated().any():
        raise ValueError("person permutation input must have one row per person")
    values = data[value_column].to_numpy(dtype=np.float64)
    labels = data["cohort"].to_numpy()
    elderly = values[labels == "elderly"]
    student = values[labels == "student"]
    if len(elderly) < 2 or len(student) < 2:
        raise ValueError("person permutation requires both cohorts")
    observed = float(np.median(elderly) - np.median(student))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        difference = float(
            np.median(values[permuted == "elderly"])
            - np.median(values[permuted == "student"])
        )
        exceed += abs(difference) >= abs(observed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        resampled_elderly = rng.choice(
            elderly,
            size=len(elderly),
            replace=True,
        )
        resampled_student = rng.choice(
            student,
            size=len(student),
            replace=True,
        )
        bootstrap[index] = (
            np.median(resampled_elderly) - np.median(resampled_student)
        )
    confidence = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n_elderly": len(elderly),
        "n_student": len(student),
        "elderly_median": float(np.median(elderly)),
        "student_median": float(np.median(student)),
        "elderly_q25": float(np.quantile(elderly, 0.25)),
        "elderly_q75": float(np.quantile(elderly, 0.75)),
        "student_q25": float(np.quantile(student, 0.25)),
        "student_q75": float(np.quantile(student, 0.75)),
        "median_difference": observed,
        "median_difference_ci_low": float(confidence[0]),
        "median_difference_ci_high": float(confidence[1]),
        "p_value": (exceed + 1) / float(permutations + 1),
        "hedges_g": hedges_g(elderly, student),
        "cliffs_delta": cliffs_delta(elderly, student),
    }


def hedges_g(elderly: np.ndarray, student: np.ndarray) -> float:
    """Bias-corrected standardized mean difference, elderly minus student."""
    first = np.asarray(elderly, dtype=np.float64)
    second = np.asarray(student, dtype=np.float64)
    degrees = len(first) + len(second) - 2
    if degrees <= 0:
        raise ValueError("Hedges g requires at least two observations")
    pooled = np.sqrt(
        (
            (len(first) - 1) * np.var(first, ddof=1)
            + (len(second) - 1) * np.var(second, ddof=1)
        )
        / degrees
    )
    if pooled <= 0:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0)
    return float(correction * (np.mean(first) - np.mean(second)) / pooled)


def cliffs_delta(elderly: np.ndarray, student: np.ndarray) -> float:
    """Probability-of-superiority contrast, elderly minus student."""
    first = np.asarray(elderly, dtype=np.float64)
    second = np.asarray(student, dtype=np.float64)
    comparisons = first[:, None] - second[None, :]
    return float(
        (np.sum(comparisons > 0) - np.sum(comparisons < 0))
        / comparisons.size
    )


def intraclass_correlation(
    table: pd.DataFrame,
    value_column: str,
) -> float:
    """One-way random-effects ICC(1,1) for unbalanced repeated cycles."""
    data = table[["person_id", value_column]].dropna()
    groups = [
        group[value_column].to_numpy(dtype=np.float64)
        for _, group in data.groupby("person_id")
    ]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        raise ValueError("ICC requires at least two people with two cycles")
    sizes = np.asarray([len(group) for group in groups], dtype=np.float64)
    means = np.asarray([np.mean(group) for group in groups])
    total = int(sizes.sum())
    people = len(groups)
    grand = float(
        sum(np.sum(group) for group in groups) / total
    )
    between = float(np.sum(sizes * (means - grand) ** 2) / (people - 1))
    within = float(
        sum(np.sum((group - np.mean(group)) ** 2) for group in groups)
        / (total - people)
    )
    effective_repeats = (
        total - float(np.sum(sizes**2)) / total
    ) / (people - 1)
    denominator = between + (effective_repeats - 1.0) * within
    if denominator <= 0:
        return float("nan")
    return float((between - within) / denominator)


def bootstrap_icc(
    table: pd.DataFrame,
    value_column: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Person-bootstrap confidence interval for ICC(1,1)."""
    data = table[["person_id", value_column]].dropna()
    person_ids = data["person_id"].astype(str).unique()
    if len(person_ids) < 2:
        raise ValueError("ICC bootstrap requires at least two people")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = rng.choice(
            person_ids,
            size=len(person_ids),
            replace=True,
        )
        parts = []
        for bootstrap_id, person_id in enumerate(selected):
            part = data.loc[
                data["person_id"].astype(str) == person_id
            ].copy()
            part["person_id"] = f"bootstrap_{bootstrap_id}"
            parts.append(part)
        estimate = intraclass_correlation(
            pd.concat(parts, ignore_index=True),
            value_column,
        )
        if np.isfinite(estimate):
            estimates.append(float(np.clip(estimate, -1.0, 1.0)))
    if not estimates:
        return float("nan"), float("nan")
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def analyze_feature_artifacts(
    feature_root: str | Path,
    output_root: str | Path,
    *,
    permutations: int,
    seed: int,
    try_random_slope: bool = True,
    log_outcomes: set[str] | None = None,
    sensitivity_sources: Mapping[str, str | Path] | None = None,
) -> dict[str, int]:
    """Run the prespecified core, variability, ICC, and phase analyses."""
    source = Path(feature_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"analysis staging already exists: {staging}")
    cycles = pd.read_csv(source / "cycle_features.csv")
    people = pd.read_csv(source / "person_features.csv")
    cycles = cycles.loc[cycles["eligible"].astype(bool)].copy()
    log_set = set(log_outcomes or ())

    core_rows: list[dict[str, object]] = []
    no_fold_rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {
        "cycle_reference": 0.5,
        "primary": {},
        "no_outer_fold": {},
    }
    for outcome_index, outcome in enumerate(CORE_OUTCOMES):
        try:
            result = fit_mixed_effect(
                cycles,
                outcome,
                try_random_slope=try_random_slope,
                log_transform=outcome in log_set,
                cycle_reference=0.5,
                include_outer_fold=True,
            )
            result["model_status"] = "mixed_effect"
        except Exception as error:  # robust prespecified fallback
            result = _permutation_fallback_model(
                cycles,
                outcome,
                permutations=permutations,
                seed=seed + outcome_index,
                error=error,
                cycle_reference=0.5,
                include_outer_fold=True,
            )
        try:
            no_fold_result = fit_mixed_effect(
                cycles,
                outcome,
                try_random_slope=try_random_slope,
                log_transform=outcome in log_set,
                cycle_reference=0.5,
                include_outer_fold=False,
            )
            no_fold_result["model_status"] = "mixed_effect"
        except Exception as error:
            no_fold_result = _permutation_fallback_model(
                cycles,
                outcome,
                permutations=permutations,
                seed=seed + 50 + outcome_index,
                error=error,
                cycle_reference=0.5,
                include_outer_fold=False,
            )
        core_rows.append(result)
        no_fold_rows.append(no_fold_result)
        diagnostics["primary"][outcome] = {
            "model_status": result["model_status"],
            "converged": bool(result["converged"]),
            "random_structure": result["random_structure"],
            "fallback_reason": result["fallback_reason"],
            "random_intercept_variance": result[
                "random_intercept_variance"
            ],
            "random_slope_variance": result["random_slope_variance"],
            "intercept_slope_covariance": result[
                "intercept_slope_covariance"
            ],
            "maximum_absolute_standardized_residual": result[
                "maximum_absolute_standardized_residual"
            ],
        }
        diagnostics["no_outer_fold"][outcome] = {
            "model_status": no_fold_result["model_status"],
            "converged": bool(no_fold_result["converged"]),
            "random_structure": no_fold_result["random_structure"],
            "fallback_reason": no_fold_result["fallback_reason"],
            "random_intercept_variance": no_fold_result[
                "random_intercept_variance"
            ],
            "random_slope_variance": no_fold_result[
                "random_slope_variance"
            ],
            "intercept_slope_covariance": no_fold_result[
                "intercept_slope_covariance"
            ],
            "maximum_absolute_standardized_residual": no_fold_result[
                "maximum_absolute_standardized_residual"
            ],
        }
    cohort_adjusted = holm_adjust(
        np.asarray([row["cohort_p_value"] for row in core_rows])
    )
    interaction_adjusted = holm_adjust(
        np.asarray([row["interaction_p_value"] for row in core_rows])
    )
    for row, cohort_p, interaction_p in zip(
        core_rows,
        cohort_adjusted,
        interaction_adjusted,
        strict=True,
    ):
        row["cohort_p_holm"] = float(cohort_p)
        row["interaction_p_holm"] = float(interaction_p)
    no_fold_cohort_adjusted = holm_adjust(
        np.asarray([row["cohort_p_value"] for row in no_fold_rows])
    )
    no_fold_interaction_adjusted = holm_adjust(
        np.asarray([row["interaction_p_value"] for row in no_fold_rows])
    )
    for row, cohort_p, interaction_p in zip(
        no_fold_rows,
        no_fold_cohort_adjusted,
        no_fold_interaction_adjusted,
        strict=True,
    ):
        row["cohort_p_holm"] = float(cohort_p)
        row["interaction_p_holm"] = float(interaction_p)

    variability_rows: list[dict[str, object]] = []
    for outcome_index, outcome in enumerate(CORE_OUTCOMES):
        column = f"{outcome}_mad"
        result = person_label_permutation(
            people.rename(columns={column: "_mad"}),
            "_mad",
            permutations=permutations,
            seed=seed + 100 + outcome_index,
        )
        result["outcome"] = outcome
        variability_rows.append(result)
    variability_adjusted = holm_adjust(
        np.asarray([row["p_value"] for row in variability_rows])
    )
    for row, adjusted in zip(
        variability_rows,
        variability_adjusted,
        strict=True,
    ):
        row["p_holm"] = float(adjusted)

    icc_rows: list[dict[str, object]] = []
    for outcome in CORE_OUTCOMES:
        for cohort, group in cycles.groupby("cohort"):
            try:
                value = intraclass_correlation(group, outcome)
                ci_low, ci_high = bootstrap_icc(
                    group,
                    outcome,
                    samples=max(100, min(permutations, 1000)),
                    seed=seed + 300 + len(icc_rows),
                )
            except ValueError:
                value = np.nan
                ci_low = np.nan
                ci_high = np.nan
            icc_rows.append(
                {
                    "outcome": outcome,
                    "cohort": cohort,
                    "people": int(group["person_id"].nunique()),
                    "cycles": int(group[outcome].notna().sum()),
                    "icc_1_1": value,
                    "icc_ci_low": ci_low,
                    "icc_ci_high": ci_high,
                }
            )

    phase_rows: list[dict[str, object]] = []
    with np.load(source / "phase_curves.npz", allow_pickle=False) as archive:
        for phase_index, metric in enumerate(
            ("theta", "omega", "tilt", "wrist")
        ):
            if metric not in archive:
                continue
            clusters = cluster_permutation_test(
                archive[metric],
                archive["person_id"],
                archive["cohort"],
                permutations=permutations,
                seed=seed + 200 + phase_index,
            )
            for cluster in clusters:
                phase_rows.append({"metric": metric, **cluster})
    phase_rows = adjust_phase_cluster_families(phase_rows)

    sensitivity_paths: dict[str, Path] = {"oof_a6": source}
    for name, path in (sensitivity_sources or {}).items():
        if name in sensitivity_paths:
            raise ValueError(f"duplicate sensitivity source: {name}")
        sensitivity_paths[str(name)] = Path(path)
    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_person_rows: list[dict[str, object]] = []
    sensitivity_exclusions: list[dict[str, object]] = []
    for source_index, (source_name, source_path) in enumerate(
        sensitivity_paths.items()
    ):
        cycle_path = source_path / "cycle_features.csv"
        if not cycle_path.is_file():
            sensitivity_exclusions.append(
                {
                    "source": source_name,
                    "artifact": str(cycle_path),
                    "reason": "missing cycle_features.csv",
                }
            )
            continue
        source_cycles = pd.read_csv(cycle_path)
        source_cycles = source_cycles.loc[
            source_cycles["eligible"].astype(bool)
        ].copy()
        source_people = pd.read_csv(source_path / "person_features.csv")
        source_rows: list[dict[str, object]] = []
        person_rows: list[dict[str, object]] = []
        for outcome_index, outcome in enumerate(CORE_OUTCOMES):
            try:
                mixed_result = fit_mixed_effect(
                    source_cycles,
                    outcome,
                    try_random_slope=try_random_slope,
                    log_transform=outcome in log_set,
                    cycle_reference=0.5,
                    include_outer_fold=True,
                )
            except Exception as error:
                sensitivity_exclusions.append(
                    {
                        "source": source_name,
                        "artifact": str(cycle_path),
                        "outcome": outcome,
                        "reason": f"mixed-model failure: {error}",
                    }
                )
            else:
                mixed_result.update(
                    {
                        "source": source_name,
                        "estimand": (
                            "mixed_model_mid_repetition_cohort_effect"
                        ),
                    }
                )
                source_rows.append(mixed_result)
            value_column = f"{outcome}_median"
            result = person_label_permutation(
                source_people,
                value_column,
                permutations=permutations,
                seed=seed + 400 + source_index * 20 + outcome_index,
                bootstrap_samples=max(100, min(permutations, 1000)),
            )
            person_rows.append(
                {
                    "outcome": outcome,
                    "source": source_name,
                    "effect": result["median_difference"],
                    "ci_low": result["median_difference_ci_low"],
                    "ci_high": result["median_difference_ci_high"],
                    "p_value": result["p_value"],
                    "n_elderly": result["n_elderly"],
                    "n_student": result["n_student"],
                }
            )
        if source_rows:
            adjusted = holm_adjust(
                np.asarray(
                    [row["cohort_p_value"] for row in source_rows]
                )
            )
            for row, p_holm in zip(source_rows, adjusted, strict=True):
                row["cohort_p_holm_within_source"] = float(p_holm)
        person_adjusted = holm_adjust(
            np.asarray([row["p_value"] for row in person_rows])
        )
        for row, p_holm in zip(person_rows, person_adjusted, strict=True):
            row["p_holm_within_source"] = float(p_holm)
        sensitivity_rows.extend(source_rows)
        sensitivity_person_rows.extend(person_rows)

    staging.mkdir(parents=True)
    try:
        pd.DataFrame(core_rows).to_csv(
            staging / "core_mixed_models.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(no_fold_rows).to_csv(
            staging / "core_mixed_models_no_fold.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(variability_rows).to_csv(
            staging / "variability_results.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(icc_rows).to_csv(
            staging / "icc_by_cohort.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(
            phase_rows,
            columns=(
                "metric",
                "start_index",
                "end_index",
                "start_phase",
                "end_phase",
                "cluster_mass",
                "p_value",
                "p_holm_across_metrics",
                "direction",
            ),
        ).to_csv(
            staging / "phase_clusters.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(
            columns=("outcome", "joint_or_region", "effect", "p_value", "p_fdr")
        ).to_csv(staging / "exploratory_fdr.csv", index=False)
        pd.DataFrame(sensitivity_rows).to_csv(
            staging / "sensitivity_mixed_models.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(sensitivity_person_rows).to_csv(
            staging / "sensitivity_person_medians.csv",
            index=False,
            float_format="%.10g",
        )
        pd.DataFrame(
            sensitivity_exclusions,
            columns=("source", "artifact", "outcome", "reason"),
        ).to_csv(
            staging / "sensitivity_exclusions.csv",
            index=False,
        )
        (staging / "model_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "core_outcomes": list(CORE_OUTCOMES),
            "correction_families": {
                "RQ1_cohort": list(CORE_OUTCOMES),
                "RQ2_variability": list(CORE_OUTCOMES),
                "RQ3_interaction": list(CORE_OUTCOMES),
                "phase_descriptors": ["theta", "omega", "tilt", "wrist"],
            },
            "inputs": {
                "cycle_features.csv": sha256_file(
                    source / "cycle_features.csv"
                ),
                "person_features.csv": sha256_file(
                    source / "person_features.csv"
                ),
                "phase_curves.npz": sha256_file(
                    source / "phase_curves.npz"
                ),
                "sensitivity_person_features": {
                    name: sha256_file(path / "person_features.csv")
                    for name, path in sensitivity_paths.items()
                },
                "sensitivity_cycle_features": {
                    name: sha256_file(path / "cycle_features.csv")
                    for name, path in sensitivity_paths.items()
                    if (path / "cycle_features.csv").is_file()
                },
            },
            "outputs": {
                path.name: sha256_file(path)
                for path in sorted(staging.iterdir())
                if path.is_file()
            },
        }
        (staging / "analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "core_outcomes": len(core_rows),
        "variability_outcomes": len(variability_rows),
        "phase_clusters": len(phase_rows),
        "sensitivity_rows": len(sensitivity_rows),
    }


def _fit_model(
    data: pd.DataFrame,
    formula: str,
    *,
    re_formula: str,
):
    model = smf.mixedlm(
        formula,
        data,
        groups=data["person_id"],
        re_formula=re_formula,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(
            reml=False,
            method="lbfgs",
            maxiter=500,
            disp=False,
        )
