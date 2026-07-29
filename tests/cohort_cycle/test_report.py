from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gymnastics.analysis.cohort_cycle.features import CORE_OUTCOMES
from gymnastics.analysis.cohort_cycle.report import (
    _repetition_panel_title,
    render_report,
)


def _write_finalized_inputs(root: Path) -> tuple[Path, Path]:
    features = root / "features"
    statistics = root / "statistics"
    features.mkdir()
    statistics.mkdir()

    people = []
    for person_index in range(12):
        cohort = "student" if person_index < 6 else "elderly"
        row = {
            "person_id": str(person_index),
            "cohort": cohort,
            "outer_fold": person_index % 2,
        }
        for outcome_index, outcome in enumerate(CORE_OUTCOMES):
            row[f"{outcome}_median"] = (
                outcome_index + person_index + (2 if cohort == "elderly" else 0)
            )
            row[f"{outcome}_mad"] = 0.1 * (outcome_index + 1)
        people.append(row)
    pd.DataFrame(people).to_csv(
        features / "person_features.csv",
        index=False,
    )

    phase = np.linspace(0.0, 1.0, 101)
    curves = np.stack(
        [
            phase + (0.2 if person["cohort"] == "elderly" else 0.0)
            for person in people
        ]
    )
    np.savez_compressed(
        features / "phase_curves.npz",
        person_id=np.asarray([person["person_id"] for person in people]),
        cohort=np.asarray([person["cohort"] for person in people]),
        theta=curves,
        omega=curves,
        tilt=curves,
        wrist=curves,
    )

    core_rows = []
    variability_rows = []
    for index, outcome in enumerate(CORE_OUTCOMES, start=1):
        core_rows.append(
            {
                "outcome": outcome,
                "cohort_effect": index / 10.0,
                "cohort_ci_low": (
                    0.0004 if index == 1 else index / 10.0 - 0.05
                ),
                "cohort_ci_high": index / 10.0 + 0.05,
                "cohort_effect_standardized": index / 20.0,
                "cohort_p_holm": index / 100.0,
                "cycle_effect": index / 20.0,
                "interaction_effect": index / 30.0,
                "interaction_p_holm": index / 80.0,
                "cycle_reference": 0.5,
                "n_people": 12,
                "n_cycles": 72,
            }
        )
        variability_rows.append(
            {
                "outcome": outcome,
                "median_difference": index / 50.0,
                "median_difference_ci_low": index / 50.0 - 0.01,
                "median_difference_ci_high": index / 50.0 + 0.01,
                "p_holm": index / 90.0,
            }
        )
    pd.DataFrame(core_rows).to_csv(
        statistics / "core_mixed_models.csv",
        index=False,
    )
    pd.DataFrame(variability_rows).to_csv(
        statistics / "variability_results.csv",
        index=False,
    )
    sensitivity_rows = []
    for source_index, source in enumerate(
        ("oof_a6", "face", "side", "deterministic")
    ):
        for outcome_index, outcome in enumerate(CORE_OUTCOMES, start=1):
            effect = outcome_index / 10.0 + source_index / 20.0
            sensitivity_rows.append(
                {
                    "source": source,
                    "outcome": outcome,
                    "cohort_effect": effect,
                    "cohort_ci_low": effect - 0.05,
                    "cohort_ci_high": effect + 0.05,
                    "cohort_p_holm_within_source": outcome_index / 20.0,
                    "cycle_reference": 0.5,
                    "estimand": (
                        "mixed_model_mid_repetition_cohort_effect"
                    ),
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(
        statistics / "sensitivity_mixed_models.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "metric": "theta",
                "start_phase": 0.35,
                "end_phase": 0.55,
                "p_value": 0.02,
                "p_holm_across_metrics": 0.08,
            }
        ]
    ).to_csv(statistics / "phase_clusters.csv", index=False)
    (statistics / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "core_outcomes": list(CORE_OUTCOMES),
            }
        ),
        encoding="utf-8",
    )
    return features, statistics


def test_report_renders_eight_source_matched_rows_and_four_panels(
    tmp_path: Path,
):
    """Report code must not invent, omit, or independently alter results."""
    features, statistics = _write_finalized_inputs(tmp_path)
    output = tmp_path / "report"

    summary = render_report(features, statistics, output)

    assert summary["core_outcomes"] == 8
    table = pd.read_csv(output / "cohort_cycle_core.csv")
    assert list(table["outcome"]) == list(CORE_OUTCOMES)
    assert table.loc[0, "cohort_p_holm"] == pytest.approx(0.01)
    latex = (output / "cohort_cycle_results.tex").read_text(encoding="utf-8")
    assert "0.0100" in latex
    assert r"\begin{table}[H]" in latex
    assert r"\begin{table*}" not in latex
    assert r"\resizebox{\linewidth}{!}" in latex
    assert "mid-repetition reference" in latex
    assert "first-cycle reference" not in latex
    assert "0.0004" in latex
    figure = output / "cohort_cycle_analysis.pdf"
    assert figure.read_bytes().startswith(b"%PDF")
    assert (output / "report_manifest.json").is_file()
    assert summary["panel_c"] == "source_matched_sensitivity"


def test_report_refuses_incomplete_core_manifest(tmp_path: Path):
    """A seven-outcome analysis must never be presented as confirmatory."""
    features, statistics = _write_finalized_inputs(tmp_path)
    (statistics / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "core_outcomes": list(CORE_OUTCOMES[:-1]),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="eight prespecified"):
        render_report(features, statistics, tmp_path / "report")


def test_repetition_panel_names_metric_and_reports_adjusted_interaction():
    """The representative trend must not look like an unlabeled finding."""
    title = _repetition_panel_title(
        label="Axial rotation ROM",
        interaction_p_holm=1.0,
    )

    assert title == (
        "C  Axial rotation ROM repetition trend "
        "(interaction $p_{Holm}=1.000$)"
    )
