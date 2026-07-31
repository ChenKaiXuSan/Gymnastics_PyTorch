# Results Reporting Completeness Design

## Objective

Create one reproducible, project-level results summary and revise the manuscript
so that every quantitative claim names its evaluation cohort and evidence
status. The revision must not turn unfinished robustness, multi-seed, or external
benchmark work into completed evidence.

## Evidence Semantics

The reporting hierarchy is:

1. Learned fusion generalization is reported primarily on the held-out
   14-person test split from the saved split manifest.
2. Learned fusion results over all 137 people remain available, but are labelled
   descriptive because that cohort includes training and validation people.
3. Fixed-corruption recovery is reported separately on the 27-person validation
   split, which is the only split where that diagnostic was measured.
4. Deterministic fusion remains a paired 137-person comparison because no model
   fitting split is involved. The triangulated reference is explicitly called
   pseudo-GT, and GT-derived joint weighting remains marked as leakage.
5. Classification results are aggregated across the three existing person-level
   folds. They are presented as mean and sample standard deviation, without
   claiming repeated-seed uncertainty.
6. Missing offset-robustness, multi-seed inference/evaluation, and public or
   independent-GT validation remain listed as pending limitations.

## Architecture

Add `gymnastics.analysis.project_results` as the source of truth for summary
generation. It reads immutable result artefacts, validates cohort membership,
produces machine-readable CSV files plus a Markdown report under
`local/runs/analysis/project_results/`, and exposes small pure functions for
unit testing.

The tracked `docs/results_summary.md` is a stable overview explaining the
evidence hierarchy and pointing to generated local artefacts. The ignored local
paper is revised in place so its protocol, learned-results table, Results,
Discussion, and Conclusion use the same cohort definitions.

## Generated Outputs

- `local/runs/analysis/project_results/learned_results_by_split.csv`
- `local/runs/analysis/project_results/learned_test_comparisons.csv`
- `local/runs/analysis/project_results/classification_summary.csv`
- `local/runs/analysis/project_results/RESULTS_SUMMARY.md`

The learned comparison output includes paired mean differences, a bootstrap
confidence interval, Wilcoxon p-values, and Holm-adjusted p-values. Statistical
tests compare methods only where the same people have both measurements.

## Validation

- Unit tests cover split assignment, unavailable corruption values,
  Holm correction, paired statistics, and classification fold aggregation.
- The generator runs against the current full local artefacts and validates the
  96/27/14 split with 137 unique people.
- Focused and full project tests run in the `gymnastic` conda environment.
- The Image and Vision Computing paper is rebuilt after revision.

## Non-Goals

- No new GPU training or inference.
- No imputation of missing robustness or seed results.
- No claims that triangulated pseudo-GT is an independent ground truth.
- No Git commit or push unless separately requested.
