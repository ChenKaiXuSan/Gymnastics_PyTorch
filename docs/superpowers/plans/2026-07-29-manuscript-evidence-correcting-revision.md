# Evidence-Correcting Manuscript Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the Neurocomputing manuscript so that its primary learned
results use the held-out private test set, its external-validity section reports
the completed Unity native-3D benchmark, and its cohort inference uses a
representative centered-cycle estimand.

**Architecture:** Keep generated evidence separate from prose. Extend the
cohort statistics and report generators in the cohort-analysis worktree, then
extend the manuscript asset generator to consume only audited CSV/JSON outputs.
Revise the multi-file LaTeX manuscript after all generated values are frozen,
and finish with automated consistency checks plus visual PDF inspection.

**Tech Stack:** Python 3.10+, NumPy, pandas, SciPy, statsmodels, matplotlib,
PyYAML, pytest, Ruff, LaTeX/latexmk, and the `gymnastic` conda environment.

## Global Constraints

- Keep A6 as the paper's method mainline, but do not claim that it improves
  positional accuracy over canonical deterministic fusion.
- Use the 14-person frozen test split as the primary private learned-model
  result; label 137-person learned values as descriptive only.
- Treat Unity native 3D as limited independent synthetic ground truth, not
  population-level or public-benchmark validation.
- Preserve the triangulated private sequence as a same-video pseudo-reference.
- Never infer missing ethics approval, consent, institution, covariates, or
  clinical validity.
- Generate numerical tables and figures from archived artifacts; do not
  hand-enter values when a source CSV or JSON exists.
- Use `conda run -n gymnastic ...` for project Python commands.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Lock the Evidence Map and Metric Semantics

**Files:**
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/artifacts/source_audit.md`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/scripts/generate_paper_assets.py`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/scripts/test_check_manuscript.py`
- Inspect:
  `src/gymnastics/analysis/project_results.py`
- Inspect:
  `src/gymnastics/fusion/rotation_aware/evaluation.py`
- Inspect:
  `src/gymnastics/benchmarks/unity/evaluation.py`

**Interfaces:**
- Consumes:
  `local/runs/analysis/project_results/learned_results_by_split.csv`,
  `learned_test_comparisons.csv`, private evaluation `report.json`, and Unity
  report JSON files.
- Produces:
  a source-audit mapping with fields `claim`, `artifact`, `population`,
  `alignment`, `reference`, `seed_count`, and `status`.

- [ ] **Step 1: Add a failing manuscript consistency test for evidence
  populations and metric definitions**

```python
def test_primary_learned_and_unity_claims_name_population_and_reference():
    manuscript = _manuscript_text()
    assert "held-out test set ($N=14$)" in manuscript
    assert "descriptive all-person" in manuscript
    assert "Unity native 3D" in manuscript
    assert "one sequence-level Sim3" in manuscript
    assert "A3-normalized" not in manuscript or "A3" in manuscript
```

- [ ] **Step 2: Run the targeted test and confirm that the current manuscript
  fails**

Run:

```bash
conda run -n gymnastic python -m pytest \
  .worktrees/rotation-aware-fusion/paper/neurocomputing/scripts/test_check_manuscript.py \
  -q
```

Expected: failure because the current primary learned table uses 137 people and
the Unity benchmark is absent.

- [ ] **Step 3: Trace the private evaluator definitions**

Run:

```bash
rg -n "similarity|Sim3|center|hip|rom_retention|peak_angular_velocity_retention" \
  src/gymnastics/fusion/rotation_aware \
  src/gymnastics/analysis/project_results.py
```

Record the exact alignment and retention denominator in `source_audit.md`.
If the implementation and current prose disagree, the implementation plus
archived artifact provenance is authoritative.

- [ ] **Step 4: Add evidence-source validation to the asset generator**

Implement loaders that reject:

```python
if int(test_rows["n_people"].unique().item()) != 14:
    raise ValueError("primary learned evidence must use the 14-person test set")
if unity_report["provenance"]["expected_samples"] != 199:
    raise ValueError("Unity report does not match the frozen 199-sample audit")
```

- [ ] **Step 5: Re-run the targeted tests**

Run the command from Step 2. Expected: the source-validation tests pass; the
manuscript-content assertion remains pending until Task 5.

- [ ] **Step 6: Commit the source-audit and validation changes**

Commit only tracked code or documentation files that belong to this task:

```bash
git add src/gymnastics/analysis/project_results.py tests/test_project_results.py
git commit -m "fix: lock manuscript evidence populations"
```

The ignored manuscript worktree remains uncommitted and is verified by its own
build artifacts.

---

### Task 2: Center the Cohort Estimand and Add Model Diagnostics

**Files:**
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/src/gymnastics/analysis/cohort_cycle/statistics.py`
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/tests/cohort_cycle/test_statistics.py`

**Interfaces:**
- Consumes:
  eligible `cycle_features.csv` rows with
  `normalized_cycle_position` in `[0, 1]`.
- Produces:
  `fit_mixed_effect(..., cycle_reference=0.5, include_outer_fold=True)` with
  centered cohort estimates, random-effect variances, fixed-effect standard
  errors, and model diagnostics.

- [ ] **Step 1: Write a failing centered-estimand test**

```python
def test_mixed_effect_cohort_effect_is_reported_at_mid_repetition():
    table = _synthetic_cycles()
    result = fit_mixed_effect(
        table,
        "outcome",
        cycle_reference=0.5,
        include_outer_fold=True,
    )
    assert result["cycle_reference"] == 0.5
    assert result["cohort_effect"] == pytest.approx(2.5, abs=0.2)
    assert result["random_intercept_variance"] >= 0.0
    assert result["random_slope_variance"] >= 0.0
```

- [ ] **Step 2: Write a failing no-fold sensitivity test**

```python
def test_mixed_effect_can_omit_artificial_outer_fold_adjustment():
    result = fit_mixed_effect(
        _synthetic_cycles(),
        "outcome",
        cycle_reference=0.5,
        include_outer_fold=False,
    )
    assert result["include_outer_fold"] is False
    assert "C(outer_fold)" not in result["formula"]
```

- [ ] **Step 3: Run the two tests and confirm failure**

```bash
conda run -n gymnastic python -m pytest \
  tests/cohort_cycle/test_statistics.py \
  -k "mid_repetition or omit_artificial" -q
```

Run from:
`.worktrees/oof-cohort-cycle-analysis`.

- [ ] **Step 4: Implement centered cycle position and diagnostic outputs**

Create `_cycle_position_centered`:

```python
data["_cycle_position_centered"] = (
    data["normalized_cycle_position"].astype(float) - cycle_reference
)
```

Use it in both fixed and random slopes. Return the formula, reference,
fixed-effect SEs, random-intercept variance, random-slope variance,
intercept-slope covariance, residual scale, AIC, BIC, convergence, and maximum
absolute standardized residual.

- [ ] **Step 5: Add the no-fold model as a named sensitivity output**

`analyze_feature_artifacts` writes:

- `core_mixed_models.csv`: centered primary model with outer-fold adjustment.
- `core_mixed_models_no_fold.csv`: same model without outer-fold fixed effects.
- `model_diagnostics.json`: both model families and their variance components.

- [ ] **Step 6: Run cohort statistics tests**

```bash
conda run -n gymnastic python -m pytest \
  tests/cohort_cycle/test_statistics.py -q
```

Expected: all statistics tests pass.

- [ ] **Step 7: Commit the centered-model implementation**

```bash
git add \
  src/gymnastics/analysis/cohort_cycle/statistics.py \
  tests/cohort_cycle/test_statistics.py
git commit -m "feat: center cohort effects at mid repetition"
```

Run from the cohort-analysis worktree.

---

### Task 3: Use One Mixed-Model Estimand Across Pose Sources

**Files:**
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/src/gymnastics/analysis/cohort_cycle/statistics.py`
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/tests/cohort_cycle/test_statistics.py`
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/src/gymnastics/analysis/cohort_cycle/cli.py`
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/tests/cohort_cycle/test_cli.py`

**Interfaces:**
- Consumes:
  each sensitivity source's `cycle_features.csv`.
- Produces:
  `sensitivity_mixed_models.csv` with the same centered model specification for
  `oof_a6`, `face`, `side`, and `deterministic`.

- [ ] **Step 1: Write a failing source-matched sensitivity test**

```python
def test_sensitivity_sources_use_centered_cycle_level_mixed_models(tmp_path):
    output = _run_synthetic_analysis(tmp_path)
    sensitivity = pd.read_csv(output / "sensitivity_mixed_models.csv")
    assert set(sensitivity["source"]) == {"oof_a6", "face"}
    assert set(sensitivity["cycle_reference"]) == {0.5}
    assert set(sensitivity["estimand"]) == {
        "mixed_model_mid_repetition_cohort_effect"
    }
```

- [ ] **Step 2: Run the test and confirm failure**

```bash
conda run -n gymnastic python -m pytest \
  tests/cohort_cycle/test_statistics.py \
  -k source_matched_sensitivity -q
```

- [ ] **Step 3: Fit the same model to each source**

For every source and outcome, load `cycle_features.csv`, filter eligible cycles,
call `fit_mixed_effect(..., cycle_reference=0.5)`, and apply Holm correction
within source across the same eight outcomes. Preserve the existing
person-median sensitivity as `sensitivity_person_medians.csv`; do not mix the
two estimands in one table.

- [ ] **Step 4: Record source exclusions explicitly**

If a source lacks cycle-level features, write one row to
`sensitivity_exclusions.csv` with `source`, `reason`, and missing artifact
instead of silently falling back to person medians.

- [ ] **Step 5: Run statistics and CLI tests**

```bash
conda run -n gymnastic python -m pytest \
  tests/cohort_cycle/test_statistics.py \
  tests/cohort_cycle/test_cli.py -q
```

- [ ] **Step 6: Commit source-matched sensitivity**

```bash
git add \
  src/gymnastics/analysis/cohort_cycle/statistics.py \
  src/gymnastics/analysis/cohort_cycle/cli.py \
  tests/cohort_cycle/test_statistics.py \
  tests/cohort_cycle/test_cli.py
git commit -m "feat: align cohort sensitivity estimands"
```

---

### Task 4: Correct Phase Multiplicity and Redesign the Cohort Figure

**Files:**
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/src/gymnastics/analysis/cohort_cycle/statistics.py`
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/src/gymnastics/analysis/cohort_cycle/report.py`
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/tests/cohort_cycle/test_statistics.py`
- Modify:
  `.worktrees/oof-cohort-cycle-analysis/tests/cohort_cycle/test_report.py`

**Interfaces:**
- Consumes:
  centered primary models, source-matched sensitivity models, variability
  results, and phase clusters.
- Produces:
  phase-family-adjusted cluster results, revised LaTeX table, and a four-panel
  PDF without a visually unsupported repetition trend.

- [ ] **Step 1: Add a failing phase-family correction test**

```python
def test_phase_clusters_are_corrected_across_descriptor_families():
    rows = _run_analysis_with_phase_effects()
    assert "p_holm_across_metrics" in rows.columns
    assert rows["p_holm_across_metrics"].ge(rows["p_value"]).all()
```

- [ ] **Step 2: Add failing report tests for the new panels**

```python
def test_report_uses_standardized_effects_and_source_sensitivity():
    report = render_report(...)
    assert report["figure_panels"] == 4
    assert report["panel_c"] == "source_matched_sensitivity"
```

Also assert that the generated table caption says “mid-repetition reference”
and that confidence intervals retain four decimal places when rounding to three
would collapse a nonzero bound to `0.000`.

- [ ] **Step 3: Run the targeted tests and confirm failure**

```bash
conda run -n gymnastic python -m pytest \
  tests/cohort_cycle/test_statistics.py \
  tests/cohort_cycle/test_report.py -q
```

- [ ] **Step 4: Implement cluster-family Holm correction**

For each metric, define its minimum cluster p-value, apply Holm across the four
metric families, and attach the adjusted family p-value to every cluster from
that metric. The manuscript may call a cluster significant only when the
family-adjusted value is below 0.05.

- [ ] **Step 5: Replace the unsupported repetition panel**

Render:

1. standardized centered cohort effects;
2. within-person MAD differences;
3. centered mixed-model sensitivity for angular speed and jerk across pose
   sources;
4. phase-normalized axial rotation with adjusted cluster annotations.

Use the raw coefficients and units in the LaTeX table; use standardized effects
only in Panel A and label them as standardized model-scale coefficients.

- [ ] **Step 6: Run all cohort-cycle tests**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle -q
conda run -n gymnastic ruff check \
  src/gymnastics/analysis/cohort_cycle tests/cohort_cycle
```

- [ ] **Step 7: Commit the corrected report generator**

```bash
git add \
  src/gymnastics/analysis/cohort_cycle/statistics.py \
  src/gymnastics/analysis/cohort_cycle/report.py \
  tests/cohort_cycle/test_statistics.py \
  tests/cohort_cycle/test_report.py
git commit -m "fix: align cohort figures with inferential evidence"
```

---

### Task 5: Regenerate Cohort and Manuscript Evidence Assets

**Files:**
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/scripts/generate_paper_assets.py`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/scripts/test_check_manuscript.py`
- Replace generated:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/tables/learned_results.tex`
- Create generated:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/tables/unity_benchmark.tex`
- Replace generated:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/tables/cohort_cycle_results.tex`
- Replace generated:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/tables/cohort_cycle_sensitivity.tex`
- Replace generated:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/figures/cohort_cycle_analysis.pdf`

**Interfaces:**
- Consumes:
  finalized CSV/JSON outputs from Tasks 1--4.
- Produces:
  immutable publication tables, figure, and copied artifact snapshots with
  source hashes.

- [ ] **Step 1: Add failing table-generation tests**

```python
def test_learned_table_uses_held_out_test_rows():
    table = generate_learned_table(...)
    assert "held-out test" in table
    assert "$N=14$" in table
    assert "60.78" in table

def test_unity_table_preserves_input_regime_boundaries():
    table = generate_unity_table(...)
    assert "Unity native 3D" in table
    assert "A6" in table
    assert "178.506" in table
    assert "triangulation" in table
```

- [ ] **Step 2: Run paper script tests and confirm failure**

```bash
conda run -n gymnastic python -m pytest \
  scripts/test_check_manuscript.py -q
```

Run from the manuscript directory.

- [ ] **Step 3: Implement held-out and Unity asset generation**

Read:

- `local/runs/analysis/project_results/learned_results_by_split.csv`;
- `local/runs/analysis/project_results/learned_test_comparisons.csv`;
- `local/runs/unity_benchmark/report/results.json`;
- `local/runs/unity_benchmark/extrinsic_learning/report/results.json`.

Fail if required populations, methods, sequences, folds, or seeds are missing.

- [ ] **Step 4: Re-run finalized cohort analysis**

From the cohort-analysis worktree, use explicit new output roots and force the
worktree's `src` directory onto the module path:

```bash
PYTHONPATH=src conda run -n gymnastic python \
  -m gymnastics.analysis.cohort_cycle.cli analyze \
  --config configs/analysis/cohort_cycle.yaml \
  --feature-root /home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/features \
  --output-root /home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/statistics_midcycle \
  --sensitivity-feature face=/home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/features_face \
  --sensitivity-feature side=/home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/features_side \
  --sensitivity-feature deterministic=/home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/features_deterministic

PYTHONPATH=src conda run -n gymnastic python \
  -m gymnastics.analysis.cohort_cycle.cli assets \
  --config configs/analysis/cohort_cycle.yaml \
  --feature-root /home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/features \
  --statistics-root /home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/statistics_midcycle \
  --output-root /home/workspace/kaixu/code/Gymnastics_PyTorch/local/runs/cohort_cycle/analysis/report_midcycle
```

- [ ] **Step 5: Generate manuscript assets**

```bash
conda run -n gymnastic python scripts/generate_paper_assets.py
```

- [ ] **Step 6: Run table and hash tests**

```bash
conda run -n gymnastic python -m pytest \
  scripts/test_check_manuscript.py -q
```

Expected: generated table values match their archived inputs.

---

### Task 6: Revise the Manuscript Around the Corrected Evidence

**Files:**
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/manuscript.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/01_introduction.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/02_related_work.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/04_method.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/05_experimental_protocol.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/06_results.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/06b_cohort_cycle_analysis.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/07_discussion.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/08_limitations.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/09_conclusion.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/sections/declarations.tex`
- Modify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/SUBMISSION_CHECKLIST.md`

**Interfaces:**
- Consumes:
  generated assets and evidence map from Tasks 1--5.
- Produces:
  a claim-evidence-consistent LaTeX manuscript.

- [ ] **Step 1: Revise the abstract and contribution claims**

State:

- private held-out A6 MPJPE is statistically indistinguishable from A2 and A5;
- A6 remains the constrained full objective, not an accuracy winner;
- Unity native-3D zero-shot transfer does not improve over direct fusion;
- the cohort application is exploratory and pose-source-sensitive.

- [ ] **Step 2: Correct the learned experimental protocol**

Separate:

- training: 96 people;
- validation/checkpoint selection and corruption diagnostics: 27 people;
- primary generalization: 14 people;
- all-person inference: 137-person descriptive diagnostic;
- OOF A6 cohort analysis: 10 person-disjoint outer folds.

- [ ] **Step 3: Add the Unity protocol and results**

Describe the one-avatar limitation, the 199 samples, the three sequences, the
Unity16 joint mapping, sequence-level Sim3 evaluation, and the input-regime
separation. Report the negative A6 transfer result before discussing calibrated
comparators.

- [ ] **Step 4: Revise the cohort section**

Interpret the cohort coefficient at normalized cycle position 0.5. Compare the
with-fold and no-fold model estimates, then report source-matched mixed-model
sensitivity. Keep person-median sensitivity in a separate paragraph if it
remains informative.

- [ ] **Step 5: Correct equations and precision**

Replace:

```latex
\theta_t^v=\operatorname{atan2}(\cdots)
\qquad
\Delta\mathbf{x}_{t,j}=\delta_j^{\max}\tanh(\mathbf{h}_{t,j})
```

Increase table precision where three decimals conceal a nonzero confidence
bound.

- [ ] **Step 6: Revise discussion, limitations, and conclusion**

The central conclusion is that canonicalization accounts for most private-data
improvement. A6 supplies an auditable constrained extension but no demonstrated
positional or external-transfer advantage. Unity shows the boundary of
uncalibrated direct-3D fusion and the benefit of geometry when calibration and
2D evidence are available.

- [ ] **Step 7: Preserve governance blockers**

Keep ethics approval/exemption, participant consent, institution address, and
data-release status visibly unresolved until the author supplies verified
wording. Do not leave a result placeholder elsewhere in the manuscript.

- [ ] **Step 8: Run manuscript checks**

```bash
conda run -n gymnastic python scripts/check_manuscript.py
conda run -n gymnastic python -m pytest scripts/test_check_manuscript.py -q
```

Expected: no result placeholder, population mismatch, numerical contradiction,
or citation orphan.

---

### Task 7: Compile and Visually Verify the Final PDF

**Files:**
- Verify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/build/manuscript.pdf`
- Verify:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/build/manuscript.log`
- Modify only if verification fails:
  the exact LaTeX source, table, figure, or test responsible.

**Interfaces:**
- Consumes:
  the revised manuscript package.
- Produces:
  a compiled PDF and final verification report.

- [ ] **Step 1: Run the complete relevant test suites**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle -q
conda run -n gymnastic python -m pytest tests/test_project_results.py -q
conda run -n gymnastic python -m pytest tests/unity_benchmark -q
```

- [ ] **Step 2: Run static checks**

```bash
conda run -n gymnastic ruff check \
  src/gymnastics/analysis/cohort_cycle \
  src/gymnastics/analysis/project_results.py \
  tests/cohort_cycle tests/test_project_results.py
```

- [ ] **Step 3: Compile from a clean manuscript build**

```bash
make clean
make
```

Run from:
`.worktrees/rotation-aware-fusion/paper/neurocomputing`.

- [ ] **Step 4: Inspect the LaTeX log**

```bash
rg -n "Undefined|Citation.*undefined|Reference.*undefined|Overfull|Underfull|Warning" \
  build/manuscript.log
```

Resolve substantive warnings. A known harmless title-page overfull box may
remain only after visual confirmation.

- [ ] **Step 5: Render and inspect critical pages**

Render the abstract, method equations, primary learned table, Unity table,
cohort figure, limitations, and declarations. Confirm:

- formulas show `atan2` and Greek delta correctly;
- tables fit without illegible scaling;
- Panel A names standardized effects;
- Panel C names source sensitivity;
- ethics text remains visible;
- no stale first-cycle or A7-superiority claim remains.

- [ ] **Step 6: Run the final claim-evidence scan**

```bash
rg -n "137-participant set|first repetition|A7 is|improve position|ground truth|clinical|causal" \
  manuscript.tex sections tables
```

Every match must be either corrected evidence, a labeled limitation, or a
negative statement.

- [ ] **Step 7: Record completion**

Report the final PDF path, test counts, remaining submission blockers, and all
files changed. Do not call the manuscript submission-ready while ethics,
consent, or institution details are unresolved.
