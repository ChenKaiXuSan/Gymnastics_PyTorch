# Results Reporting Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible project results summary and make the manuscript's conclusions match the actual train/validation/test evidence.

**Architecture:** A focused analysis module reads the learned-fusion split manifest, per-person metrics, and classification fold metrics. It emits validated CSV/Markdown artefacts, while tracked documentation and the ignored local manuscript describe the same evidence hierarchy without fabricating unfinished experiments.

**Tech Stack:** Python 3.10, NumPy, SciPy, CSV/JSON, pytest, LaTeX.

## Global Constraints

- Run project Python and tests with `conda run -n gymnastic`.
- Treat the 14-person test split as primary learned-model generalization evidence.
- Label the 137-person learned result as descriptive and the 27-person corruption result as validation-only.
- Keep missing robustness, multi-seed evaluation, and external benchmark results pending.
- Do not commit or push this change without a separate request.

---

### Task 1: Tested Result Aggregation Core

**Files:**
- Create: `tests/test_project_results.py`
- Create: `src/gymnastics/analysis/project_results.py`

**Interfaces:**
- Produces: `load_split_manifest(path) -> dict[str, set[str]]`
- Produces: `summarize_learned_by_split(rows, splits) -> list[dict[str, object]]`
- Produces: `holm_adjust(p_values) -> list[float]`
- Produces: `paired_comparisons(rows, reference_method, metric, seed) -> list[dict[str, object]]`
- Produces: `summarize_classification(metric_paths) -> list[dict[str, object]]`

- [ ] **Step 1: Write failing tests for split-aware aggregation**

Create fixtures containing train, validation, and test people, with corruption
values present only for validation. Assert that means and measured counts remain
separate and that unavailable observations are not counted.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_project_results.py -q
```

Expected: collection fails because `gymnastics.analysis.project_results` does
not exist.

- [ ] **Step 3: Implement split validation and aggregation**

Reject duplicate people across splits, reject missing or extra metric people,
parse finite values only, and return explicit `n_people` and `n_measured`.

- [ ] **Step 4: Add failing tests for Holm and paired statistics**

Assert monotone Holm-adjusted p-values in ranked p-value order, paired sample
counts, mean differences, and deterministic bootstrap intervals for a fixed
seed.

- [ ] **Step 5: Implement Holm and paired statistics**

Use `scipy.stats.wilcoxon` for paired tests and percentile bootstrap resampling
of person-level mean differences. Preserve the input comparison order.

- [ ] **Step 6: Add failing tests for classification aggregation**

Create three fold files for one run and assert model/target parsing, mean,
sample standard deviation, and fold count for each `test/acc_*` and `test/f1_*`
metric.

- [ ] **Step 7: Implement classification aggregation**

Parse the run name from the directory above the date segment, load the single
JSON object from each `fold_*_test_metrics.txt`, and aggregate matching runs over
their available folds.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run the command from Step 2 and require all tests to pass.

### Task 2: Real-Data Generator And Project Summary

**Files:**
- Modify: `src/gymnastics/analysis/project_results.py`
- Create: `docs/results_summary.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the pure aggregation functions from Task 1.
- Produces: `generate_project_results(...) -> dict[str, Path]`
- Produces: `python -m gymnastics.analysis.project_results`

- [ ] **Step 1: Add a failing end-to-end generator test**

Build a temporary learned CSV, split manifest, and classification directory.
Assert the four expected outputs are written and the Markdown labels test,
descriptive-all, and validation-only evidence explicitly.

- [ ] **Step 2: Run the end-to-end test and verify RED**

Run only the new generator test and confirm failure because the public generator
does not yet exist.

- [ ] **Step 3: Implement the generator and CLI**

Write CSV atomically through temporary sibling files, produce a concise Markdown
report, expose path overrides for testing, and default to the current local
experiment paths.

- [ ] **Step 4: Run the generator on current artefacts**

Run:

```bash
conda run -n gymnastic python -m gymnastics.analysis.project_results
```

Require a 96/27/14 split, 137 unique learned-result people, and three folds per
complete classification run.

- [ ] **Step 5: Add the tracked documentation entry points**

Write `docs/results_summary.md` with the validated headline values, evidence
boundaries, failure coverage, and pending work. Add a short Results link in
`README.md`.

- [ ] **Step 6: Re-run focused tests**

Run all `tests/test_project_results.py` tests.

### Task 3: Manuscript Evidence Correction

**Files:**
- Modify: `paper/image_and_vision_computing/sections/05_experimental_protocol.tex`
- Modify: `paper/image_and_vision_computing/sections/06_results.tex`
- Modify: `paper/image_and_vision_computing/sections/07_discussion.tex`
- Modify: `paper/image_and_vision_computing/sections/08_conclusion.tex`
- Modify: `paper/image_and_vision_computing/tables/learned_results.tex`
- Modify: `paper/image_and_vision_computing/artifacts/source_audit.md`
- Modify: `paper/image_and_vision_computing/SUBMISSION_CHECKLIST.md`

**Interfaces:**
- Consumes: generated learned split and paired-comparison CSV files.
- Produces: a paper whose claims identify held-out, descriptive, and
  validation-only cohorts.

- [ ] **Step 1: Record exact current claims**

Locate every learned-result, corruption, seed-count, 100-epoch, Holm, and
robustness statement before patching.

- [ ] **Step 2: Correct protocol and learned-results table**

Disclose 96/27/14 person-level splits, make test `N=14` the primary table,
move fixed-corruption recovery to validation `N=27`, and label all-person
numbers descriptive. State that A9 stopped at epoch 85 with its best checkpoint
near epoch 83.

- [ ] **Step 3: Correct Results, Discussion, and Conclusion**

Replace all-cohort generalization wording with held-out-test wording. Preserve
negative A8/A9 results and explicitly limit conclusions to the current
pseudo-GT, single-seed evidence.

- [ ] **Step 4: Refresh audit and checklist**

Mark only verifiably completed artefacts as complete. Leave paired multi-seed,
offset robustness, and independent external validation unchecked.

- [ ] **Step 5: Build the paper**

Run:

```bash
GYMNASTICS_SOURCE_ROOT=/home/workspace/kaixu/code/Gymnastics_PyTorch make -C paper/image_and_vision_computing
```

Expected: LaTeX build succeeds with no unresolved references introduced by the
revision.

### Task 4: Regression And Evidence Verification

**Files:**
- Modify only if a verification failure identifies an in-scope defect.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: final verification evidence and an explicit remaining-work list.

- [ ] **Step 1: Run focused analysis tests**

```bash
conda run -n gymnastic python -m pytest tests/test_project_results.py tests/test_compare_fused_triangulated.py -q
```

- [ ] **Step 2: Run the full test suite**

```bash
conda run -n gymnastic python -m pytest -q
```

- [ ] **Step 3: Check generated and written claims**

Confirm test `N=14`, validation corruption `N=27`, all-cohort descriptive
`N=137`, and absence of claims that pending experiments are complete.

- [ ] **Step 4: Inspect repository status**

Report tracked changes separately from ignored paper/local artefacts. Do not
commit or push.
