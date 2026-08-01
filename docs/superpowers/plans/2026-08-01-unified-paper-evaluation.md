# Unified Sports Engineering Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate every pseudo-reference paper comparison with the Table 1 similarity-plus-framewise-hip-centering protocol and use the same 14 held-out participants in all main comparison tables.

**Architecture:** Keep one shared compact-sequence evaluator that emits person- and joint-level rows for all 137 participants. Derive a 14-person held-out camera summary for main Table 2 and a separate 137-person secondary summary for Online Resource, rejecting protocol or participant-set mismatches before aggregation.

**Tech Stack:** Python, NumPy, pandas, SciPy, pytest, LaTeX, GNU Make, `gymnastic` conda environment.

## Global Constraints

- Use the fixed 137-person dataset and the fixed 14-person test split.
- Use only the fixed 14-person test split for main Tables 1--3.
- Keep complete 137-person deterministic and camera-assisted summaries in Online Resource only.
- Fit one similarity transform per cycle, then perform framewise hip centering.
- Aggregate errors within participant before cross-participant summaries.
- Treat triangulated 3D only as a same-video evaluation pseudo-reference.
- Do not alter training checkpoints or rerun model training.

---

### Task 1: Held-out and all-participant summary contract

**Files:**
- Modify: `paper/sports_engineering/scripts/generate_comparison_tables.py`
- Test: `paper/sports_engineering/scripts/test_generate_comparison_tables.py`

**Interfaces:**
- Consumes: the unified person-level frame and the fixed test-person tuple.
- Produces: `build_extrinsic_summaries(person_metrics, test_people) -> tuple[pandas.DataFrame, pandas.DataFrame]`; the first frame covers exactly the held-out people and the second covers the full people set.

- [x] Add a test with four synthetic participants and a two-person test tuple; assert the main summary has `n=2`, the secondary summary has `n=4`, and an unknown test participant raises `ValueError`.
- [x] Run the focused test and confirm it fails because `build_extrinsic_summaries` does not exist.
- [x] Implement exact test-person selection before calling the existing protocol-checked paired summary.
- [x] Run the focused test and confirm it passes.
- [x] Add a rendering test proving the improved-person denominator comes from summary `n`, not a hard-coded `/137`.
- [x] Run the rendering test and confirm it fails on the current hard-coded denominator.
- [x] Parameterize the renderer with population text and table label, then rerun the tests.

### Task 2: Generate separate main and supplementary evidence

**Files:**
- Modify: `paper/sports_engineering/scripts/generate_comparison_tables.py`
- Modify generated files under `paper/sports_engineering/generated/`
- Test: `paper/sports_engineering/scripts/test_generate_comparison_tables.py`

**Interfaces:**
- Consumes: `pseudo_reference_person_metrics_matched_137.csv` and the fixed split.
- Produces: `extrinsic_comparison_test14.csv`, `extrinsic_comparison_137.csv`, `extrinsic_comparison.tex`, and `extrinsic_comparison_all137.tex`.

- [x] Wire `main()` to build both summaries from the same unified person frame.
- [x] Render main Table 2 with held-out wording and Online Resource with secondary all-participant wording.
- [x] Regenerate once and record means, deltas, confidence intervals, adjusted p-values, and improvement counts for both populations.
- [x] Regenerate a second time and compare hashes for deterministic output.

### Task 3: Revise and verify the manuscript package

**Files:**
- Modify: `paper/sports_engineering/manuscript.tex`
- Modify: `paper/sports_engineering/online_resource_1.tex`
- Modify: `paper/sports_engineering/README.md`
- Modify: `paper/sports_engineering/SUBMISSION_CHECKLIST.md`
- Modify: `paper/sports_engineering/scripts/check_sports_engineering.py`
- Modify: `docs/research/extrinsic_fusion_results_2026-07-29.md`

**Interfaces:**
- Consumes: unified 14-person main summary, unified 137-person secondary summary, and the existing 14-person learned comparison.
- Produces: main Tables 1--3 with one held-out population, an explicitly secondary all-participant Online Resource table, updated statistics, and compiled PDFs.

- [x] Replace every 137-person main-camera claim in the abstract, Methods, Results, Discussion and checker anchors with the held-out result.
- [x] Add the 137-person camera table and calibration association to Online Resource with explicit secondary-analysis wording.
- [x] Run the 11-item statistical fallacy scan and keep shared-evidence, small held-out sample, multiplicity and descriptive all-participant boundaries explicit.
- [x] Run `conda run -n gymnastic make -C paper/sports_engineering all` and inspect both PDFs for ordering and overflow.
- [x] Run `conda run -n gymnastic python -m pytest -q` and confirm zero failures.
- [x] Commit the cohort-harmonized tables and manuscript on `codex/extrinsic-joint-paper`.
