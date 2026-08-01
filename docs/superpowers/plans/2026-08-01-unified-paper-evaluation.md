# Unified Sports Engineering Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate every pseudo-reference paper comparison with the Table 1 similarity-plus-framewise-hip-centering protocol.

**Architecture:** Extend the existing paper table generator with one shared compact-sequence evaluator that emits both person- and joint-level rows. Feed both the 137-person extrinsic table and the 14-person joint table from this evaluator, and reject any protocol mismatch before statistical aggregation.

**Tech Stack:** Python, NumPy, pandas, SciPy, pytest, LaTeX, GNU Make, `gymnastic` conda environment.

## Global Constraints

- Use the fixed 137-person dataset and the fixed 14-person test split.
- Fit one similarity transform per cycle, then perform framewise hip centering.
- Aggregate errors within participant before cross-participant summaries.
- Treat triangulated 3D only as a same-video evaluation pseudo-reference.
- Do not alter training checkpoints or rerun model training.

---

### Task 1: Shared matched evaluator

**Files:**
- Modify: `paper/sports_engineering/scripts/generate_comparison_tables.py`
- Test: `paper/sports_engineering/scripts/test_generate_comparison_tables.py`

**Interfaces:**
- Consumes: matched `(candidate, reference)` cycle arrays and `SkeletonSpec`.
- Produces: `evaluate_matched_metrics(person_id, method, matched_cycles, skeleton) -> tuple[dict, pandas.DataFrame]` with the `similarity_plus_hip_centering` protocol label.

- [ ] Add a test whose candidate differs from reference by frame-varying root translation and assert pooled person MPJPE and all joint MPJPE values are near zero.
- [ ] Run the focused test and confirm it fails because `evaluate_matched_metrics` does not exist.
- [ ] Implement the shared evaluator and make the existing joint-only helper delegate to it.
- [ ] Run the generator test module and confirm all tests pass.

### Task 2: Protocol-checked 137-person summary

**Files:**
- Modify: `paper/sports_engineering/scripts/generate_comparison_tables.py`
- Test: `paper/sports_engineering/scripts/test_generate_comparison_tables.py`

**Interfaces:**
- Consumes: person-level rows for `avg_body_current`, `extrinsic_r_average`, and `extrinsic_r_quality_average`.
- Produces: a Table 2 summary only when every row declares `similarity_plus_hip_centering` and each method covers the same 137 people.

- [ ] Add a test that passes a similarity-only baseline into `build_extrinsic_summary` and expects a protocol-mismatch error.
- [ ] Run the focused test and confirm it fails under the current implementation.
- [ ] Add protocol validation and replace old person-metric CSV inputs with unified compact-sequence reevaluation.
- [ ] Write `extrinsic_person_metrics_matched_137.csv` and include the protocol in `extrinsic_comparison_137.csv`.
- [ ] Run the generator tests and confirm all tests pass.

### Task 3: Regenerate evidence and revise manuscript

**Files:**
- Modify generated files under `paper/sports_engineering/generated/`
- Modify: `paper/sports_engineering/manuscript.tex`
- Modify: `paper/sports_engineering/online_resource_1.tex`
- Modify: `paper/sports_engineering/README.md`
- Modify: `paper/sports_engineering/scripts/check_sports_engineering.py`
- Modify submission artifacts under `paper/sports_engineering/submission/`

**Interfaces:**
- Consumes: unified 137-person summary CSV and existing 14-person learned comparison.
- Produces: consistent Table 1–3 wording, updated statistics, compiled PDFs, and reproducible source archive.

- [ ] Run the table generator once and record all changed means, deltas, confidence intervals, adjusted p-values, improvement counts, and camera-fit correlation.
- [ ] Run the generator a second time and compare output hashes to confirm deterministic regeneration.
- [ ] Replace every obsolete Table 2 number and protocol description in the abstract, Methods, Results, Discussion, checker anchors, and documentation.
- [ ] Compile `make -C paper/sports_engineering package` and inspect both LaTeX logs for errors, unresolved references, and overfull boxes.
- [ ] Run the focused paper/fusion regression suite and confirm zero failures.
- [ ] Commit the unified evaluation, regenerated evidence, and submission package on `codex/extrinsic-joint-paper`.

