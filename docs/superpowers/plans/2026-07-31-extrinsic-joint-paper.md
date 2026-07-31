# Extrinsic Comparators and Per-Joint Paper Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add camera-extrinsic deterministic comparators and test-set per-joint accuracy tables to the Sports Engineering manuscript with reproducible, source-checked generation.

**Architecture:** A standalone Sports Engineering table generator joins the fixed 14-person split with existing learned, deterministic, and extrinsic per-person/per-joint CSVs. It validates complete person-method-joint coverage, performs participant-level summaries, and emits auditable CSV plus LaTeX fragments consumed by the main article and Online Resource.

**Tech Stack:** Python 3.10, pandas, NumPy, SciPy, pytest, LaTeX (`sn-jnl`, `booktabs`, `longtable`).

## Global Constraints

- Run project Python and tests through `conda run -n gymnastic`.
- Do not retrain any model or overwrite `local/runs` artifacts.
- Keep A6 as the calibration-free mainline and label extrinsic methods as camera-assisted comparators.
- Use all 137 people only for parameter-free extrinsic comparisons; use exactly the fixed 14 test people for learned per-joint comparisons.
- Aggregate joint errors within person before averaging across people; convert repository coordinate units to millimetres by multiplying by 1000.
- Treat the triangulated reference as same-video pseudo-reference, not independent motion-capture ground truth.

---

### Task 1: Source-checked comparison table generator

**Files:**
- Create: `paper/sports_engineering/scripts/generate_comparison_tables.py`
- Create: `paper/sports_engineering/scripts/test_generate_comparison_tables.py`

**Interfaces:**
- Consumes: fixed split JSON; learned and extrinsic `metrics_by_joint.csv`; deterministic and extrinsic `metrics_by_person.csv`; canonical `MHR70_NAMES` and `MAJOR_JOINT_INDICES`.
- Produces: `load_test_people(path)`, `load_joint_metrics(path, methods)`, `build_joint_summary(...)`, `build_extrinsic_summary(...)`, `render_main_joint_table(...)`, `render_all_joint_table(...)`, `render_extrinsic_table(...)`, and a CLI `main()`.

- [ ] **Step 1: Write failing tests for split and participant-first aggregation**

Use literal two-person fixtures whose frame/point counts differ but whose per-person MPJPE means are 10 and 30 mm. Assert that `build_joint_summary` returns 20 mm, proving the implementation does not point-weight people. Assert that a split with 13 or 15 test people raises `ValueError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest paper/sports_engineering/scripts/test_generate_comparison_tables.py -q
```

Expected: collection fails because `generate_comparison_tables.py` does not exist.

- [ ] **Step 3: Implement validated loaders and joint aggregation**

Implement these exact contracts:

```python
def load_test_people(path: Path) -> tuple[str, ...]: ...
def load_joint_metrics(path: Path, methods: tuple[str, ...]) -> pd.DataFrame: ...
def build_joint_summary(
    learned: pd.DataFrame,
    extrinsic: pd.DataFrame,
    test_people: tuple[str, ...],
) -> pd.DataFrame: ...
```

Normalize `person_id` to string and `joint` to integer. Reject duplicate `(person_id, method, joint)` rows, non-finite MPJPE, missing requested methods, non-70-joint coverage, and any test-person mismatch. Output one row per canonical joint with columns `joint`, `joint_name`, `A0`, `A1`, `A2`, `A6`, `extrinsic_r_average`, and `extrinsic_r_quality_average`, all in mm.

- [ ] **Step 4: Verify GREEN for aggregation tests**

Run the focused pytest command and expect all aggregation tests to pass.

- [ ] **Step 5: Write failing tests for extrinsic paired statistics and LaTeX**

Create four-person literal fixtures and independently hand-calculate the method-minus-baseline mean difference and improved-person count. Assert that the summary includes `mean_mm`, `std_mm`, `delta_mm`, `ci_low_mm`, `ci_high_mm`, `p_holm`, and `improved_people`. Assert that main-table LaTeX contains exactly 20 joint rows, bolds only the row minimum, and escapes joint labels; assert that all-joint LaTeX contains 70 rows and the quality-weighted method.

- [ ] **Step 6: Run tests and verify RED for missing statistics/renderers**

Run the focused pytest command. Expected: failures name `build_extrinsic_summary` and rendering functions as missing.

- [ ] **Step 7: Implement paired summaries, Holm correction, renderers, and CLI**

Use a fixed-seed 10,000-resample participant bootstrap on paired differences, two-sided `scipy.stats.wilcoxon`, and Holm correction across the two planned extrinsic-vs-body comparisons. The CLI writes:

```text
paper/sports_engineering/generated/extrinsic_comparison_137.csv
paper/sports_engineering/generated/joint_accuracy_test14.csv
paper/sports_engineering/generated/extrinsic_comparison.tex
paper/sports_engineering/generated/joint_accuracy_main.tex
paper/sports_engineering/generated/joint_accuracy_all70.tex
```

The main joint table uses `MAJOR_JOINT_INDICES`; the supplementary longtable uses `MHR70_NAMES` in exact order.

- [ ] **Step 8: Run focused tests and the existing fusion-matrix tests**

Run:

```bash
conda run -n gymnastic python -m pytest \
  paper/sports_engineering/scripts/test_generate_comparison_tables.py \
  tests/test_fuse_experiment_matrix.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add paper/sports_engineering/scripts/generate_comparison_tables.py \
  paper/sports_engineering/scripts/test_generate_comparison_tables.py
git commit -m "feat: generate extrinsic and joint paper tables"
```

### Task 2: Generate and validate experiment artifacts

**Files:**
- Create: `paper/sports_engineering/generated/extrinsic_comparison_137.csv`
- Create: `paper/sports_engineering/generated/joint_accuracy_test14.csv`
- Create: `paper/sports_engineering/generated/extrinsic_comparison.tex`
- Create: `paper/sports_engineering/generated/joint_accuracy_main.tex`
- Create: `paper/sports_engineering/generated/joint_accuracy_all70.tex`

**Interfaces:**
- Consumes: Task 1 CLI and immutable local experiment artifacts.
- Produces: versioned paper-ready numeric evidence and LaTeX fragments.

- [ ] **Step 1: Run the generator against the formal 137-person artifacts**

```bash
conda run -n gymnastic python paper/sports_engineering/scripts/generate_comparison_tables.py
```

Expected provenance: 137 people for the extrinsic matrix; exactly the split IDs `1, 106, 116, 117, 130, 136, 24, 36, 49, 51, 52, 60, 79, 85` for joint analysis; 70 joints for every requested method.

- [ ] **Step 2: Cross-check generated aggregate values**

Assert from the generated CSV that body-frame average is 64.045 mm, Extrinsic-R average is 62.031 mm, Extrinsic-R quality average is 63.251 mm within 0.001 mm, and the preferred equal-weight method improves 118/137 people.

- [ ] **Step 3: Run the 11-item statistical fallacy scan**

Check unit-of-analysis, pseudoreplication, multiple comparisons, confidence-interval interpretation, effect-size/practical-size separation, shared-evidence bias, subgroup imbalance, post hoc joint selection, causal language, missing-data handling, and descriptive-vs-held-out labeling. Record cautions in manuscript prose rather than altering results.

- [ ] **Step 4: Commit generated evidence**

```bash
git add paper/sports_engineering/generated
git commit -m "results: add extrinsic and per-joint paper tables"
```

### Task 3: Integrate tables into the Sports Engineering manuscript

**Files:**
- Modify: `paper/sports_engineering/manuscript.tex`
- Modify: `paper/sports_engineering/online_resource_1.tex`
- Modify: `paper/sports_engineering/Makefile`
- Modify: `paper/sports_engineering/README.md`
- Modify: `paper/sports_engineering/scripts/check_sports_engineering.py`

**Interfaces:**
- Consumes: Task 2 LaTeX fragments and validated evidence.
- Produces: compiled main manuscript, Online Resource, and submission archive containing generated table sources.

- [ ] **Step 1: Write a failing manuscript integration check**

Extend `check_sports_engineering.py` to require labels `tab:extrinsic-comparison`, `tab:joint-accuracy-main`, and `tab:joint-accuracy-all70`; require the explicit phrases `camera-assisted comparator` and `same-video evidence`; and require the source archive list to include `generated/*.tex`. Run it and verify failure before editing LaTeX.

- [ ] **Step 2: Modify Methods and Results**

Add the estimated-extrinsic rotation convention and transductive/shared-evidence boundary to Evaluation hierarchy. Add a Results subsection that inputs `generated/extrinsic_comparison.tex` and `generated/joint_accuracy_main.tex`, reports the 3.15%/118-of-137 improvement, and describes which joint classes improve without uncorrected per-joint significance claims.

- [ ] **Step 3: Modify Discussion, Limitations, abstract, and article counts**

State that camera rotation is a useful calibration-assisted comparator but does not replace A6 or establish absolute accuracy. Mention the camera-fit quality dependence and the G0--G5 negative result. Update the abstract only with the camera-assisted secondary result and update the declared table counts.

- [ ] **Step 4: Add the complete 70-joint table to Online Resource**

Load `longtable`, input `generated/joint_accuracy_all70.tex`, and add a short interpretation that the table is descriptive and uses the same 14 held-out people. Keep G0--G5 as a negative camera-feature result.

- [ ] **Step 5: Update build and packaging**

Make table generation a prerequisite of `make all`; include generated CSV/TEX files in `submission/sports_engineering_source.zip`; document the generator and source artifacts in README.

- [ ] **Step 6: Run manuscript checks and both LaTeX builds**

```bash
conda run -n gymnastic python paper/sports_engineering/scripts/check_sports_engineering.py
make -C paper/sports_engineering clean all
```

Expected: check exit 0; manuscript and Online Resource PDFs compile; no undefined references or LaTeX errors.

- [ ] **Step 7: Run final regression suite**

```bash
conda run -n gymnastic python -m pytest \
  paper/sports_engineering/scripts/test_generate_comparison_tables.py \
  tests/test_fuse_experiment_matrix.py \
  tests/rotation_aware/test_real_camera_evaluation.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit manuscript integration**

```bash
git add paper/sports_engineering
git commit -m "paper: add extrinsic and per-joint comparisons"
```
