# Out-of-Fold Cohort and Repeated-Cycle Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce leakage-controlled A6 out-of-fold poses for all 137 participants, quantify elderly-cohort versus student-cohort motion and repeated-cycle differences, and integrate only generated, audited results into the Neurocomputing manuscript.

**Architecture:** Add a self-contained `gymnastics.analysis.cohort_cycle` package and a `gymnastics cohort-cycle` CLI. Existing rotation-aware training and inference remain the only model execution path; the new package generates and audits folds, merges test-only publications, extracts prespecified features, runs statistics, and renders paper assets. Every stage consumes immutable machine-readable artifacts from the preceding stage, and paper prose never computes results.

**Tech Stack:** Python 3.10+, NumPy, pandas, SciPy, statsmodels, matplotlib, PyYAML, PyTorch through the existing A6 pipeline, pytest, LaTeX/latexmk.

## Global Constraints

- Run project Python, tests, inference, and training through `conda run -n gymnastic`.
- Preserve the existing fold-0 test set exactly and reuse its three A6 checkpoints only after hash/provenance validation.
- Use seed 0 for the primary 10-fold OOF publication; fold-0 seeds 1 and 2 are sensitivity outputs only.
- Never use cohort labels in A6 inputs, losses, validation, checkpoint selection, or feature selection.
- Never read triangulated pseudo-reference data during cross-fit training, OOF publication, or cohort feature extraction.
- Treat `elderly` and `student` as cohort labels, not measured ages or causal ageing exposure.
- Do not write result directions, significance claims, abstract claims, or highlights until finalized result tables exist.
- Keep generated experiment outputs under `local/runs/cohort_cycle/`; keep only configurations, code, tests, and paper source under version control.
- Do not overwrite the user's unrelated dirty worktree changes.

---

## Task 1: Add the isolated CLI and analysis dependency

**Files:**

- Modify: `pyproject.toml`
- Modify: `src/gymnastics/cli.py`
- Create: `src/gymnastics/analysis/cohort_cycle/__init__.py`
- Create: `src/gymnastics/analysis/cohort_cycle/cli.py`
- Test: `tests/cohort_cycle/test_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
def test_top_level_cli_routes_cohort_cycle(monkeypatch):
    assert gymnastics.cli.main(["cohort-cycle", "--help"]) == 0


def test_cohort_cycle_parser_exposes_pipeline_stages():
    parser = make_parser()
    for command in ("folds", "audit", "features", "analyze", "assets"):
        assert parser.parse_args([command, "--config", "x.yaml"]).command == command
```

- [ ] **Step 2: Run the tests and confirm the command is absent**

Run:

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_cli.py -q
```

Expected: failure because `cohort-cycle` and its parser do not exist.

- [ ] **Step 3: Add `statsmodels>=0.14` to the `analysis` optional dependency**

Keep training dependencies unchanged. Install the updated analysis and test extras before statistical tests:

```bash
conda run -n gymnastic python -m pip install -e '.[analysis,test]'
```

- [ ] **Step 4: Implement the thin command router**

Add:

```python
"cohort-cycle": ("gymnastics.analysis.cohort_cycle.cli", "main", True)
```

The new parser exposes `folds`, `audit`, `features`, `analyze`, and `assets`. Each handler accepts a resolved YAML configuration and delegates to a focused module; `cli.py` contains no numerical analysis.

- [ ] **Step 5: Run focused and top-level CLI tests**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_cli.py tests/structure/test_cli.py -q
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/gymnastics/cli.py src/gymnastics/analysis/cohort_cycle tests/cohort_cycle/test_cli.py
git commit -m "feat: add cohort cycle analysis CLI"
```

## Task 2: Generate deterministic cohort-stratified outer folds

**Files:**

- Create: `configs/analysis/cohort_cycle.yaml`
- Create: `configs/analysis/cohort_cycle_a6_train.yaml`
- Create: `src/gymnastics/analysis/cohort_cycle/config.py`
- Create: `src/gymnastics/analysis/cohort_cycle/cohorts.py`
- Create: `src/gymnastics/analysis/cohort_cycle/folds.py`
- Test: `tests/cohort_cycle/test_cohorts.py`
- Test: `tests/cohort_cycle/test_folds.py`

- [ ] **Step 1: Write cohort mapping and fold invariant tests**

Cover:

```python
assert counts == {"elderly": 80, "student": 57}
assert fold_sizes == [14, 14, 14, 14, 14, 14, 14, 13, 13, 13]
assert all(counts_by_fold[i]["elderly"] == 8 for i in range(10))
assert manifest["folds"]["00"]["test"] == EXISTING_FOLD0_TEST
assert set(train).isdisjoint(val | test)
assert set.union(*test_sets) == all_137_people
```

Also test that ID 135 is rejected, mapping-file hashes are stable, validation has 27 people, and a second run produces byte-identical JSON.

- [ ] **Step 2: Confirm the tests fail**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_cohorts.py tests/cohort_cycle/test_folds.py -q
```

- [ ] **Step 3: Implement authoritative cohort loading**

`cohorts.py` reads both source CSVs, normalizes IDs to canonical numeric strings, verifies their agreement, excludes absent S55/ID135, and returns:

```python
@dataclass(frozen=True)
class CohortRecord:
    person_id: str
    cohort: Literal["elderly", "student"]
```

The generated manifest stores SHA-256 hashes of both mapping inputs.

- [ ] **Step 4: Implement deterministic 10-fold construction**

Use a fixed `split_seed` from the YAML. Fold 0 must copy the exact train/validation/test lists from:

`local/runs/fuse_rotation_aware/runs/all137_a6_e100_seed0/split_manifest.json`

For folds 1--9, stratify the remaining test people by cohort and deterministically choose 27 validation people from the non-test pool. Write:

- `local/runs/cohort_cycle/folds/fold_00.json` through `fold_09.json`, in the existing rotation-aware `{train, val, test}` schema;
- `local/runs/cohort_cycle/folds/crossfit_manifest.json`, containing cohorts, fold membership, source hashes, split seed, and per-fold JSON hashes;
- `local/runs/cohort_cycle/run_registry.json`, mapping fold 0 to `all137_a6_e100_seed0` and folds 1--9 to `cohort_oof_fXX_a6_e100_s0`.

- [ ] **Step 5: Add the configuration**

The YAML defines source mappings, fold-0 manifest/checkpoints, output roots, A6 config, run-ID template, QC thresholds, core metrics, permutation/bootstrap counts, and random seeds. Paths may be overridden by environment variables but resolve to current `local/runs` locations by default.

- [ ] **Step 6: Run tests and generate the real fold artifacts**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_cohorts.py tests/cohort_cycle/test_folds.py -q
conda run -n gymnastic gymnastics cohort-cycle folds --config configs/analysis/cohort_cycle.yaml
```

Inspect:

```bash
conda run -n gymnastic python -m json.tool local/runs/cohort_cycle/folds/crossfit_manifest.json
```

- [ ] **Step 7: Commit**

```bash
git add configs/analysis/cohort_cycle.yaml src/gymnastics/analysis/cohort_cycle tests/cohort_cycle
git commit -m "feat: generate cohort stratified OOF folds"
```

## Task 3: Audit checkpoint reuse and publish OOF inference safely

**Files:**

- Create: `src/gymnastics/analysis/cohort_cycle/oof.py`
- Test: `tests/cohort_cycle/test_oof.py`
- Modify: `src/gymnastics/analysis/cohort_cycle/cli.py`

- [ ] **Step 1: Write failing provenance tests**

Synthetic registries must reject:

- a published person absent from the run's test split;
- duplicate person/cycle publications;
- missing people or cycles;
- checkpoint split-hash mismatch;
- cache-manifest mismatch;
- triangulated-root dependencies;
- stale stored absolute paths when the current explicit paths and hashes do not validate.

They must accept a valid test-only publication and preserve source `face_map` and `side_map`.

- [ ] **Step 2: Run the tests to verify failure**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_oof.py -q
```

- [ ] **Step 3: Implement checkpoint/run audit**

Define:

```python
@dataclass(frozen=True)
class OOFRun:
    outer_fold: int
    run_id: str
    seed: int
    checkpoint: Path
    split_manifest: Path
    inference_root: Path
```

Validate checkpoint SHA-256, current split JSON hash, checkpoint split hash, cache-manifest identities, A6 ablation, seed, and exact test membership. Historical absolute paths inside fold-0 metadata are informational only; current explicit paths plus content hashes are authoritative.

- [ ] **Step 4: Implement immutable merge and provenance**

`audit` copies or hard-links only validated test-person cycle NPZ files into:

`local/runs/cohort_cycle/oof_seed0/person_<id>/cycle_<cycle>/prediction.npz`

Write atomically:

- `oof_provenance.csv`, one row per cycle;
- `oof_audit.json`, coverage, duplicate, mismatch, and hash results;
- `oof_manifest.json`, hashes of every published NPZ and upstream artifact.

Abort before publication if all 137 people and all 928 expected cycles are not covered exactly once.

- [ ] **Step 5: Test and commit**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_oof.py -q
git add src/gymnastics/analysis/cohort_cycle tests/cohort_cycle/test_oof.py
git commit -m "feat: audit and publish test only OOF poses"
```

## Task 4: Implement cycle QC, normalization, and the eight core outcomes

**Files:**

- Create: `src/gymnastics/analysis/cohort_cycle/qc.py`
- Create: `src/gymnastics/analysis/cohort_cycle/preprocess.py`
- Create: `src/gymnastics/analysis/cohort_cycle/features.py`
- Create: `src/gymnastics/analysis/cohort_cycle/joints.py`
- Test: `tests/cohort_cycle/test_qc.py`
- Test: `tests/cohort_cycle/test_preprocess.py`
- Test: `tests/cohort_cycle/test_features.py`

- [ ] **Step 1: Write synthetic failing tests**

Test:

- 80% frame and joint validity thresholds and minimum 60 frames;
- finite strictly increasing timestamps;
- interpolation of internal gaps no longer than 10%, with longer gaps rejected;
- 101-point linear phase normalization with exact endpoints;
- direction reversal producing aligned signed trajectories without changing absolute scalars;
- normalized cycle positions from 0 to 1;
- known axial ROM, p95 angular speed, peak phase, tilt, wrist lead, duration, and angular jerk;
- negligible-ROM jerk rejection;
- leave-one-cycle-out repeatability returning zero for identical cycles and a known displacement otherwise;
- person summaries unavailable with fewer than four eligible cycles.

- [ ] **Step 2: Confirm failure**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_qc.py tests/cohort_cycle/test_preprocess.py tests/cohort_cycle/test_features.py -q
```

- [ ] **Step 3: Implement QC and preprocessing as pure functions**

No cohort argument is accepted by QC or feature functions. Load only `kpts_body`, `theta_fused_rad`, `omega_fused_rad_s`, `timestamps`, `frame_valid`, `joint_valid`, frame maps, and metadata. Use the shared MHR70 names and the classification 20-joint set, copied through one tested adapter rather than duplicated numeric indices.

- [ ] **Step 4: Implement typed cycle output**

```python
@dataclass(frozen=True)
class CycleFeatures:
    person_id: str
    cycle_id: str
    outer_fold: int
    cycle_index: int
    normalized_cycle_position: float
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    values: Mapping[str, float | None]
```

Preserve raw metric eligibility flags. Define the wrist convention in one docstring and test it against synthetic left/right rotations before reuse.

- [ ] **Step 5: Write deterministic tables**

`features` writes:

- `cycle_features.csv`;
- `person_features.csv` with median, MAD, and robust slope;
- `qc_exclusions.csv`;
- `qc_summary.json`;
- `phase_curves.npz`;
- `feature_manifest.json` with source and output hashes.

- [ ] **Step 6: Run tests and a two-person smoke extraction**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_qc.py tests/cohort_cycle/test_preprocess.py tests/cohort_cycle/test_features.py -q
conda run -n gymnastic gymnastics cohort-cycle features --config configs/analysis/cohort_cycle.yaml --person 1 --person 85
```

- [ ] **Step 7: Commit**

```bash
git add src/gymnastics/analysis/cohort_cycle tests/cohort_cycle
git commit -m "feat: extract cohort cycle motion descriptors"
```

## Task 5: Implement confirmatory and exploratory statistics

**Files:**

- Create: `src/gymnastics/analysis/cohort_cycle/multiplicity.py`
- Create: `src/gymnastics/analysis/cohort_cycle/statistics.py`
- Create: `src/gymnastics/analysis/cohort_cycle/phase_statistics.py`
- Test: `tests/cohort_cycle/test_multiplicity.py`
- Test: `tests/cohort_cycle/test_statistics.py`
- Test: `tests/cohort_cycle/test_phase_statistics.py`

- [ ] **Step 1: Write synthetic recovery tests**

Generate deterministic nested data with known cohort, repetition, and interaction effects. Assert correct coefficient signs and approximate values for:

```text
outcome ~ cohort + normalized_cycle_position
          + cohort:normalized_cycle_position + C(outer_fold)
```

with person random intercepts. Also test random-slope fallback metadata, cohort-label permutation, person bootstrap, Hedges' g, Cliff's delta, ICC, Holm, BH-FDR, and contiguous phase-cluster correction.

- [ ] **Step 2: Verify the tests fail**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_multiplicity.py tests/cohort_cycle/test_statistics.py tests/cohort_cycle/test_phase_statistics.py -q
```

- [ ] **Step 3: Implement the mixed-model pipeline**

Use `statsmodels.formula.api.mixedlm`. Try a person random slope for normalized cycle position first; retain it only when converged, covariance is nonsingular, and diagnostics pass. Otherwise fit a random-intercept model and store the fallback reason. Apply prespecified log transforms only to positive right-skewed outcomes listed in the YAML.

- [ ] **Step 4: Implement robust secondary inference**

Permute cohort labels at the person level while preserving cohort counts and all cycles. Bootstrap people within cohort. Report effect estimate, 95% CI, raw p, corrected p, sample people/cycles, convergence, transform, and fallback for every result.

- [ ] **Step 5: Implement correction families**

Write separate Holm families for:

- RQ1 cohort main effects over eight core outcomes;
- RQ2 person-MAD contrasts over eight core outcomes;
- RQ3 cohort-by-cycle-position interactions over eight core outcomes.

Use BH-FDR only for explicitly exploratory joints/regions and person-level cluster permutation for phase curves.

- [ ] **Step 6: Write immutable analysis outputs**

`analyze` writes:

- `core_mixed_models.csv`;
- `variability_results.csv`;
- `icc_by_cohort.csv`;
- `phase_clusters.csv`;
- `exploratory_fdr.csv`;
- `sensitivity_results.csv`;
- `model_diagnostics.json`;
- `analysis_manifest.json`.

- [ ] **Step 7: Run tests and commit**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_multiplicity.py tests/cohort_cycle/test_statistics.py tests/cohort_cycle/test_phase_statistics.py -q
git add src/gymnastics/analysis/cohort_cycle tests/cohort_cycle
git commit -m "feat: add hierarchical cohort cycle statistics"
```

## Task 6: Generate auditable tables and figures

**Files:**

- Create: `src/gymnastics/analysis/cohort_cycle/report.py`
- Test: `tests/cohort_cycle/test_report.py`
- Modify: `src/gymnastics/analysis/cohort_cycle/cli.py`
- Modify: `paper/neurocomputing/scripts/generate_paper_assets.py`

- [ ] **Step 1: Write failing report tests**

Assert that report generation:

- reads only finalized CSV/JSON/NPZ analysis artifacts;
- emits a core table row for exactly eight prespecified outcomes;
- renders all four requested figure panels;
- reproduces corrected p-values and CIs verbatim from source tables;
- is byte-stable for table/source CSV and content-stable for the figure;
- refuses incomplete analysis manifests.

- [ ] **Step 2: Confirm failure**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_report.py -q
```

- [ ] **Step 3: Implement publication asset generation**

Write:

- `paper/neurocomputing/artifacts/cohort_cycle_core.csv`;
- `paper/neurocomputing/artifacts/cohort_cycle_qc.csv`;
- `paper/neurocomputing/tables/cohort_cycle_results.tex`;
- `paper/neurocomputing/figures/cohort_cycle_analysis.pdf`;
- supplementary fold, QC, sensitivity, exploratory, and diagnostics tables under `paper/neurocomputing/supplement/`.

The four-panel figure contains a cohort-effect forest plot, MAD comparison, model-estimated repetition trends, and phase curves with corrected clusters.

- [ ] **Step 4: Test and commit**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_report.py -q
git add src/gymnastics/analysis/cohort_cycle tests/cohort_cycle paper/neurocomputing/scripts/generate_paper_assets.py
git commit -m "feat: render cohort cycle paper assets"
```

## Task 7: Run the fold-0 end-to-end pilot before new training

**Files:**

- Generated only: `local/runs/cohort_cycle/pilot/`
- Generated only: `local/runs/fuse_rotation_aware/inference/all137_a6_s1_e100/`
- Generated only: `local/runs/fuse_rotation_aware/inference/all137_a6_s2_e100/`

- [ ] **Step 1: Check GPU and artifact inventory**

```bash
nvidia-smi
conda run -n gymnastic gymnastics cohort-cycle audit --config configs/analysis/cohort_cycle.yaml --check-only
```

- [ ] **Step 2: Re-run test-only fold-0 seed-0 inference if its current publication is incomplete**

```bash
conda run -n gymnastic gymnastics fuse rotation-aware infer \
  --config configs/fusion/rotation_aware.yaml \
  --fold local/runs/cohort_cycle/folds/fold_00.json \
  --run-id all137_a6_e100_seed0 \
  --checkpoint local/runs/fuse_rotation_aware/runs/all137_a6_e100_seed0/checkpoints/best.pt
```

- [ ] **Step 3: Infer fold-0 seeds 1 and 2 for sensitivity**

Run the same command with `all137_a6_s1_e100` and `all137_a6_s2_e100` plus their matching checkpoints.

- [ ] **Step 4: Pilot merge, features, and statistics on fold 0**

Use `--fold 0 --pilot` so the audit expects 14 people rather than all 137:

```bash
conda run -n gymnastic gymnastics cohort-cycle audit --config configs/analysis/cohort_cycle.yaml --fold 0 --pilot
conda run -n gymnastic gymnastics cohort-cycle features --config configs/analysis/cohort_cycle.yaml --pilot
conda run -n gymnastic gymnastics cohort-cycle analyze --config configs/analysis/cohort_cycle.yaml --pilot
```

Confirm schema, QC rates, model convergence handling, plot rendering, and no triangulated-root access. Do not interpret pilot cohort p-values.

- [ ] **Step 5: Record the pilot audit**

Commit only any code/test corrections revealed by the pilot. Do not commit generated `local/runs` data.

## Task 8: Train A6 seed-0 models for outer folds 1--9

**Files:**

- Generated only: `local/runs/fuse_rotation_aware/runs/cohort_oof_f01_a6_e100_s0/` through `cohort_oof_f09_a6_e100_s0/`
- Generated only: `local/runs/cohort_cycle/training_status.csv`

- [ ] **Step 1: Freeze and record the execution environment**

Store git commit, `conda list --explicit`, CUDA/PyTorch versions, GPU model, fold hashes, config hash, and existing fold-0 checkpoint hashes in the run registry.

- [ ] **Step 2: Launch each fold with the existing A6 trainer**

For `XX=01..09`:

```bash
conda run -n gymnastic gymnastics fuse rotation-aware train \
  --config configs/analysis/cohort_cycle_a6_train.yaml \
  --fold local/runs/cohort_cycle/folds/fold_XX.json \
  --run-id cohort_oof_fXX_a6_e100_s0 \
  --ablation A6
```

The training config fixes seed 0, batch size 32, 100 epochs, and the same A6 architecture/loss schedule as the reused fold-0 model. Run no more folds concurrently than available GPUs and memory permit.

- [ ] **Step 3: Validate every completed run before starting inference**

Require:

- exactly the declared train/validation/test people;
- no test person in training or validation;
- `no_pseudo_gt_training=true`;
- finite training/validation metrics;
- a best checkpoint;
- matching split/config/cache hashes;
- no triangulated dependency in recorded inputs.

- [ ] **Step 4: Update resumable training status**

Write one row per fold with state, start/end time, best epoch, validation loss, checkpoint hash, and failure reason. A failed fold may be rerun under the same ID only if the prior directory is preserved and the replacement gets a new attempt suffix.

## Task 9: Infer, merge, analyze, and run sensitivities

**Files:**

- Generated only: `local/runs/fuse_rotation_aware/inference/cohort_oof_fXX_a6_e100_s0/`
- Generated only: `local/runs/cohort_cycle/oof_seed0/`
- Generated only: `local/runs/cohort_cycle/analysis/`

- [ ] **Step 1: Infer only each fold's test people**

For `XX=01..09`:

```bash
conda run -n gymnastic gymnastics fuse rotation-aware infer \
  --config configs/analysis/cohort_cycle_a6_train.yaml \
  --fold local/runs/cohort_cycle/folds/fold_XX.json \
  --run-id cohort_oof_fXX_a6_e100_s0
```

- [ ] **Step 2: Build and audit the full primary OOF publication**

```bash
conda run -n gymnastic gymnastics cohort-cycle audit --config configs/analysis/cohort_cycle.yaml
```

Require exactly 137 people, 928 cycles, ten folds, zero duplicate publications, zero split violations, and matching hashes before continuing.

- [ ] **Step 3: Extract core and exploratory features**

```bash
conda run -n gymnastic gymnastics cohort-cycle features --config configs/analysis/cohort_cycle.yaml
```

Review differential missingness by cohort and fold before outcome models.

- [ ] **Step 4: Run confirmatory statistics**

```bash
conda run -n gymnastic gymnastics cohort-cycle analyze --config configs/analysis/cohort_cycle.yaml
```

Record all convergence/fallback decisions and all correction families. Null results remain in the output.

- [ ] **Step 5: Run prespecified pose-source sensitivities**

Repeat compatible core descriptors for:

- face-view A6 input;
- side-view A6 input;
- deterministic `avg_body_current`;
- OOF A6;
- fold-0 A6 seeds 0, 1, and 2.

Label A6-only effects `model-dependent`, opposing single-view effects `view-sensitive`, and cross-source agreement `source-robust but not biomechanically validated`.

- [ ] **Step 6: Generate final assets**

```bash
conda run -n gymnastic gymnastics cohort-cycle assets --config configs/analysis/cohort_cycle.yaml
```

## Task 10: Integrate generated evidence into the manuscript

**Files:**

- Create: `paper/neurocomputing/sections/06b_cohort_cycle_analysis.tex`
- Modify: `paper/neurocomputing/manuscript.tex`
- Modify: `paper/neurocomputing/sections/01_introduction.tex`
- Modify: `paper/neurocomputing/sections/02_related_work.tex`
- Modify: `paper/neurocomputing/sections/05_experimental_protocol.tex`
- Modify: `paper/neurocomputing/sections/07_discussion.tex`
- Modify: `paper/neurocomputing/sections/08_limitations.tex`
- Modify: `paper/neurocomputing/sections/09_conclusion.tex`
- Modify: `paper/neurocomputing/references.bib`
- Modify: `paper/neurocomputing/scripts/check_manuscript.py`
- Modify, only after final results: `paper/neurocomputing/manuscript.tex` abstract and `paper/neurocomputing/highlights.txt`
- Test: `tests/cohort_cycle/test_manuscript_claims.py`

- [ ] **Step 1: Add failing evidence/wording checks**

Check:

- the cohort table values exactly match generated CSV;
- all eight outcomes appear once;
- no unsupported phrases such as `ageing effect`, `caused by age`, `biomechanically validated`, or `ground-truth joint angle`;
- `elderly cohort` and `student cohort` are defined;
- every numerical cohort claim has a generated source;
- abstract remains at most 250 words and highlights satisfy journal limits;
- no cohort result placeholder remains after finalization.

- [ ] **Step 2: Add methods text before seeing outcomes**

Extend the Introduction, Related Work, and Experimental Protocol with the frozen cohort question, 10-fold cross-fitting, QC, eight outcomes, mixed model, multiplicity control, and sensitivity design. Add verified primary literature only; do not infer missing demographics.

- [ ] **Step 3: Add the standalone generated-results section**

Include `sections/06b_cohort_cycle_analysis.tex` immediately after `06_results.tex`. Structure it as:

1. cross-fitted coverage and QC;
2. between-cohort descriptors;
3. within-person cycle variability;
4. repetition-order trends;
5. exploratory phase/body-region results;
6. pose-source and seed sensitivities.

Populate directions, estimates, CIs, and corrected p-values only from finalized tables.

- [ ] **Step 4: Update interpretation and limitations**

Separate typical motion, within-person variability, and repetition trends. Explicitly state cross-sectional cohort association, unavailable age/sex/body-size covariates, recruitment/recording confounding, estimated-pose limitations, limited cycles per person, and incomplete seed coverage.

- [ ] **Step 5: Update abstract/highlights only if supported**

Add at most one methods sentence and one results sentence. If sensitivities are unstable, omit the result from highlights and describe it only in Results/Limitations.

- [ ] **Step 6: Control manuscript length**

Move detailed A7--A9 negative controls, extended deterministic interpretation, full-joint results, and diagnostics to supplementary material. Target a net addition of 800--1,200 words.

- [ ] **Step 7: Run manuscript and PDF verification**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle/test_manuscript_claims.py -q
cd paper/neurocomputing
make assets
make check
make pdf
```

Expected: no pending cohort values, undefined citations/references, unsupported causal language, or LaTeX errors.

- [ ] **Step 8: Commit**

```bash
git add paper/neurocomputing tests/cohort_cycle/test_manuscript_claims.py
git commit -m "paper: add OOF cohort and repeated cycle analysis"
```

## Task 11: Final verification and evidence audit

**Files:**

- Verify: all changed source, test, config, and paper files
- Generated report: `local/runs/cohort_cycle/final_audit.json`

- [ ] **Step 1: Run the focused suite**

```bash
conda run -n gymnastic python -m pytest tests/cohort_cycle -q
```

- [ ] **Step 2: Run affected existing suites**

```bash
conda run -n gymnastic python -m pytest \
  tests/rotation_aware/test_cli.py \
  tests/rotation_aware/test_dataset.py \
  tests/rotation_aware/test_inference.py \
  tests/structure/test_cli.py -q
```

- [ ] **Step 3: Run provenance and leakage audits**

```bash
conda run -n gymnastic gymnastics cohort-cycle audit \
  --config configs/analysis/cohort_cycle.yaml \
  --strict --write-final-audit
```

The final audit must certify:

- 137/137 people and 928/928 cycles covered once;
- every publication came from its person's outer-test fold;
- checkpoint, split, cache, feature, result, table, and figure hashes match;
- no triangulated pose was used by training or feature extraction;
- correction families contain exactly the prespecified hypotheses;
- manuscript numbers match generated assets.

- [ ] **Step 4: Confirm ethics/consent language**

Before declaring the paper complete, obtain the study's actual ethics approval and consent wording for this secondary cohort comparison. Do not invent an approval body, protocol number, or consent statement.

- [ ] **Step 5: Inspect the compiled deliverable**

Open `paper/neurocomputing/build/manuscript.pdf`, verify the four-panel figure and table are legible, and record page/word counts in `final_audit.json`.

- [ ] **Step 6: Review diff and repository state**

```bash
git diff --check
git status --short
git log --oneline -8
```

Confirm unrelated pre-existing user changes were neither staged nor modified by this work.
