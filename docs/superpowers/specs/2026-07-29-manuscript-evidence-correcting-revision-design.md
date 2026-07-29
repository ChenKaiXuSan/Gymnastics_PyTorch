# Evidence-Correcting Manuscript Revision Design

**Date:** 2026-07-29
**Target manuscript:** `paper/neurocomputing` in the
`rotation-aware-fusion` worktree
**Target venue class:** Neurocomputing-style applied machine-learning journal
**Revision type:** Major, evidence-correcting revision
**Author decision:** Keep A6 as the paper's method mainline, but do not claim
that it improves positional accuracy over canonical deterministic fusion.

## Objective

Revise the manuscript so that every headline claim is supported by an
appropriate evaluation population and a clearly defined estimand. The revision
will replace in-sample learned-model evidence with held-out evidence, add the
completed Unity native-3D benchmark as limited independent validation,
re-estimate the cohort contrast at a representative repetition position, and
make unresolved governance requirements explicit.

## Evidence Policy

1. Training, validation, test, all-person descriptive, and downstream
   cross-fitted populations remain separate in text, tables, and figures.
2. The 14-person private-data test split is the primary learned-model
   generalization result. Results over all 137 people are descriptive only.
3. Unity native 3D is an independent synthetic evaluation reference. It is not
   described as population-level validation because the benchmark contains one
   avatar, one environment, and one fixed camera pair.
4. Triangulated private-data poses remain an upstream pseudo-reference derived
   from the same videos. They are never called ground truth.
5. A6 remains the principal proposed architecture. Its supported contribution
   is label-free, view-swap-invariant, bounded temporal fusion with explicit
   rotation constraints. The paper will state that A6 does not outperform the
   best deterministic or lighter learned baselines on positional accuracy.
6. No missing seed, independent-reference, ethics, consent, demographic, or
   clinical evidence will be inferred or invented.

## Revision Scope

### 1. Primary learned evaluation

- Replace the current 137-person A0--A9 table as the primary learned result
  with the frozen 14-person held-out test comparison.
- Report person-level mean and SD, paired bootstrap confidence intervals, and
  available paired comparisons from generated evaluation artifacts.
- Retain the 137-person values only in an explicitly labeled descriptive
  appendix or secondary table if they add diagnostic value.
- Treat the 27-person fixed-corruption results as validation diagnostics, not
  test-set robustness evidence.
- Keep A6 as the selected mainline because A7's descriptive all-person ROM
  pattern does not reproduce on the held-out test split.

### 2. Unity native-3D benchmark

- Add a compact dataset-and-protocol paragraph describing 199 samples across
  three sequences, the Unity16 joint subset, one Sim3 alignment per sequence,
  and Unity native 3D as evaluation ground truth.
- Add a zero-shot table containing the best single view, best direct-3D
  deterministic fusion, A4--A7 or the best learned rows, and SAM3D-2D
  triangulation.
- State the negative transfer result directly: A6 and the other zero-shot
  learned fusion variants do not beat the best single view or best direct-3D
  fusion, while triangulation is substantially more accurate.
- Include the two-fold, three-seed calibrated learned baselines only as a
  separate input-regime comparison. Do not pool calibrated 2D-to-3D,
  calibrated 3D-to-3D, and uncalibrated post-estimation methods into one
  ranking.
- Use the benchmark to narrow external-validity claims, not to imply that
  synthetic validation resolves private-cohort generalization.

### 3. Metric and evaluation definitions

- Trace the private learned evaluator and deterministic evaluator to determine
  whether each uses hip-centering, sequence-level Sim3, or another alignment.
- Define the exact transformation before each reported MPJPE table.
- Define the denominator for ROM and peak-angular-velocity retention. If A3 is
  the normalization reference, rename the metric accordingly; if the
  triangulated pseudo-reference is the denominator, regenerate inconsistent
  rows.
- Prohibit numerical comparisons across tables that use different alignment
  rules or evidence populations.
- Add a generated provenance table or footnote mapping every headline result
  to its artifact, population, alignment, and seed count.

### 4. Cohort and repeated-cycle analysis

- Refit each mixed model after centering normalized cycle position at 0.5.
  The cohort coefficient will then represent the adjusted contrast at the
  midpoint of a participant's observed repetitions.
- Retain the cohort-by-cycle-position interaction as the repetition-trend test.
- Report a sensitivity model without artificial outer-fold fixed effects.
- Where generated source data permit, apply the same centered mixed-model
  estimand to OOF A6, face-only, side-only, and deterministic fusion. If an
  input source cannot support this model, state the reason and keep its
  person-median result in a separately labeled estimand family.
- Report random-intercept and random-slope variance, convergence status, and
  residual/influence diagnostics.
- Correct phase-curve cluster findings across the four descriptor families, or
  label them as uncorrected exploratory analyses.
- Regenerate the cohort figure. Forest-plot rows must not imply that
  coefficients on heterogeneous raw and log scales are directly comparable.
  Repetition trends must include confidence bands or be moved to a
  non-inferential descriptive panel.

### 5. Manuscript positioning

- Preserve the current title unless the revised abstract still implies an
  accuracy advantage that the results do not show.
- Rewrite the abstract so that the held-out private test and Unity results
  precede the downstream cohort application.
- Reframe the contribution as an auditable method and evaluation design,
  emphasizing the supervision boundary, symmetry, and motion-preservation
  objectives.
- Explicitly state that canonicalization accounts for most of the private-data
  gain and that the residual learner adds no demonstrated positional benefit.
- Treat the cohort analysis as a hypothesis-generating application rather than
  a second confirmatory contribution.
- Expand related work only with references already verified from primary
  sources. New citations require source verification before insertion.

### 6. Editorial and governance corrections

- Correct `\operatorname{atan2}` and `\delta_j^{\max}` in the method equations.
- Increase precision where rounded confidence bounds visually collapse to
  zero.
- Complete the institution name and postal address only from author-provided
  information.
- Retain a visible submission blocker for ethics approval or exemption,
  participant consent, and privacy governance until the author supplies the
  verified wording.
- Keep the raw videos private. Describe code, derived keypoint, configuration,
  and aggregate-result availability according to the final release decision.
- Update the source audit and submission checklist so that they match the
  regenerated manuscript.

## Files and Data Flow

- Statistical code and generated cohort artifacts:
  `src/gymnastics/analysis/cohort_cycle/`,
  `local/runs/cohort_cycle/`, and the
  `codex/oof-cohort-cycle-analysis` worktree.
- Independent benchmark artifacts:
  `local/runs/unity_benchmark/report/`,
  `local/runs/unity_benchmark/supervised_finetune/report/`, and
  `local/runs/unity_benchmark/extrinsic_learning/report/`.
- Private learned evaluation artifacts:
  `local/runs/fuse_rotation_aware/evaluation/` and generated comparison CSVs.
- Manuscript source:
  `.worktrees/rotation-aware-fusion/paper/neurocomputing/`.
- Manuscript generation scripts remain the only path for numerical table and
  figure updates; generated values will not be hand-entered when a source
  artifact exists.

## Non-Goals

- Do not claim that the Unity benchmark is a public population benchmark.
- Do not run a new human study or infer missing participant covariates.
- Do not convert the observational cohort comparison into a causal ageing
  claim.
- Do not present Unity-supervised fine-tuning as self-supervised A6 training.
- Do not conceal negative external validation or the absence of full private
  A4--A7 multi-seed replication.
- Do not remove the ethics blocker merely to make the PDF appear
  submission-ready.

## Verification

1. Unit tests cover every changed analysis or asset-generation behavior.
2. Generated tables are compared against their JSON/CSV sources.
3. The manuscript consistency checker reports no numerical contradiction,
   unresolved result placeholder, orphan citation, or population-label
   mismatch.
4. LaTeX compilation completes successfully.
5. The final PDF receives visual inspection for formula rendering, table
   overflow, figure legibility, and transformed-scale labeling.
6. A final claim-evidence audit checks the abstract, contributions, results,
   discussion, limitations, and conclusion against the same evidence map.

## Acceptance Criteria

- The primary learned table uses only the 14-person held-out test set.
- A6 is not described as improving MPJPE or corruption recovery over the best
  comparator unless a generated paired analysis supports that statement.
- Unity native-3D results appear with their one-avatar limitation and negative
  zero-shot conclusion.
- Every MPJPE and retention value names its alignment or denominator.
- The main cohort coefficient is centered at cycle position 0.5, and
  sensitivity analyses identify any estimand change.
- Phase-localized claims are multiplicity-corrected or explicitly
  exploratory.
- Equations render with valid mathematical operators and symbols.
- Ethics, consent, institution, and data-release status are accurate and
  visibly unresolved where author evidence is still missing.
- The final PDF compiles and passes automated and visual checks.
