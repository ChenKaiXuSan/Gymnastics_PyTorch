# Out-of-Fold Cohort and Repeated-Cycle Analysis Design

## Objective

Add a leakage-controlled downstream application analysis to the
rotation-aware fusion paper. The analysis uses out-of-fold A6 fused poses to
compare:

1. person-level motion descriptors between the elderly and student cohorts;
2. cycle-to-cycle variation within each person;
3. change over repeated cycle order; and
4. phase-resolved and body-region patterns as exploratory evidence.

The analysis supports the practical usefulness of A6 for interpretable motion
analysis. It does not establish absolute biomechanical accuracy, prove that A6
is more accurate than deterministic fusion, or identify a causal effect of
ageing.

## Evidence and Causal Boundaries

- The dataset contains a cohort label, not participant-level age, sex, height,
  weight, or BMI.
- Results must be described as differences associated with the `elderly` and
  `student` cohorts. They must not be described as causal ageing effects.
- Recruitment source, recording batch, body size, sex composition, and other
  unavailable variables may confound cohort comparisons.
- All pose-derived outcomes are estimated kinematic descriptors. They are not
  clinical joint angles, laboratory motion-capture measurements, or validated
  biomechanical endpoints.
- Multiple cycles from one person are repeated observations, not independent
  participants.
- Cohort labels may be used to stratify cross-fitting folds, but they must not
  enter A6 inputs, training losses, validation scores, checkpoint selection, or
  feature selection.

## Cohort Inventory

- Elderly cohort: person IDs `1` through `80` (`N=80`, 539 cycles).
- Student cohort: person IDs `81` through `134` and `136` through `138`
  (`N=57`, 389 cycles).
- Person `135` / student `S55` is absent.
- Total: 137 people and 928 cycles.
- Each person has 6 to 9 cycles; the median is 7 in both cohorts.
- Cohort membership is validated from
  `/home/data/xchen/gymnastics/raw/person/student_id_mapping.csv` and
  `/home/data/xchen/gymnastics/raw/person/organize_from_dropbox_20260718.csv`.

## Ten-Fold Cross-Fitting

### Outer folds

Use ten person-disjoint, cohort-stratified outer folds:

- Preserve the current 14-person A6 test set as outer fold 0. It contains eight
  elderly participants and six students:
  `1, 24, 36, 49, 51, 52, 60, 79, 85, 106, 116, 117, 130, 136`.
- Partition the remaining 123 people deterministically into nine folds.
- Every fold contains exactly eight elderly participants.
- Among the student participants, seven folds contain six students and three
  folds contain five students. Fold 0 is one of the six-student folds.
- Seven folds therefore contain 14 people and three contain 13 people.
- Within each outer training pool, reserve 27 cohort-stratified people for
  validation. Folds with 14 outer-test people use 96 train / 27 validation /
  14 test. Folds with 13 outer-test people use 97 train / 27 validation /
  13 test.
- A deterministic manifest records every person's cohort, outer fold, train /
  validation / test role, split-generation seed, and source mapping hashes.

### Checkpoint reuse and new training

- Reuse the existing seed-0, seed-1, and seed-2 A6 checkpoints for fold 0.
- Use seed 0 as the primary cross-fitted model.
- Train one seed-0 A6 checkpoint from scratch for each of folds 1 through 9.
- Do not initialize a new outer-fold model from a checkpoint that has seen the
  new fold's test people.
- Existing fold-0 seeds 1 and 2 provide a prespecified seed-sensitivity check.
- Additional seeds for folds 1 through 9 are optional follow-up evidence, not a
  prerequisite for the first complete cohort analysis.

### Out-of-fold inference publication

- Each person is inferred only by the seed-0 checkpoint for the outer fold in
  which that person is a test member.
- Merge the ten test-only inference publications into one immutable
  `oof_seed0` result root.
- Reject duplicate people, missing people, split-manifest mismatches, checkpoint
  provenance mismatches, and cycles absent from the prepared cache.
- Publish a machine-readable provenance table mapping each output cycle to
  person, cohort, outer fold, run ID, checkpoint hash, cache-manifest hash, and
  source frame maps.

## Cycle Quality Control

Quality control is defined before cohort outcomes are inspected.

- A cycle is globally eligible when at least 80% of its frames have
  `frame_valid=True`, timestamps are finite and strictly increasing, and at
  least 60 valid frames remain.
- A metric is eligible when all joints required for that metric are valid in at
  least 80% of frames.
- Linear interpolation may bridge internal missing runs covering no more than
  10% of the cycle. Longer runs make the affected metric unavailable.
- A person-level metric requires at least four eligible cycles.
- Exclusions are reported by person, cycle, metric, cohort, and outer fold.
- Exclusion decisions never use cohort differences, effect directions,
  statistical significance, or pseudo-reference error.
- Differential missingness between cohorts is reported before outcome results.

## Preprocessing

1. Read `kpts_body`, `theta_fused_rad`, `omega_fused_rad_s`, `timestamps`,
   `frame_valid`, and `joint_valid` from the out-of-fold A6 exports.
2. Use the A6 pelvis-centred, orientation-normalised, trial-scaled body frame.
   Angular and temporal primary outcomes need no millimetre scale.
3. Unwrap the pelvis-to-thorax axial angle within each complete cycle.
4. Align direction for phase-resolved aggregation: if the dominant signed axial
   excursion is negative, multiply the signed axial trajectory and associated
   signed coordination trajectories by `-1`. Direction-invariant scalar
   outcomes use absolute values and are unchanged.
5. Resample eligible trajectories by linear interpolation to 101 phase points
   from 0% through 100%. Linear interpolation is used to avoid spline overshoot.
6. Define normalized cycle position as `(cycle_index - 1) /
   (number_of_cycles - 1)`, so the first and last repetitions map to 0 and 1.

## Prespecified Core Outcomes

Eight outcomes form the confirmatory analysis family.

1. **Trunk axial rotation ROM**: 95th minus 5th percentile of the unwrapped
   pelvis-to-thorax axial angle, in radians.
2. **High angular speed**: 95th percentile of absolute axial angular velocity,
   in radians per second.
3. **Peak-rotation phase**: normalized phase at the maximum absolute deviation
   of axial angle from its cycle median.
4. **Trunk tilt**: 95th percentile of the absolute angle between the
   hip-centre-to-shoulder-centre vector and the body-frame vertical axis.
5. **Wrist lead / wrapping angle**: robust high-percentile lagging-wrist angle
   relative to the pelvis-centred trunk frame, with lagging side determined
   from the aligned rotation direction. The exact joint and sign convention
   must be shared with the existing `gymnastics.analysis.metrics` definitions
   and tested on synthetic poses.
6. **Cycle duration**: last valid timestamp minus first valid timestamp, in
   seconds.
7. **Dimensionless angular jerk**: log-transformed square-root normalized jerk,
   `log(sqrt(T^5 / A^2 * integral(jerk^2 dt)) + eps)`, where `T` is cycle
   duration and `A` is robust axial ROM. Cycles with negligible `A` are marked
   unavailable rather than divided by an unstable denominator.
8. **Whole-body trajectory repeatability error**: leave-one-cycle-out RMS
   distance between a cycle and the person's median phase-normalized trajectory
   over the 20 major body joints used by the classification mapping. Distances
   are measured in A6 body-frame trial-scale units, not millimetres.

For each outcome, the person-level typical value is the median over eligible
cycles and within-person variability is the median absolute deviation (MAD).
The first-to-last slope is estimated against normalized cycle position.

## Exploratory Outcomes

- Phase-resolved axial angle, angular velocity, trunk tilt, and wrist lead.
- Body-region and joint-level movement amplitude, high velocity, and
  repeatability for trunk, arms, legs, hands, and feet.
- Full 70-joint results remain exploratory because distal hand and foot joints
  are not independently validated.
- Exploratory outcomes never replace a null core outcome in the main claims.

## Statistical Analysis

### Cycle-level mixed-effects model

For each eligible core cycle outcome:

```text
outcome ~ cohort + normalized_cycle_position
          + cohort:normalized_cycle_position
          + outer_fold + (1 | person)
```

- `cohort` answers the adjusted between-cohort question.
- `normalized_cycle_position` answers the shared repetition-order trend.
- The interaction answers whether trends differ by cohort.
- `outer_fold` controls systematic differences between cross-fitted models.
- Add a person-specific random slope for normalized cycle position when the
  model converges without a singular fit. Otherwise use the random-intercept
  model and report the fallback.
- Positive, right-skewed outcomes may use a prespecified log transform.
- Inspect residuals, influence, convergence, and heteroscedasticity.
- If assumptions fail materially, retain person-level descriptive estimates
  and use a cohort-stratified person-label permutation test as the robust
  inferential fallback.

### Within-person variation

- Compare person-level MAD between cohorts with a person-label permutation
  test.
- Report cohort medians and IQRs, median difference, Hedges' `g`, Cliff's
  delta, and person-stratified bootstrap 95% confidence intervals.
- Estimate ICC separately by cohort as auxiliary repeatability evidence with
  person-bootstrap confidence intervals.

### Multiple comparisons

- RQ1 cohort main effects across the eight core outcomes form one family and use
  Holm correction.
- RQ2 MAD comparisons across the eight core outcomes form a second Holm family.
- RQ3 cohort-by-cycle-position interactions form a third Holm family.
- Joint- and body-region exploratory summaries use Benjamini-Hochberg FDR.
- Phase-resolved curves use cluster-based permutation over contiguous phase
  samples, resampling at the person level.
- A significant result in one cohort and a non-significant result in the other
  is never treated as evidence of a group difference; the interaction is tested
  directly.

## Sensitivity Analyses

Repeat the core feature extraction for:

- face-view A6 input;
- side-view A6 input;
- deterministic `avg_body_current`; and
- out-of-fold A6.

Interpretation is fixed in advance:

- consistent direction and comparable magnitude across sources reduces concern
  that A6 alone created the cohort contrast;
- an effect appearing only in A6 is labelled model-dependent;
- opposing face and side effects with an intermediate A6 effect are labelled
  view-sensitive;
- fold-0 agreement across seeds 0, 1, and 2 provides limited seed-stability
  evidence;
- sensitivity agreement does not establish biomechanical truth.

## Software Boundaries

Implementation should keep training orchestration, feature extraction,
statistics, and paper-asset generation separate:

- cross-fold manifest generation and validation extend the existing
  `gymnastics.fusion.rotation_aware` workflow;
- motion descriptors and quality-control functions live in a focused cohort
  analysis module under `gymnastics.analysis`;
- statistical analysis reads a tidy cycle-level table and writes immutable
  machine-readable result tables;
- figure and LaTeX table generation reads only finalized analysis tables;
- no paper claim is calculated independently inside LaTeX-generation code.

All project Python commands, tests, training, inference, and analysis use the
`gymnastic` conda environment.

## Required Result Artefacts

- ten-fold split manifest with hashes;
- cross-fit run registry and completion audit;
- one OOF cycle-level feature CSV or Parquet table;
- one person-level summary table;
- core mixed-model estimates and diagnostics;
- within-person variability estimates;
- phase-cluster and FDR-corrected exploratory results;
- pose-source and fold-0 seed sensitivity tables;
- QC and exclusion report;
- publication figure source data;
- one main multi-panel figure and one core results table;
- supplementary fold, QC, full-body, sensitivity, and diagnostic materials.

## Manuscript Integration

### Abstract and highlights

- Add one cross-fitting/application-method sentence and one result sentence only
  after results are finalized.
- Keep the abstract within 250 words by shortening A8--A9 details.
- Do not claim an ageing effect or biomechanical validation.
- Consider replacing one highlight with an out-of-fold repeated-motion analysis
  highlight only if the result is stable across sensitivity analyses.

### Introduction

- Motivate repeated-motion analysis beyond pose-error metrics.
- Add a secondary research question covering cohort-associated typical motion,
  cycle variability, and repetition-order trends.
- Add one contribution for the leakage-controlled downstream analysis.

### Related work

Extend applied motion analysis with verified literature on:

- markerless 3D pose for sport and movement analysis;
- cohort comparison in older and younger populations; and
- repeated-measure / cycle-variability analysis.

### Experimental protocol

Add subsections for:

- application cohorts and ten-fold cross-fitting;
- cycle QC, normalization, and core descriptors; and
- hierarchical, variability, multiplicity, and sensitivity analyses.

### Results

Add a standalone section after fusion results:

`Downstream cohort and repeated-cycle analysis`

with subsections:

1. cross-fitted coverage and quality control;
2. between-cohort movement differences;
3. within-person cycle variability;
4. repetition-order trends;
5. phase-resolved and body-region analysis; and
6. pose-source and seed sensitivity.

The section is stored separately and included after
`sections/06_results.tex`, without renaming the existing discussion and
limitation files.

### Discussion, limitations, and conclusion

- Interpret typical movement, variability, and repetition trend separately.
- Relate findings to verified domain literature without causal language.
- State whether findings are consistent across pose sources.
- Add missing demographics, recruitment/recording confounding, cross-sectional
  design, estimated-pose, repeated-cycle count, and seed-coverage limitations.
- Add one restrained conclusion paragraph describing what cross-fitted A6
  enables, not what ageing causes.

### Space control

The current manuscript is approximately 6,969 words and 18 pages. Add roughly
1,600 to 2,250 words across protocol, results, discussion, and limitations, then
move extended deterministic interpretation, A7--A9 negative-control detail,
full-body results, and diagnostics to supplementary material. Target a net
increase of approximately 800 to 1,200 words.

## Publication Tables and Figures

### Main figure

One four-panel figure:

1. forest plot of person-level cohort effects;
2. cohort comparison of within-person MAD;
3. model-estimated repetition-order trends; and
4. representative phase-normalized curves with corrected phase clusters.

### Main table

For each core outcome:

- elderly median and IQR;
- student median and IQR;
- adjusted cohort effect and 95% CI;
- Holm-adjusted cohort p-value;
- variability effect and corrected p-value; and
- cohort-by-cycle interaction and corrected p-value.

### Supplement

- complete fold membership and provenance;
- QC and exclusions;
- full joint/body-region results;
- face/side/deterministic/A6 sensitivity;
- fold-0 seed sensitivity; and
- model convergence and residual diagnostics.

## Claim Rules

- Do not write result directions before generated analysis artefacts exist.
- Do not promote exploratory outcomes into the core family after seeing results.
- If the core comparison is null, report confidence intervals and retained
  effect bounds rather than removing the analysis.
- If a result appears only in A6, call it model-dependent.
- If a result is stable across A6, deterministic fusion, and single views,
  describe it as source-robust but not biomechanically validated.
- The downstream analysis demonstrates analytical usability, not superior A6
  accuracy.

## Verification

- Unit tests for fold counts, person disjointness, cohort stratification, reuse
  of fold 0, and full OOF coverage.
- Unit tests for direction alignment, phase interpolation, each core metric,
  invalid-run handling, and leave-one-cycle-out repeatability.
- Synthetic statistical tests with known cohort, trend, and interaction effects.
- Deterministic rerun checks for manifests, features, tables, and figures.
- Audit that no triangulated pose enters cross-fit training or cohort feature
  extraction.
- Audit that no train or validation person is published as OOF for the same
  checkpoint.
- Paper checks for pending values, cohort/age causal wording, table-source
  agreement, and abstract/highlight length.

## Completion Boundary

The addition is complete only when:

1. all 137 people have exactly one primary seed-0 OOF publication;
2. every analysis table is generated from the OOF publications;
3. QC, model diagnostics, multiplicity corrections, and sensitivity results are
   complete;
4. manuscript claims match the generated tables;
5. ethics and consent language for secondary pose analysis and cohort comparison
   is confirmed; and
6. the manuscript compiles with no pending result markers, undefined
   references, or unsupported cohort claims.
