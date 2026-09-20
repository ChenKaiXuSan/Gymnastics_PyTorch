# Results Summary

This page is the tracked entry point for the project's results. The learned
rows below (A0--A9) belong to the **archived rotation-aware model**
(`src/fusion/archive/rotation_aware`, frozen 2026-09-19); they are kept because
the Sports Engineering manuscript reports them. Results for the active
cycle-aware model and the external-baseline main matrix are in the dated
reports under [research/](research/), in particular
[cycle_aware_architecture_ablation_2026-09-20.md](research/cycle_aware_architecture_ablation_2026-09-20.md)
and [main_matrix_external_baselines_2026-09-19.md](research/main_matrix_external_baselines_2026-09-19.md).

The numeric tables are regenerated from local per-person and per-fold artefacts:

```bash
python -m fusion.analysis.project_results
```

The detailed generated report is
`local/runs/analysis/project_results/RESULTS_SUMMARY.md`. It remains local
because the underlying experiment outputs are intentionally not tracked by Git.

## Evidence hierarchy

| Result family | Primary cohort | Interpretation |
|---|---:|---|
| Deterministic fusion matrix | 137 people | Paired descriptive comparison; no learned parameters |
| Learned A0--A9 generalization | held-out test, 14 people | Primary learned-model evidence |
| Learned A0--A9 all-person result | 137 people | Descriptive only; includes train and validation people |
| Fixed-corruption recovery | validation, 27 people | Validation diagnostic; unavailable on test people |

The learned split contains 96 training, 27 validation, and 14 test people, with
no person overlap.

## Main findings

### Triangulated pseudo-reference

The reconstructed pseudo-reference covers 137 people, 928 cycles, and 147,297
paired frames, with no missing face/side pairs. Mean reprojection error is
7.22 px (median 6.10 px; maximum 32.59 px). The strict validator reports all
137 expected people and 928 cycles as structurally complete, while retaining 26
quality warnings for review. This is a pseudo-reference derived from the same
camera observations, not independent motion-capture ground truth.

### Deterministic fusion

Across all 137 people, `avg_body_current` is the recommended leakage-free
method (mean person MPJPE 64.05 mm in the deterministic harness). The nominally
lower `sim3_face_stable_joint_weight` result is excluded from recommendation
because its per-joint weights are derived from the triangulated pseudo-reference
used for evaluation.

### Rotation-aware learned fusion

Held-out test MPJPE is the primary learned-model result:

| Ablation | Test MPJPE, mean ± SD (mm) | Interpretation |
|---|---:|---|
| A0 | 77.92 ± 12.52 | face-only input |
| A1 | 75.84 ± 13.93 | side-only input |
| A2 | 60.85 ± 8.98 | deterministic canonical arithmetic |
| A5 | 60.80 ± 8.96 | learned rotation/temporal model |
| A6 | 60.78 ± 8.72 | full self-supervised model |
| A7 | 61.31 ± 8.87 | ROM-anchor variant |
| A8 | 92.02 ± 9.59 | twist residual degrades position accuracy |
| A9 | 94.11 ± 9.14 | rate anchor does not repair A8 |

A6 substantially improves over either single view, but it does not establish an
MPJPE improvement over A2 or A5 on the 14-person test set. Paired bootstrap
intervals and Holm-adjusted Wilcoxon results are generated in
`learned_test_comparisons.csv`. A7's apparent twist/smoothness gain in the
descriptive 137-person aggregate does not reproduce on the test split (ROM
retention 0.948 versus 1.000 for A6), so A6 remains the paper mainline. A8 and A9
are useful negative results. A9 stopped at epoch 85 of the nominal 100-epoch
schedule, with its best checkpoint near epoch 83.

The corresponding 137-person A6 MPJPE is 65.71 mm, but that number includes
people used for fitting and validation and is therefore descriptive rather than
held-out generalization evidence.

### Fixed-corruption diagnostic

Fixed-corruption recovery was measured only for the 27 validation people:
A4 0.0878, A5 0.0880, A6 0.1205, A7 0.0870, A8 0.3782, and A9 0.3960.
It must not be reported as a 137-person or test-set result.

### Failure-structure diagnostics (2026-09-16)

- Silent single-view corruption (27 validation people, two tiers): at the
  default tier arithmetic fusion is equal or better everywhere; at the severe
  tier the learned models absorb 29--42% of the damage for the two
  anatomy-detectable families (impulse spikes: A5 0.0696 vs A2 0.0726,
  p=3.0e-8; thorax rotation bias: A5 p=1.1e-4) and tie or lose for
  motion-plausible families. Artefacts:
  `local/runs/analysis/single_view_silent_corruption/`.
- FreeMan common-mode failure analysis (552 sessions, 587,124 frames):
  per-view catastrophic failures (>150 mm) co-occur 57x above independence
  (~95% co-occurrence); median error is orientation-independent; oracle
  per-frame view selection improves the two-view mean by only 5.5%. Cross-view
  redundancy provides no exploitable independent failures, bounding any
  post-estimation fusion. Artefacts:
  `local/runs/analysis/view_reliability_phase/`.

## Failure and uncertainty coverage

- The triangulation report ranks the worst people and cycles by reprojection
  error; the strict validator retains quality warnings instead of silently
  dropping them.
- Learned A8/A9 regressions and the earlier A9 gradient failure are documented
  as negative/stability findings.
- Absolute metric scale remains limited by the monocular reconstruction scale.
- The learned evaluation uses triangulated pseudo-reference data originating
  from the same video observations.

## Evidence still pending

- Inference and evaluation of additional random seeds on the private data
  (A6 seeds 1 and 2 are now evaluated zero-shot on FreeMan, where the
  three-seed PA-MPJPE spread is below 0.1 mm; the private-data evaluation
  remains seed 0 only).
- Independent motion-capture ground truth. A zero-shot FreeMan evaluation
  over the ten most trunk-rotation-relevant subjects (552 sessions, 60% of
  the release, stride 1) is complete: fusion reaches 102.9--103.4 mm
  per-frame PA-MPJPE against the markerless multi-view reference versus
  106.3/106.9 mm for the single views, naive baselines 129.8 mm, and the A6
  three-seed spread is below 0.05 mm. The report lives at
  `local/runs/freeman_benchmark_cluster/report/`. The remaining 30 FreeMan
  subjects and any marker-based validation are still pending.

These items remain limitations; they are not inferred from the available runs.
