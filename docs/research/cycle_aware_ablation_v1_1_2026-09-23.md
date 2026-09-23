# Cycle-aware v1.1: ablation on the comparison protocol (private data, 2026-09-23)

Supersedes `cycle_aware_architecture_ablation_2026-09-20.md`, which measured
the v1 architecture under the old protocol. Everything here is the v1.1
architecture (loss v3), the 5 subject-disjoint folds, phase-normalised
windows, per-frame Procrustes-aligned error, and the **12 joints the external
comparison uses** (shoulders, elbows, wrists, hips, knees, ankles), so an
ablation delta can be read against the method gaps of
`strict_external_baselines_2026-09-21.md`.

Every row is produced by the same evaluator as the comparison table:

```bash
python -m fusion external-published model --dataset gymnastics \
    --run local/runs/cycle_aware/gymnastics_v1_1_<variant>_5fold_seed0 --joints comparison12
```

which on all 20 joints reproduces each sweep's own `test/pa_mpjpe`.

## Result (137 participants; "better" = the variant beats the full model on that participant)

| Variant | PA-MPJPE (mm) | Δ vs full | participants better | p (Holm) |
|---|---:|---:|---:|---:|
| **Full model (v1.1)** | **19.3 ± 1.0** | — | — | — |
| − residual head | 18.1 ± 1.0 | **+1.20** | 121/137 | 6.2e-21 |
| − learned reliability (equal weights) | 18.2 ± 1.1 | **+1.19** | 114/137 | 3.3e-18 |
| − FiLM conditioning | 18.4 ± 1.1 | +0.95 | 96/137 | 3.7e-11 |
| − all motion branches and FiLM (pose only) | 18.4 ± 1.1 | +0.91 | 99/137 | 4.7e-11 |
| − long-motion branch | 18.7 ± 1.0 | +0.60 | 93/137 | 6.0e-08 |
| − cross-view attention | 19.4 ± 1.0 | −0.04 | 62/137 | 0.45 |
| − phase encoding | 19.4 ± 0.9 | −0.04 | 50/137 | 0.15 |
| − short-motion branch | 19.4 ± 1.0 | −0.05 | 53/137 | 0.15 |
| *seed 1 of the full model* | 19.1 ± 1.0 | +0.18 | 85/137 | (seed, not an ablation) |
| *seed 2 of the full model* | 19.1 ± 1.0 | +0.26 | 82/137 | (seed, not an ablation) |

Positive Δ means the **variant is better**, i.e. removing that component
lowers the error. Statistics: Wilcoxon signed-rank over the 137 participants
(each is a held-out measurement of both variants), Holm-corrected across the
eight ablations; 10 000-sample bootstrap CIs are in
`local/runs/external_published/model/*/summary_model_12joints.json` and the
comparison JSONs.

## Reading

1. **No learned component improves accuracy on clean inputs.** Three of the
   eight removals are significant improvements (residual +1.20, reliability
   +1.19, FiLM +0.95 mm) and the rest are indistinguishable from the full
   model. The cheapest configuration on these joints is the reliability-free,
   residual-free variant, i.e. the closed-form depth-aware average.
2. **That reproduces the v1 finding** (`..._2026-09-20.md`): on clean data the
   reliability head outputs equal weights and the residual has no target that
   differs from the two-view average, so both only add variance; under
   synthetic corruption the same two components were the *only* ones that
   recovered anything (7.9 vs 13.6 mm). The architecture is validated as a
   corruption denoiser, not as a clean-input accuracy gain.
3. **Joint set changes the verdict for cross-view attention.** On the 20
   major joints removing it costs 0.71 mm (18.35 → 19.06, 5/5 folds); on the
   12 comparison joints the difference vanishes (−0.04 mm, p = 0.45). The 8
   joints that differ are the feet (toes, heels) plus nose and neck, so
   whatever cross-view attention contributes, it contributes on the
   extremities the external methods do not predict. Worth a per-joint
   breakdown before claiming it as a component win.
4. **Seed spread is smaller than every significant ablation** (0.18-0.26 mm
   across seeds 0-2 versus 0.60-1.20 mm), so the deltas above are not seed
   noise; the fold spread (±1.0 mm) is by far the largest source of variation
   and is why the statistics are paired over participants.

## What is missing

* The `robustness` sweep (corruption-augmented training, 23.0 mm on 20
  joints) has no checkpoints on disk, so it could not be re-scored here; it
  needs a re-run if the paper reports it.
* No ablation on FreeMan: all rows are private-data only.
* The 2026-09-20 note's proposed next experiment (`recovery_target:
  reference`, i.e. a recovery target that differs from the two-view average,
  using FreeMan's multi-view reference and never the private pseudo-GT) is
  still not run. It is the only proposal on the table that could make the
  learned components pay off on clean data.
