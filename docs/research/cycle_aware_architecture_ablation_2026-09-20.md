# Cycle-aware model: architecture ablation (private data, 2026-09-20)

Protocol: 5-fold subject-disjoint cross-validation on the 137 private
participants (`configs/cycle_aware/folds/gymnastics`, cohort-stratified),
seed 0, 50 epochs, `mhr70_major` (20 joints), 128-sample windows (2 cycles x
64 samples), default losses (`recovery 1.0, periodicity 0.1, symmetry 0.1,
residual 0.01`). One preset switched off per sweep
(`configs/cycle_aware/experiment/*.yaml`); everything else identical to the
baseline sweep `gymnastics_v1_5fold_seed0`. Each fold is one gpu job of about
6 minutes. Table built by `python -m gymnastics.fusion.ablation_table`
(`local/runs/cycle_aware/ablation_gymnastics_seed0.{md,json}`).

Two evaluation conditions, because the learned components behave differently
on them:

* **clean**: held-out test fold, unmodified inputs, per-frame Procrustes-aligned
  MPJPE against the triangulated pseudo-reference. This is the accuracy
  question. Values are canonical units x 500 (one torso length ~ 0.5 m), so
  they are approximate millimetres and are **not** comparable with the paper's
  60 mm protocol (one Sim3 per cycle plus hip centring, 70 joints).
* **corrupted**: validation fold with the fixed replayed corruption; error of
  damaged joints against the clean pseudo-target. This is the robustness
  question.

| preset | clean PA fused | clean PA base | fused vs base | corrupted err fused | corrupted err base | fused vs base | clean vs full model (paired folds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full model | 13.78 ± 0.66 | 13.74 | +0.3% | 7.92 ± 0.41 | 14.07 | −43.7% | — |
| no short-motion branch | 13.78 | 13.75 | +0.2% | 8.15 | 13.94 | −41.5% | −0.00 (2/5 worse) |
| no long-motion branch | 13.78 | 13.74 | +0.3% | 8.06 | 13.88 | −42.0% | −0.00 (3/5 worse) |
| no phase encoding | 13.80 | 13.76 | +0.3% | 8.07 | 14.11 | −42.8% | +0.03 (4/5 worse) |
| no FiLM | 13.86 | 13.74 | +0.9% | 9.34 | 13.48 | −30.7% | +0.08 (5/5 worse) |
| no cross-view attention | 13.76 | 13.71 | +0.4% | 8.85 | 13.87 | −36.2% | −0.02 (1/5 worse) |
| equal reliability (no learned weights) | 13.83 | 13.73 | +0.8% | 7.69 | 17.65 | −56.4% | +0.05 (4/5 worse) |
| no residual | **13.60** | 13.60 | 0 | 13.61 | 13.61 | 0 | **−0.18 (0/5 worse)** |
| pose only (no motion branches, no FiLM) | 13.82 | 13.75 | +0.6% | 9.18 | 13.69 | −33.0% | +0.04 (4/5 worse) |

Fold-to-fold SD of the clean metric is 0.6–0.7, so clean differences below
about 0.1 are noise; the paired sign counts are the more useful column.

## Findings

1. **On clean data every variant equals the reliability-weighted average.**
   The residual never helps on real inputs: in all eight sweeps with a
   residual the fused pose is 0.2–0.9 % *worse* than its own base pose, and
   the best clean number is the variant without a residual (13.60, better
   than the full model on 5/5 folds by 1.3 %). Test-fold weight entropy is
   0.69 = ln 2 everywhere: on clean inputs the reliability head outputs equal
   weights. This reproduces the rotation-aware result (learned == average).

2. **Under synthetic corruption the residual is the only component that
   recovers anything.** Removing it takes the corrupted error from 7.9 to
   13.6 (= base, no recovery at all); every other ablation keeps most of the
   43.7 % reduction.

3. **Learned reliability does act under corruption**, but only on the base
   pose: with equal weights the base error is 17.65 versus 14.07 with learned
   weights (weights shift away from the damaged view), yet the residual then
   compensates fully (7.69, even lower than the full model). So reliability
   and residual are redundant for corruption recovery, and neither transfers
   to clean data.

4. **Among the encoder pieces FiLM matters most for recovery** (−43.7 % →
   −30.7 %), then cross-view attention (→ −36.2 %). Short-motion, long-motion
   and phase encoding contribute 1–2 points each; removing all motion
   branches plus FiLM (`pose_only`) lands at −33 %, i.e. most of the recovery
   comes from the spatial branch plus the residual head.

## What this means for the model

* The architecture is currently validated only as a *synthetic-corruption
  denoiser*. Nothing in the encoder converts into accuracy on real inputs,
  because the recovery target on clean views is the two-view average and the
  clean-test reference shares the views' errors.
* The cheapest honest configuration for accuracy is `no_residual`
  (reliability-weighted average, 13.60), and even that is statistically tied
  with the plain average.
* To make the ablation meaningful for accuracy, the training signal has to
  differ from the average on real data: the `recovery_target: reference`
  option being added (FreeMan multi-view / Unity references, never the
  private pseudo-reference) is the right next experiment, together with
  evaluation under the paper protocol (export to `fused_sequence.npz` +
  `gymnastics fuse rotation-aware evaluate`) so the numbers become comparable
  with Table 1 and the reliability-weighted classical baseline (57.4 mm).

## Reproduction

```bash
bash pegasus/submit_cycle_aware_5fold.sh gymnastics                    # baseline
for e in no_short_motion no_long_motion no_phase no_film no_cross_view equal_reliability no_residual pose_only; do
  bash pegasus/submit_cycle_aware_5fold.sh gymnastics $e               # or ACCOUNT=HP260146 for the gen_S queue
done
PYTHONPATH=src python -m gymnastics.fusion.ablation_table \
  --baseline local/runs/cycle_aware/gymnastics_v1_5fold_seed0 \
  local/runs/cycle_aware/gymnastics_v1_{no_short_motion,no_long_motion,no_phase,no_film,no_cross_view,equal_reliability,no_residual,pose_only}_5fold_seed0 \
  --scale-mm 500 --out local/runs/cycle_aware/ablation_gymnastics_seed0
```

Jobs ran between 22:05 and 00:40 JST on 2026-09-19/20; the last 17 folds
were moved from the SKIING/gpu queue to the HP260146/gen_S queue because the
gpu queue was saturated by other users.
