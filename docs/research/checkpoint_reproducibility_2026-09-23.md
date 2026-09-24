# The reported weights were never saved (2026-09-23)

## Symptom

Scoring a finished sweep's checkpoints through the external-baseline evaluator
(`python -m fusion external-published model --run <sweep>`) reproduced the
sweep's own test numbers exactly for every label-free run, but not for the
reference-supervised ones: Fit3D `v1_1_refsup` came out at 26.6 mm where the
sweep reported 24.84 mm (fold_01: 24.0 vs 22.8).

## Cause

Two facts combine:

1. `run_fold` calls `trainer.test(module, datamodule=...)` on the module in
   memory, i.e. **the last epoch's weights**, and never passes a `ckpt_path`.
2. `ModelCheckpoint(monitor="val/total", save_top_k=1, save_last=True)` does
   **not** keep the last epoch. This Lightning version refreshes `last.ckpt`
   only when a monitored save actually happened in the same step
   (`ModelCheckpoint.on_validation_end`: `if self._last_global_step_saved ==
   trainer.global_step ...`), so with `save_top_k=1` the file is a copy of the
   best-`val/total` checkpoint — identical bytes, identical mtime.

So the weights every table reports were never written to disk. The gap is
invisible whenever validation improves until the end (label-free runs: best
epoch 44-49 of 50) and real when it does not. Reference-supervised runs stop
improving early (best epoch 5-41) because **validation replays the corruption**
(`data.validate_with_corruption: true`), so `val/total` selects for robustness,
not for clean accuracy.

A probe settles it: the best checkpoint of Fit3D `refsup` fold_01 scored
through the sweep's own code path (`test_only=true checkpoint=...`) gives
23.97 mm — the evaluator's 24.0, not the sweep's 22.8. The evaluator and the
sweep agree; the weights differed.

## Fix

`build_trainer` adds a second, unmonitored checkpoint
(`filename="final", monitor=None, every_n_epochs=1, save_top_k=1`), which
Lightning overwrites every epoch, so `final.ckpt` is the last epoch — the
weights the run reports. `last.ckpt` (best `val/total`) is kept.
Regression test: `tests/fusion/test_lightning_hydra.py::
test_trainer_saves_the_reported_final_epoch_weights`.

## Re-runs (determinism check)

The six reference-supervised sweeps were re-run with the new callback (same
seed, same data, same code otherwise) and their `final.ckpt` copied back into
the original sweep directories. Five of six reproduced the original numbers
**per fold, to the last reported digit**:

| Sweep | reported | re-run | best epoch (`last.ckpt`) |
|---|---:|---:|---|
| `fit3d_v1_1_refsup` | 24.84 | 24.84 | 12-35 |
| `freeman_rep_v1_1_refsup` | 38.00 | 38.00 | 15-41 |
| `freeman_all40_external_tcn_refsup` | 40.12 | 40.12 | 11-16 |
| `freeman_all40_external_metapose_mlp_refsup` | 39.72 | 39.72 | 5-21 |
| `freeman_all40_external_smoothnet_refsup` | 39.91 | 39.91 | 39-45 |
| `freeman_all40_external_muc_weights_refsup` | 42.86 | 42.97 | 22-45 |

`muc_weights` is the only sweep with a reliability cross-entropy
(`loss.reliability.weight=0.02`); its softmax path is not bit-deterministic on
the GPU. Its `result.json` is the original run (42.86) while the copied
`final.ckpt` scores 42.97 — the one row where checkpoint and table differ, by
0.11 mm. Everything else in the repository is now checkpoint-reproducible.

## What did not change

No reported number moves: every table was already "last epoch, no model
selection", which stays the convention. Label-free rows, ablations, the
deterministic matrix and the strict external baselines (which use each
release's own selection rule, documented in
`strict_external_baselines_2026-09-21.md`) are unaffected.

## Open

`val/total` is computed under corruption replay, so "best validation" means
"most robust", which is not what a reader assumes. Either log a clean
`val/pa_mpjpe` next to it or validate without corruption if any future work
wants to select models on validation at all.
